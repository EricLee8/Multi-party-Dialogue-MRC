import os
import re
import torch
import json
import string
import collections
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from collections import deque
from .config import *


class Example(object):
    def __init__(self, context, utterances, relations, question, qid, ori_start_pos=None, ori_end_pos=None, answer=None):
        self.context = context
        self.utterances = utterances
        self.relations = relations
        self.question = question
        self.qid = qid
        self.ori_start_pos = ori_start_pos
        self.ori_end_pos = ori_end_pos
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "context: " + self.context + '\n'
        s += "utterances: " + self.utterances + '\n'
        s += "relations: " + self.relations + '\n'
        s += "question: " + self.question + '\n'
        s += "qid: " + self.qid + '\n'
        s += "answer: " + self.answer
        return s


class InputFeature(object):
    def __init__(self, qid, input_ids, token_type_ids, attention_mask, p_mask, offset_mapping,\
         context, utterance_ids_dict, speaker_ids_dict, start_pos=None, end_pos=None, is_impossible=None):
        self.qid = qid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.p_mask = p_mask
        self.offset_mapping = offset_mapping
        self.context = context
        self.utterance_ids_dict = utterance_ids_dict
        self.speaker_ids_dict = speaker_ids_dict
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.is_impossible = is_impossible


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(self.features[index].token_type_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[index].p_mask, dtype=torch.long)
        data_info['offset_mapping'] = self.features[index].offset_mapping
        data_info['context'] = self.features[index].context
        data_info['utterance_ids_dict'] = self.features[index].utterance_ids_dict
        data_info['speaker_ids_dict'] = self.features[index].speaker_ids_dict
        data_info['start_pos'] = torch.tensor(self.features[index].start_pos, dtype=torch.long) if\
            self.features[index].start_pos is not None else None
        data_info['end_pos'] = torch.tensor(self.features[index].end_pos, dtype=torch.long) if\
            self.features[index].end_pos is not None else None
        data_info['is_impossible'] = torch.tensor(self.features[index].is_impossible, dtype=torch.float) if\
            self.features[index].is_impossible is not None else None
        return data_info
    
    def __len__(self):
        return len(self.features)


def _cuda(x):
    if USE_CUDA:
        return x.cuda(device="cuda:"+str(args.cuda))
    else:
        return x


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def convert_index_to_text(offset_mapping, orig_text, start_index, end_index):
    orig_start_idx = offset_mapping[start_index][0]
    orig_end_idx = offset_mapping[end_index][1]
    return orig_text[orig_start_idx : orig_end_idx]


# in some cases the model will extract long sentence whose first tokens equals to the last tokens
def clean_answer(s):
    def _get_max_matched_str(tlist):
        for length in range(1, len(tlist)):
            if s[:length] == s[-length:]:
                return length
        return -1

    token_list = s.split(' ')
    if len(token_list) > 20:
        max_length = _get_max_matched_str(token_list)
        if max_length == -1:
            rtv = s
        else:
            rtv = " ".join(token_list[:max_length])
        return rtv
    return s


def collate_fn(data):
    data_info = {}
    float_type_keys = ['speaker_target']
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = _cuda(torch.stack(data_info[k]))
        if isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                id_key_list = [torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float) for sub_dict in data_info[k]] # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = _cuda(id_key_tensor)
            data_info[k] = new_dict
    return data_info


def read_examples(input_file, training=True):
    examples = []
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']['dialogues']

    for dialogue in tqdm(input_data):
        context = dialogue['context']
        utterances = dialogue['edus']
        relations = dialogue['relations']
        for qa in dialogue['qas']:
            question = qa['question']
            qid = qa['id']
            if not training: # during inference
                exp = Example(context, utterances, relations, question, qid)
                examples.append(exp)
                continue
            if qa['is_impossible'] or len(qa['answers']) == 0:
                exp = Example(context, utterances, relations, question, qid, -1, -1, '')
                examples.append(exp)
                continue

            for answer in qa['answers']: # during training
                ans_text = answer['text']
                ori_start_pos = answer['answer_start']
                ori_end_pos = ori_start_pos + len(ans_text)
                exp = Example(context, utterances, relations, question, qid, ori_start_pos, ori_end_pos, ans_text)
                examples.append(exp)
    if args.debug:
        examples = examples[:2000] if training else examples[:400]
    if args.small:
        examples = examples[:100] if training else examples[:100]
    return examples


def convert_examples_to_features(examples, tokenizer, speaker_mask_path, max_length, training=True, max_utterance_num=14):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_target_span(target_ids:list, input_ids:list, id_list=None, use_rfind=False):
        if id_list is None:
            id_list = [i for i in range(len(input_ids))]
        span_start_index, span_end_index = args.max_length-1, args.max_length-1
        for idx in range(len(id_list)): # sometimes id_list will exceed the length of input_ids
            id_list[idx] = min(len(input_ids)-1, id_list[idx])
            id_list[idx] = max(0, id_list[idx])
        id_list = list(set(id_list)) # get rid of redundent ids
        for idx in id_list if not use_rfind else id_list[::-1]:
            is_found = False
            if input_ids[idx] == target_ids[0]:
                is_found = True
                for offset in range(1, len(target_ids)):
                    if idx+offset > len(input_ids)-1: # out of range
                        is_found = False
                        break
                    if input_ids[idx+offset] != target_ids[offset]:
                        is_found = False
                        break
                if is_found:
                    span_start_index, span_end_index = idx, idx+len(target_ids)-1
                    break
        span = (span_start_index, span_end_index)
        return span
    
    def _get_utterance_gather_ids(input_ids, utterance_num):
        assert utterance_num <= max_utterance_num
        pad_utter_num = max_utterance_num - utterance_num
        utterance_gather_ids = []

        for idx, token_id in enumerate(input_ids):
            if token_id == tokenizer.sep_token_id:
                utterance_gather_ids.append(idx)
            if len(utterance_gather_ids) == utterance_num: break
        assert len(utterance_gather_ids) == utterance_num, "{}, {},{}".format(\
            str(utterance_num), str(len(utterance_gather_ids)), tokenizer.convert_ids_to_tokens(input_ids))

        utterance_p_mask = [1]*utterance_num + [0]*pad_utter_num
        repeat_num = [utterance_gather_ids[0]+1] + [utterance_gather_ids[i]-utterance_gather_ids[i-1] for i in range(1, utterance_num)]
        assert sum(repeat_num) == utterance_gather_ids[-1] + 1, "%d, %d"%(sum(repeat_num), utterance_gather_ids[-1]+1)
        remain_seq_length = args.max_length - utterance_gather_ids[-1] - 1
        if utterance_num == max_utterance_num:
            repeat_num[-1] += remain_seq_length
        else: # pad_utter_num >= 1
            num_for_each = remain_seq_length // pad_utter_num
            repeat_num += [num_for_each] * pad_utter_num
            remain_seq_length -= num_for_each * pad_utter_num
            if remain_seq_length > 0: repeat_num[-1] += remain_seq_length
        assert sum(repeat_num) == args.max_length

        if utterance_num < max_utterance_num:
            utterance_pad_id = 0 if args.model_type == 'xlnet' else 1
            utterance_gather_ids += [utterance_pad_id] * pad_utter_num
        assert len(utterance_gather_ids) == max_utterance_num
        return utterance_gather_ids, utterance_p_mask, repeat_num

    def _get_key_utterance_target(start_pos, end_pos, utterance_gather_ids):
        for idx, cur_utter_id in enumerate(utterance_gather_ids):
            if start_pos < cur_utter_id and end_pos < cur_utter_id:
                return idx
        return -1

    def _get_pos_after_tokenize(pos, offset_mapping, start=True):
        for idx, se in enumerate(offset_mapping):
            if se[0] == se[1] == 0: # skip pad token
                continue
            if pos==se[0] and start or pos==se[1] and not start:
                return idx
        return max_length-1

    print("Converting examples to features...")
    with open(speaker_mask_path, "r") as f:
        speaker_masks = json.load(f)
    max_tokens, max_answer_tokens, max_question_tokens = 0, 0, 0

    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id,\
         tokenizer.bos_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]

    total_num, unptr_num, too_long_num = len(examples), 0, 0
    features = []
    all_num, neg_num = 0, 0
    for exp in tqdm(examples):
        speaker_counts = {}
        for utterance_dict in exp.utterances:
            speaker = utterance_dict['speaker']
            if speaker not in speaker_counts.keys():
                speaker_counts[speaker] = 1
            else: 
                speaker_counts[speaker] += 1
        speaker_id_map = {name: idx+1 for idx, name in enumerate(speaker_counts.keys())}
        speaker_emb_ids = [speaker_id_map[utter_dict['speaker']] for utter_dict in exp.utterances]
        speaker_emb_ids += [0]*(max_utterance_num-len(exp.utterances)) # 0 for padding

        # use given mask to gaurantee reproducibility
        speaker_mask_dict = speaker_masks.get(exp.qid, {})
        if training and not exp.answer == '' and len(speaker_mask_dict) == 0:
            unptr_num += 1
            continue
        selected_speaker = speaker_masks[exp.qid]['speaker']
        speaker_mask_index = speaker_masks[exp.qid]['index']

        question = exp.question
        answer_text = exp.answer
        context = ''
        speaker_target, speaker_target_mask = [], [1]*len(exp.utterances) + [0]*(max_utterance_num-len(exp.utterances))
        mask_ori_start_pos = -1
        selected_speaker_index = 0

        for uidx, utterance_dict in enumerate(exp.utterances):
            text = utterance_dict['text']
            speaker = utterance_dict['speaker']
            all_num += 1
            if speaker == selected_speaker:
                selected_speaker_index += 1
                if selected_speaker_index == speaker_mask_index:
                    mask_ori_start_pos = len(context) # here should originally be + len(tokenizer.sep_token+' '), but later should be minus
                    speaker_target.append(0)
                    speaker_target_mask[uidx] = 0 # mask for self
                    target_speaker_gather_id = [uidx]
                    all_num -= 1
                else:
                    speaker_target.append(1)
            else:
                speaker_target.append(0)
                neg_num += 1
            context += tokenizer.sep_token + ' ' + speaker + ': ' + text + ' '
        speaker_target += [0]*(max_utterance_num-len(exp.utterances))

        context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
        if args.model_type == 'xlnet': context = context.lower()
        context = tokenizer.pad_token + ' ' + context
        mask_ori_start_pos += len(tokenizer.pad_token + ' ')
        mask_ori_end_pos = mask_ori_start_pos + len(selected_speaker)
        assert context[mask_ori_start_pos: mask_ori_end_pos] == selected_speaker

        context_max_length = args.max_length - args.question_max_length
        context_length = len(tokenizer.encode(context)) # including [CLS] and [SEP]
        remain_length = context_max_length - context_length
        context += ' '.join([tokenizer.pad_token] * remain_length)
        assert len(tokenizer.encode(context)) >= context_max_length

        question_length = len(tokenizer.encode(question)) - 1 # except the [CLS] and including the [SEP]
        if question_length > args.question_max_length:
            while len(tokenizer.encode(question)) - 1 > args.question_max_length:
                question = question[:-1]
        remain_length = args.question_max_length - question_length
        question += ' '.join([tokenizer.pad_token] * remain_length)
        question_length = len(tokenizer.encode(question)) - 1
        assert question_length == args.question_max_length, question_length

        ids_dict = tokenizer.encode_plus(context, question, padding='max_length',\
             truncation=True, max_length=max_length, return_offsets_mapping=True)
        offset_mapping = ids_dict['offset_mapping']
        input_ids = ids_dict['input_ids']
        token_type_ids = ids_dict['token_type_ids']
        attention_mask = ids_dict['attention_mask']
        for i in range(len(attention_mask)):
            if input_ids[i] == tokenizer.pad_token_id:
                attention_mask[i] = 0
        p_mask = [1] * len(input_ids)
        for i in range(len(input_ids)):
            if input_ids[i] in p_mask_ids or token_type_ids[i] == 1:
                p_mask[i] = 0
        text_len = len(tokenizer.encode(context + ' ' + tokenizer.sep_token + ' ' + question))
        if text_len > max_length: too_long_num += 1

        utterance_gather_ids, utterance_p_mask, utterance_repeat_num = _get_utterance_gather_ids(input_ids, len(exp.utterances))
        mask_start_pos = _get_pos_after_tokenize(mask_ori_start_pos, offset_mapping, True)
        mask_end_pos = _get_pos_after_tokenize(mask_ori_end_pos, offset_mapping, False)
        assert mask_start_pos != max_length-1 and mask_end_pos != max_length-1
        speaker_attn_mask = attention_mask.copy()
        for i in range(mask_start_pos, mask_end_pos+1):
            speaker_attn_mask[i] = 0
        s_from_ids = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[mask_start_pos: mask_end_pos+1]))
        assert s_from_ids.replace(' ', '') == selected_speaker.lower(), "{}, {}".format(s_from_ids.replace(' ', ''), selected_speaker.lower())

        utterance_ids_dict = {
            'utterance_gather_ids': utterance_gather_ids,
            'utterance_p_mask': utterance_p_mask,
            'utterance_repeat_num': utterance_repeat_num,
            'speaker_emb_ids': speaker_emb_ids
        }

        speaker_ids_dict = {
            'speaker_attn_mask': speaker_attn_mask,
            'target_speaker_gather_id': target_speaker_gather_id,
            'speaker_target': speaker_target,
            'speaker_target_mask': speaker_target_mask,
        }

        # inference
        if not training:
            utterance_ids_dict.update({'key_utterance_target': None})
            f_tmp = InputFeature(exp.qid, input_ids, token_type_ids, attention_mask, p_mask,\
                 offset_mapping, context, utterance_ids_dict, speaker_ids_dict)
            features.append(f_tmp)
            continue
        # training
        is_impossible = 1 if exp.answer == '' else 0
        start_pos, end_pos = _get_target_span(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_text)),\
             input_ids) if not is_impossible else (args.max_length-1, args.max_length-1)
        if not is_impossible and (start_pos==max_length-1 or end_pos==max_length-1):
            unptr_num += 1
            # print(exp.qid)
            continue

        key_utterance_target = _get_key_utterance_target(start_pos, end_pos, utterance_gather_ids) if not is_impossible else 14
        assert key_utterance_target != -1, "qid: {}, start: {}, end: {}, utter_gather_ids: {}".format(exp.qid, start_pos, end_pos, utterance_gather_ids)
        utterance_ids_dict.update({'key_utterance_target': key_utterance_target})
        
        f_tmp = InputFeature(exp.qid, input_ids, token_type_ids, attention_mask, p_mask, offset_mapping,\
             context, utterance_ids_dict, speaker_ids_dict, start_pos, end_pos, is_impossible)
        features.append(f_tmp)
        max_tokens = max(max_tokens, text_len)
        max_answer_tokens = max(max_answer_tokens, len(tokenizer.encode(exp.answer)))
        max_question_tokens = max(max_question_tokens, len(tokenizer.encode(question))-1)

    print(neg_num / all_num)
    if training: print("max token length, max_answer_length, max_question_length: ", max_tokens, max_answer_tokens, max_question_tokens)
    return features, total_num, unptr_num, too_long_num


def get_dataset(input_file, save_path, tokenizer, max_length, training=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "dev", "test"]:
        if type_ in input_file:
            postfix = type_
            break
    speaker_mask_path = "data/speaker_mask_{}.json".format(postfix)
    example_path = os.path.join(save_path, "example_{}.cache".format(postfix))
    if not os.path.exists(example_path):
        examples = read_examples(input_file, training=training)
        if not args.colab:
            print("Examples saved to " + example_path)
            torch.save(examples, example_path)
    else:
        print("Read {}_examples from cache...".format(postfix))
        examples = torch.load(example_path)
    feature_path = os.path.join(save_path, "feature_{}.cache".format(postfix))
    if not os.path.exists(feature_path):
        features, _, _, _ = convert_examples_to_features(examples, tokenizer, speaker_mask_path, max_length, training=training)
        if not args.colab:
            print("Features saved to " + feature_path)
            torch.save(features, feature_path)
    else:
        print("Read {}_features from cache...".format(postfix))
        features = torch.load(feature_path)
    dataset = Dataset(features)
    return dataset

    
if __name__ == "__main__":
    input_file = "data/train.json"
    speaker_mask_path = "data/speaker_mask_train.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizerFast, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    all_examples = read_examples(input_file, training=True)
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    all_features, total_num, unptr_num, too_long_num = convert_examples_to_features(all_examples,\
         tokenizer, speaker_mask_path, max_length=args.max_length, training=True)
    print(total_num, unptr_num, (total_num-unptr_num) / total_num, too_long_num / total_num)

    # from transformers import XLNetTokenizerFast, ElectraTokenizerFast, BertTokenizerFast
    # from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    # # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
    # tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length, training=True)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    # for batch in tqdm(dataloader):
    #     pass
        # print(to_list(batch['utterance_ids_dict']['utterance_gather_ids'][0]))
        # print(to_list(batch['speaker_ids_dict']['speaker_target_mask'][0]))