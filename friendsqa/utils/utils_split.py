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
    def __init__(self, context, utterances, relations, question, qid, ori_start_pos=None,\
         ori_end_pos=None, answer=None, key_uid=None):
        self.context = context
        self.utterances = utterances
        self.relations = relations
        self.question = question
        self.qid = qid
        self.ori_start_pos = ori_start_pos
        self.ori_end_pos = ori_end_pos
        self.answer = answer
        self.key_uid = key_uid

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
         context, utterance_ids_dict, speaker_ids_dict, start_pos=None, end_pos=None):
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


def read_examples(input_file, tokenizer, training=True):
    gather_token = tokenizer.cls_token if args.use_cls_for_gather else tokenizer.sep_token
    def _get_pruned_context(start_uid: int, utterances: list):
        max_context_length = args.max_length - args.question_max_length
        utterance_start_poses, speaker_start_poses = [], []
        context = ''
        for idx in range(start_uid, len(utterances)):
            utter_dict = utterances[idx]
            speaker = ' & '.join(utter_dict['speakers'])
            text = utter_dict['utterance']
            tmp_context = context + gather_token + ' ' + speaker + (': ' if args.model_type != 'xlnet' else ' : ') + text + ' '
            cur_len = len(tokenizer.encode(tmp_context.strip()[len(gather_token + ' '):])) + (2 if args.use_cls_for_gather else 1)
            # plus 1 for later adding pad token in the front, additional 1 for the final [CLS] token if we use cls as gather token
            if cur_len > max_context_length:
                return context, idx, utterance_start_poses, speaker_start_poses
            context += gather_token + ' '
            speaker_start_poses.append(len(context))
            context += speaker + (': ' if args.model_type != 'xlnet' else ' : ')
            utterance_start_poses.append(len(context))
            context += text + ' '
        return context, len(utterances), utterance_start_poses, speaker_start_poses

    examples = []
    target_examples = {}
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    
    max_utter_num, number_before_pruning = 0, 0
    for scene in tqdm(input_data):
        paragraphs = scene['paragraphs']
        for dialogue in paragraphs:
            relations = None
            utterances = dialogue['utterances:']
            max_utter_num = max(max_utter_num, len(utterances))
            for idx in range(len(utterances)):
                utterances[idx]['utterance'] = utterances[idx]['utterance'].lower()
                for sidx in range(len(utterances[idx]['speakers'])):
                    utterances[idx]['speakers'][sidx] = utterances[idx]['speakers'][sidx].lower()
            for idx in range(len(dialogue['qas'])):
                dialogue['qas'][idx]['question'] = dialogue['qas'][idx]['question'].lower()
                for aidx in range(len(dialogue['qas'][idx]['answers'])):
                    dialogue['qas'][idx]['answers'][aidx]['answer_text'] = dialogue['qas'][idx]['answers'][aidx]['answer_text'].lower()
            
            idx = 0
            context_dict_list = []
            while idx < len(utterances):
                context, next_idx, utterance_start_poses, speaker_start_poses = _get_pruned_context(idx, utterances)
                context = context.strip()[len(gather_token + ' '):] # remove the first gather token and ' '
                context = tokenizer.pad_token + ' ' + context
                if args.use_cls_for_gather:
                    context += ' ' + gather_token
                modified_offset = len(tokenizer.pad_token + ' ') - len(gather_token + ' ')
                utterance_start_poses = [pos + modified_offset for pos in utterance_start_poses]
                speaker_start_poses = [pos + modified_offset for pos in speaker_start_poses]
                context_dict_list.append(
                    {'start_uid': idx,
                     'end_uid': next_idx-1,
                     'context': context,
                     'utterance_start_poses': utterance_start_poses,
                     'speaker_start_poses': speaker_start_poses,
                     'utterances': utterances[idx: next_idx]
                    }
                )
                for ii in range(next_idx - idx):
                    a = context[utterance_start_poses[ii]: utterance_start_poses[ii]+len(utterances[ii+idx]['utterance'])]
                    b = utterances[ii+idx]['utterance']
                    assert a == b, "{} vs. {}".format(a, b)
                idx = next_idx
            assert idx == len(utterances)

            for qa in dialogue['qas']:
                question = qa['question']
                qid = qa['id']
                if not training: # during inference
                    number_before_pruning += 1
                    target_uids = [answer['utterance_id'] for answer in qa['answers']]
                    target_examples[qid] = target_uids

                    for context_dict in context_dict_list:
                        exp = Example(context_dict['context'], context_dict['utterances'], relations,\
                             question, qid+'-'+str(context_dict['start_uid']))
                        examples.append(exp)
                    continue

                for answer in qa['answers']: # during training
                    ans_text = answer['answer_text']
                    answer_utter_id = answer['utterance_id']
                    context_id = -1
                    for idx, context_dict in enumerate(context_dict_list):
                        if answer_utter_id in range(context_dict['start_uid'], context_dict['end_uid']+1):
                            context_id = idx
                    assert context_id != -1
                    context_dict = context_dict_list[context_id]
                    real_utterances = context_dict['utterances']
                    real_answer_utter_id = answer_utter_id - context_dict['start_uid']
                    corresponding_utter = real_utterances[real_answer_utter_id]['utterance']
                    assert corresponding_utter == utterances[answer_utter_id]['utterance']
                    if answer['is_speaker']:
                        ori_start_pos = context_dict['speaker_start_poses'][real_answer_utter_id]
                        ori_end_pos = ori_start_pos + len(ans_text)
                    else:
                        inner_start_pos = len(' '.join(corresponding_utter.split(' ')[:answer['inner_start']]) + ('' if answer['inner_start']==0 else ' ')) # get the token position in the corresponding utterance
                        utter_offset = context_dict['utterance_start_poses'][real_answer_utter_id]
                        ori_start_pos = utter_offset + inner_start_pos
                        ori_end_pos = ori_start_pos + len(ans_text)
                        tmp = ' '.join(corresponding_utter.split(' ')[answer['inner_start']: answer['inner_end']+1])
                        if tmp != ans_text:
                            print("Can't extract from context! Discard! {} vs. {}".format(tmp, ans_text))
                            continue
                    assert context_dict['context'][ori_start_pos: ori_end_pos] == ans_text,\
                        "##{}## vs. ##{}##".format(context[ori_start_pos: ori_end_pos], ans_text)
                    
                    exp = Example(context_dict['context'], real_utterances, relations, question, qid,\
                         ori_start_pos, ori_end_pos, ans_text, key_uid=real_answer_utter_id)
                    examples.append(exp)

    if not training:
        prefix = 'dev' if 'dev' in input_file else 'tst' if 'tst' in input_file else 'draw'
        with open(input_file.split('/')[0] + '/' + prefix + '_uids_target.json', "w") as wf:
            json.dump(target_examples, wf)
    print("Max utterance number: {}".format(max_utter_num))
    if not training:
        print("Total number of examples before pruning: {}".format(number_before_pruning))
    print("Total number of examples after pruning: {}".format(len(examples)))
    if args.debug:
        examples = examples[:4000] if training else examples[:400]
    if args.small:
        examples = examples[:2000] if training else examples[:400]
    return examples


def convert_examples_to_features(examples, tokenizer, max_length, speaker_mask_path, training=True, max_utterance_num=128):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_utterance_gather_ids(input_ids, utterance_num, gather_token_id, cur_max_length):
        assert utterance_num <= max_utterance_num
        pad_utter_num = max_utterance_num - utterance_num
        utterance_gather_ids = []
        skip_cls_flag = True if args.use_cls_for_gather else False

        for idx, token_id in enumerate(input_ids):
            if token_id == gather_token_id:
                if skip_cls_flag:
                    skip_cls_flag = False # only skip the first [CLS] if we use cls as gather token
                    continue
                utterance_gather_ids.append(idx)
            if len(utterance_gather_ids) == utterance_num: break
        assert len(utterance_gather_ids) == utterance_num, "{}, {},{}".format(\
            str(utterance_num), str(len(utterance_gather_ids)), tokenizer.convert_ids_to_tokens(input_ids))

        utterance_p_mask = [1]*utterance_num + [0]*pad_utter_num
        repeat_num = [utterance_gather_ids[0]+1] + [utterance_gather_ids[i]-utterance_gather_ids[i-1] for i in range(1, utterance_num)]
        assert sum(repeat_num) == utterance_gather_ids[-1] + 1, "%d, %d"%(sum(repeat_num), utterance_gather_ids[-1]+1)
        remain_seq_length = cur_max_length - utterance_gather_ids[-1] - 1
        if utterance_num == max_utterance_num:
            repeat_num[-1] += remain_seq_length
        else: # pad_utter_num >= 1
            num_for_each = remain_seq_length // pad_utter_num
            repeat_num += [num_for_each] * pad_utter_num
            remain_seq_length -= num_for_each * pad_utter_num
            if remain_seq_length > 0: repeat_num[-1] += remain_seq_length
        assert sum(repeat_num) == cur_max_length

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
    gather_token, gather_token_id = (tokenizer.cls_token, tokenizer.cls_token_id) if\
         args.use_cls_for_gather else (tokenizer.sep_token, tokenizer.sep_token_id)

    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id,\
         tokenizer.bos_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]

    total_num, unptr_num, too_long_num = len(examples), 0, 0
    features = []
    all_num, neg_num = 0, 0
    for fidx, exp in enumerate(tqdm(examples)):
        speaker_counts = {}
        for utter_dict in exp.utterances:
            speaker = ' & '.join(utter_dict['speakers'])
            if speaker not in speaker_counts.keys():
                speaker_counts[speaker] = 1
            else: 
                speaker_counts[speaker] += 1
        speaker_id_map = {name: idx+1 for idx, name in enumerate(speaker_counts.keys())}
        speaker_emb_ids = [speaker_id_map[' & '.join(utter_dict['speakers'])] for utter_dict in exp.utterances]
        speaker_emb_ids += [0]*(max_utterance_num-len(exp.utterances)) # 0 for padding

        # use given mask to gaurantee reproducibility
        selected_speaker = speaker_masks[exp.qid + "-{}".format(fidx) if training else exp.qid]['speaker']
        speaker_mask_index = speaker_masks[exp.qid + "-{}".format(fidx) if training else exp.qid]['index']

        question = exp.question
        answer_text = exp.answer
        context = ''
        speaker_target, speaker_target_mask = [], [1]*len(exp.utterances) + [0]*(max_utterance_num-len(exp.utterances))
        mask_ori_start_pos = -1
        selected_speaker_index = 0

        for uidx, utter_dict in enumerate(exp.utterances):
            text = utter_dict['utterance']
            speaker = ' & '.join(utter_dict['speakers'])
            all_num += 1
            if speaker == selected_speaker:
                selected_speaker_index += 1
                if selected_speaker_index == speaker_mask_index:
                    mask_ori_start_pos = len(context) # here should originally be + len(gather_token+' '), but later should be minus
                    speaker_target.append(0)
                    speaker_target_mask[uidx] = 0 # mask for self
                    target_speaker_gather_id = [uidx]
                    all_num -= 1
                else:
                    speaker_target.append(1)
            else:
                speaker_target.append(0)
                neg_num += 1
            context += gather_token + ' ' + speaker + (': ' if args.model_type != 'xlnet' else ' : ') + text + ' '
        speaker_target += [0]*(max_utterance_num-len(exp.utterances))

        context = context.strip()[len(gather_token)+1:] # remove the first sep token and ' '
        context = tokenizer.pad_token + ' ' + context
        if args.use_cls_for_gather:
            context += ' ' + gather_token
        assert context == exp.context
        mask_ori_start_pos += len(tokenizer.pad_token + ' ')
        mask_ori_end_pos = mask_ori_start_pos + len(selected_speaker)
        assert context[mask_ori_start_pos: mask_ori_end_pos] == selected_speaker, "@{}@ vs. @{}@, qid: {}, ori_s_before_pad: {}, context: {}".format(\
            context[mask_ori_start_pos: mask_ori_end_pos], selected_speaker, exp.qid, mask_ori_start_pos-len(tokenizer.pad_token+' '), context)

        # padding context
        if not args.draw:
            context_max_length = args.max_length - args.question_max_length
            context_length = len(tokenizer.encode(context)) # including [CLS] and [SEP]
            remain_length = context_max_length - context_length
            if args.model_type != 'xlnet':
                context += ' '.join([tokenizer.pad_token] * remain_length)
            else:
                context = ' '.join([tokenizer.pad_token] * remain_length) + context
                mask_ori_start_pos += len(' '.join([tokenizer.pad_token] * remain_length))
                mask_ori_end_pos += len(' '.join([tokenizer.pad_token] * remain_length))
                if training:
                    exp.ori_start_pos += len(' '.join([tokenizer.pad_token] * remain_length))
                    exp.ori_end_pos += len(' '.join([tokenizer.pad_token] * remain_length))
            assert len(tokenizer.encode(context)) >= context_max_length

        # padding question
        if not args.draw:
            question_length = len(tokenizer.encode(question)) - 1 # except the [CLS] and including the [SEP]
            if question_length > args.question_max_length:
                while len(tokenizer.encode(question)) - 1 > args.question_max_length:
                    question = question[:-1]
            remain_length = args.question_max_length - question_length
            if args.model_type != 'xlnet':
                question += ' '.join([tokenizer.pad_token] * remain_length)
            else:
                question = ' '.join([tokenizer.pad_token] * remain_length) + question
            question_length = len(tokenizer.encode(question)) - 1
            assert question_length == args.question_max_length, question_length

        if not args.draw:
            ids_dict = tokenizer.encode_plus(context, question, padding='max_length',\
                truncation=True, max_length=max_length, return_offsets_mapping=True)
        else:
            ids_dict = tokenizer.encode_plus(context, question, return_offsets_mapping=True)
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
        max_tokens = max(max_tokens, text_len)
        max_question_tokens = max(max_question_tokens, len(tokenizer.encode(question))-1)
        if training:
            max_answer_tokens = max(max_answer_tokens, len(tokenizer.encode(exp.answer)))
        if text_len > max_length:
            too_long_num += 1
            continue

        utterance_gather_ids, utterance_p_mask, utterance_repeat_num = _get_utterance_gather_ids(input_ids, len(exp.utterances),\
             gather_token_id, cur_max_length=max_length if not args.draw else len(input_ids))
        mask_start_pos = _get_pos_after_tokenize(mask_ori_start_pos, offset_mapping, True)
        mask_end_pos = _get_pos_after_tokenize(mask_ori_end_pos, offset_mapping, False)
        assert mask_start_pos != max_length-1 and mask_end_pos != max_length-1, "Speaker: {}, qid: {}".format(selected_speaker, exp.qid)
        speaker_attn_mask = attention_mask.copy()
        for i in range(mask_start_pos, mask_end_pos+1):
            speaker_attn_mask[i] = 0
        s_from_ids = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[mask_start_pos: mask_end_pos+1]))
        assert s_from_ids.replace(' ', '') == selected_speaker.replace(' ', ''), "{} vs. {}, start_pos: {}, end_pos: {}".format(\
            s_from_ids, selected_speaker, mask_start_pos, mask_end_pos)

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
        start_pos = _get_pos_after_tokenize(exp.ori_start_pos, offset_mapping, start=True)
        end_pos = _get_pos_after_tokenize(exp.ori_end_pos, offset_mapping, start=False)
        if start_pos==max_length-1 or end_pos==max_length-1:
            unptr_num += 1
            # print(exp.qid)
            continue

        key_utterance_target = _get_key_utterance_target(start_pos, end_pos, utterance_gather_ids)
        assert key_utterance_target != -1
        assert key_utterance_target == exp.key_uid
        utterance_ids_dict.update({'key_utterance_target': key_utterance_target})
        
        f_tmp = InputFeature(exp.qid, input_ids, token_type_ids, attention_mask, p_mask, offset_mapping,\
             context, utterance_ids_dict, speaker_ids_dict, start_pos, end_pos)
        features.append(f_tmp)

    print("Speaker negtive label rate: %.2f%%" % (neg_num / all_num * 100))
    print("Too long number: %d, too long rate: %.2f%%" %(too_long_num, too_long_num/total_num*100))
    if training:
        print("max_token_length: {}, max_answer_length: {}, max_question_length: {}".format(max_tokens, max_answer_tokens, max_question_tokens))
        print("Unpointable number: %d, unpointable rate: %.2f %%" %(unptr_num, unptr_num/total_num*100))
    else:
        print("max_token_length: {}, max_question_length: {}".format(max_tokens, max_question_tokens))
    return features


def get_dataset(input_file, cache_path, tokenizer, max_length, training=True):
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    postfix = ""
    for type_ in ["trn", "dev", "tst", "draw"]:
        if type_ in input_file:
            postfix = type_
            break
    speaker_mask_path = input_file.replace('friendsqa', 'speaker_mask')
    example_path = os.path.join(cache_path, "example_{}.cache".format(postfix))
    if not os.path.exists(example_path):
        examples = read_examples(input_file, tokenizer, training=training)
        if not args.colab:
            print("Examples saved to " + example_path)
            torch.save(examples, example_path)
    else:
        print("Read {}_examples from cache...".format(postfix))
        examples = torch.load(example_path)
    feature_path = os.path.join(cache_path, "feature_{}.cache".format(postfix))
    if not os.path.exists(feature_path):
        features = convert_examples_to_features(examples, tokenizer, max_length, speaker_mask_path, training=training, max_utterance_num=128 if not args.draw else 16)
        if not args.colab:
            print("Features saved to " + feature_path)
            torch.save(features, feature_path)
    else:
        print("Read {}_features from cache...".format(postfix))
        features = torch.load(feature_path)
    dataset = Dataset(features)
    return dataset

    
if __name__ == "__main__":
    input_file = "data/friendsqa_tst.json"
    speaker_mask_path = input_file.replace('friendsqa', 'speaker_mask')

    # from transformers import XLNetTokenizerFast, ElectraTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
    # # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    # # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    # tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # all_examples = read_examples(input_file, tokenizer, training=True)
    # all_features = convert_examples_to_features(all_examples,\
    #      tokenizer, args.max_length, speaker_mask_path, training=True)

    from transformers import XLNetTokenizerFast, ElectraTokenizerFast, BertTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length, training=False)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        pass
        # print(to_list(batch['utterance_ids_dict']['utterance_gather_ids'][0]))
        # print(to_list(batch['speaker_ids_dict']['speaker_target_mask'][0]))