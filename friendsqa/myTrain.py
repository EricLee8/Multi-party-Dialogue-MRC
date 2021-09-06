import os
import json
import torch
import numpy as np
import random
import warnings
from math import sqrt
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, XLNetTokenizerFast, ElectraTokenizerFast
from transformers import BertConfig, XLNetConfig, ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluate_v2 import main as evaluate_on_squad, EVAL_OPTS
from utils.config import *

from models import (baseline, our)
MRC_MODEL_LIST = [baseline, our]

from utils.utils_split import get_dataset, collate_fn,\
    clean_answer, _cuda


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast)
}


warnings.filterwarnings("ignore")
device = torch.device("cuda:"+str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, "friendsqa_trn.json")
eval_path = os.path.join(args.data_path, "friendsqa_dev.json")
test_path = os.path.join(args.data_path, "friendsqa_tst.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)

    patience_turns = 0
    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = t_total // (args.epochs*5)
    steps = 0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask'],
                      'p_mask': batch['p_mask'],
                      'utterance_ids_dict': batch['utterance_ids_dict'],
                      'start_pos': batch['start_pos'],
                      'end_pos': batch['end_pos']
                     }
            if args.add_speaker_mask:
                inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            if t_total is not None:
                scheduler.step()
            if len(outputs)==4 and args.add_speaker_mask:
                span_loss, utter_loss, speaker_loss = outputs[1].item(), outputs[2].item(), outputs[3].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,UL:%.3f,SpL:%.3f" \
                    %(loss.item(), span_loss, utter_loss, speaker_loss))
            elif len(outputs)==3 and not args.add_speaker_mask:
                span_loss, utter_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,UL:%.3f" \
                    %(loss.item(), span_loss, utter_loss))
            elif len(outputs)==3 and args.add_speaker_mask:
                span_loss, speaker_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,SpL:%.3f" \
                    %(loss.item(), span_loss, speaker_loss))
            else:
                span_loss = outputs[1].item()
                pbar.set_description("Loss:%.3f,SL:%.3f" \
                    %(loss.item(), span_loss))
            model.zero_grad()
            if steps != 0 and steps % logging_step == 0:
                print("Epoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate(model, eval_dataloader, tokenizer, is_test=False)
                test_result = evaluate(model, test_dataloader, tokenizer, is_test=True)
                print("Eval Result:", eval_result)
                print("Test Result:", test_result)
            steps += 1

    eval_result = evaluate(model, eval_dataloader, tokenizer, is_test=False)
    test_result = evaluate(model, test_dataloader, tokenizer, is_test=True)
    print("Eval Result:", eval_result)
    print("Test Result:", test_result)


def evaluate(model, eval_loader, tokenizer, is_test=False):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    answer_dict, na_dict = {}, {}
    correct_num, all_num = 0, 0
    target_file_path = 'data/' + ('tst' if is_test else 'dev') + '_uids_target.json'
    with open(target_file_path, "r") as f:
        target_uids_dict = json.load(f)

    for _, batch in pbar:
        cur_batch_size = len(batch['input_ids'])

        inputs = {'input_ids': batch['input_ids'],
                    'token_type_ids': batch['token_type_ids'],
                    'attention_mask': batch['attention_mask'],
                    'p_mask': batch['p_mask'],
                    'context': batch['context'],
                    'utterance_ids_dict': batch['utterance_ids_dict'],
                    'offset_mapping': batch['offset_mapping'],
                    'qid': batch['qid']
                 }
        if args.add_speaker_mask:
            inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
        outputs = model(**inputs)
        answer_list = outputs[0]
        if args.add_speaker_mask:
            b_correct_num, b_all_num = outputs[1]
            correct_num += b_correct_num
            all_num += b_all_num
        for qid, ans_record in answer_list:
            real_qid = qid.split('-')[0]
            offset = int(qid.split('-')[1])
            ans_record['span_pred_uid'] += offset
            if 'model_pred_uid' in ans_record.keys(): ans_record['model_pred_uid'] += offset
            if real_qid not in answer_dict.keys():
                answer_dict[real_qid] = ans_record
            else:
                cur_best_prob = answer_dict[real_qid]['prob']
                if ans_record['prob'] > cur_best_prob:
                    answer_dict[real_qid] = ans_record
    # computing utterance matching (UM)
    assert len(answer_dict) == len(target_uids_dict)
    all_example_num, model_pred_correct_num, span_um_num = len(answer_dict), 0, 0
    for qid, target_uids in target_uids_dict.items():
        ans_record = answer_dict[qid]
        span_um_num += 1 if ans_record['span_pred_uid'] in target_uids else 0
        if 'model_pred_uid' in ans_record.keys():
            model_pred_correct_num += 1 if ans_record['model_pred_uid'] in target_uids else 0
    model_um = model_pred_correct_num / all_example_num
    span_um = span_um_num / all_example_num

    # computing f1 and em using official SQuAD transcript
    answer_dict = {qid: ans_record['answer_text'] for qid, ans_record in answer_dict.items()}
    with open(args.pred_file, "w") as f:
        json.dump(answer_dict, f, indent=2)
    if args.add_speaker_mask:
        print("Speaker prediction acc: %.5f"%(correct_num/all_num))
    evaluate_options = EVAL_OPTS(data_file=test_path if is_test else eval_path,
                                 pred_file=args.pred_file,
                                 na_prob_file=None)
    res = evaluate_on_squad(evaluate_options)
    em = res['exact']
    f1 = res['f1']
    rtv_dict = {'em': em, 'f1': f1, 'um': span_um, 'model_um': model_um}
    model.train()

    return rtv_dict


if __name__ == "__main__":
    set_seed()
    MRCModel = MRC_MODEL_LIST[args.model_num].MRCModel
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    config = config_class.from_pretrained(args.model_name)
    if args.model_type != 'xlnet':
        config.start_n_top = 5
        config.end_n_top = 5

    # training
    train_dataset = get_dataset(train_path, args.cache_path,\
            tokenizer, args.max_length, training=True)
    eval_dataset = get_dataset(eval_path, args.cache_path,\
            tokenizer, args.max_length, training=False)
    test_dataset = get_dataset(test_path, args.cache_path,\
            tokenizer, args.max_length, training=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MRCModel.from_pretrained(args.model_name, config=config)
    if hasattr(model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.load_mha_params()
    model = model.to(device)

    train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)

