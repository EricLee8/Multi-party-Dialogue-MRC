import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import XLNetModel, XLNetConfig, XLNetPreTrainedModel, XLNetTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import BertLayer
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from utils.config import *
from utils.utils_split import convert_index_to_text, to_list


_PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_log_prob", "end_log_prob"])


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetModel, XLNetPreTrainedModel, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'xlnet': 'transformer', 'electra': 'electra'}
CLS_INDEXES = {'bert': 0, 'xlnet': -1, 'electra': 0}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MRCModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.question_start = args.max_length - args.question_max_length
        self.speaker_mha_layers = args.mha_layer_num
        if args.model_type == 'xlnet': self.question_start -= 1

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'xlnet':
            self.transformer = XLNetModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.attn_fct = nn.Linear(config.hidden_size * 3, 1)
        self.utter_filter = nn.Linear(config.hidden_size*4, 1)
        self.speaker_detector = nn.Linear(config.hidden_size*4, 1)
        for i in range(self.speaker_mha_layers):
            mha = BertLayer(config)
            self.add_module("MHA_{}".format(str(i)), mha)
        self.fusion_fct = nn.Sequential(
            nn.Linear(config.hidden_size*4, config.hidden_size),
            nn.Tanh()
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        p_mask=None,
        context=None,
        utterance_ids_dict=None,
        speaker_ids_dict=None,
        offset_mapping=None,
        qid=None,
        start_pos=None,
        end_pos=None,
        output_attentions=False
    ):
        utterance_gather_ids = utterance_ids_dict['utterance_gather_ids']
        utterance_p_mask = utterance_ids_dict['utterance_p_mask']
        utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
        key_utterance_target = utterance_ids_dict['key_utterance_target']
        speaker_attn_mask = speaker_ids_dict['speaker_attn_mask']
        speaker_gather_ids = utterance_ids_dict['utterance_gather_ids']
        target_speaker_gather_id = speaker_ids_dict['target_speaker_gather_id']
        speaker_target = speaker_ids_dict['speaker_target']
        speaker_target_mask = speaker_ids_dict['speaker_target_mask']

        training = start_pos is not None and end_pos is not None
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        hidden_states = transformer_outputs[0] # (bsz, seqlen, hsz)
        speaker_hidden_states = transformer_outputs[1 if args.model_type=='electra' else 2][-(self.speaker_mha_layers+1)] # (bsz, seqlen, hsz)
        bsz, slen, hsz = hidden_states.size()

        span_loss_fct = CrossEntropyLoss()
        speaker_loss_fct = nn.BCEWithLogitsLoss() if args.model_type=='electra' else nn.BCEWithLogitsLoss(reduction='sum')
        utter_loss_fct = CrossEntropyLoss(ignore_index=128)

        # deal with speaker information
        speaker_attentions = []
        hidden_states_detached = speaker_hidden_states.detach() # (bsz, slen, hsz)
        speaker_attn_mask[:, self.question_start:] = 0 # (bsz, slen)
        speaker_attn_mask = (1 - speaker_attn_mask) * -1e30
        speaker_attn_mask = speaker_attn_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.config.num_attention_heads, slen, -1) # (bsz, n_heads, slen, slen)
        attn_mask_expand = attention_mask
        attn_mask_expand[:, self.question_start:] = 0
        attn_mask_expand = (1 - attn_mask_expand) * -1e30
        attn_mask_expand = attn_mask_expand.unsqueeze(1).unsqueeze(1).expand(-1, self.config.num_attention_heads, slen, -1)
        
        mha_outs = getattr(self, "MHA_0")(hidden_states_detached, attention_mask=speaker_attn_mask if training else attn_mask_expand, output_attentions=output_attentions)
        speaker_mha_out = mha_outs[0] # (bsz, slen, hsz)
        if output_attentions: speaker_attentions.append(mha_outs[1])
        for i in range(1, self.speaker_mha_layers):
            mha_outs = getattr(self, "MHA_{}".format(str(i)))(speaker_mha_out, speaker_attn_mask if training else attn_mask_expand, output_attentions=output_attentions)
            speaker_mha_out = mha_outs[0]
            if output_attentions: speaker_attentions.append(mha_outs[1])
        speaker_embs = speaker_mha_out.gather(dim=1, index=speaker_gather_ids.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, max_utterm hsz)
        masked_speaker_embs = speaker_embs.gather(dim=1, index=target_speaker_gather_id.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, 1, hsz)
        masked_speaker_embs_expand = masked_speaker_embs.expand_as(speaker_embs) # (bsz, max_utter, hsz)
        speaker_logits = self.speaker_detector(
            torch.cat([speaker_embs, masked_speaker_embs_expand,\
                        speaker_embs*masked_speaker_embs_expand, speaker_embs-masked_speaker_embs_expand], dim=-1)
            ).squeeze(-1) # (bsz, max_utter)
        speaker_logits = speaker_logits * speaker_target_mask - 1e30 * (1-speaker_target_mask)

        # fuse information
        fused_hidden_states = self.fusion_fct(
            torch.cat([hidden_states, speaker_mha_out, hidden_states*speaker_mha_out, hidden_states-speaker_mha_out], dim=-1)
        )
        question_emb = torch.mean(fused_hidden_states[:, self.question_start:, :], dim=1) # (bsz, hsz)

        # deal with start position
        start_logits = self.start_predictor(fused_hidden_states, question_emb, p_mask=p_mask) # (bsz, seqlen)

        # deal with utterance prediction
        utter_embs = fused_hidden_states.gather(dim=1, index=utterance_gather_ids.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, max_utter, hsz)
        question_emb_expand = question_emb.unsqueeze(1).expand_as(utter_embs)
        utter_logits = self.utter_filter(torch.cat(
            [utter_embs, question_emb_expand, utter_embs*question_emb_expand, utter_embs-question_emb_expand], dim=-1)
        ).squeeze(-1) # (bsz, max_utter)
        utter_logits = utter_logits * utterance_p_mask - 1e30 * (1-utterance_p_mask)
        utter_weights = torch.softmax(utter_logits, dim=-1)
        utter_weights_repeated = utter_weights.view(-1).repeat_interleave(utterance_repeat_num.view(-1)).view(bsz, -1) # (bsz, slen)

        if training:
            end_logits = self.end_predictor(fused_hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            utter_loss = utter_loss_fct(utter_logits, key_utterance_target)
            speaker_loss = speaker_loss_fct(speaker_logits, speaker_target) * 10 if args.model_type=='electra' else speaker_loss_fct(speaker_logits, speaker_target) / bsz
            total_loss = span_loss + utter_loss + speaker_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None

            # calculate speaker prediction acc
            speaker_index = self.sigmoid(speaker_logits) > 0.5 # (bsz)
            correct_num = ((speaker_index==speaker_target.long())==speaker_target_mask).sum().item()
            all_num = speaker_target_mask.sum().item()

            # calculate UM from utterance prediction task
            model_pred_uids = to_list(torch.max(utter_weights, dim=-1).indices) # (bsz)

            # get answer span through beam search
            start_log_probs = F.softmax(start_logits, dim=-1) * utter_weights_repeated  # shape (bsz, slen)
            max_start_index, max_start_value = to_list(torch.max(start_log_probs, dim=-1).indices)[0], to_list(torch.max(start_log_probs, dim=-1).values)[0]
            coresponding_utter_value = to_list(utter_weights_repeated)[0][max_start_index]
            # print(max_start_value, coresponding_utter_value)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(fused_hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            fused_hidden_states_expanded = fused_hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(fused_hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1) * utter_weights_repeated.unsqueeze(-1).expand_as(end_logits) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            
            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)
            
            answer_list = []
            utterance_indices = torch.tensor([i for i in range(128 if not args.draw else 16)]).unsqueeze(0).expand(bsz, -1) # (bsz, utter_num)
            utteracne_id_expand = utterance_indices.reshape(-1).repeat_interleave(utterance_repeat_num.view(-1).cpu()).view(bsz, -1) # (bsz, slen)
            utteracne_id_expand = to_list(utteracne_id_expand)
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '') # in case the long sentence answer
                model_pred_uid = model_pred_uids[bidx]
                answer_list.append((qid[bidx],\
                     {"answer_text": best_text, "prob": best_prob, "span_pred_uid": span_pred_uid, "model_pred_uid": model_pred_uid}))

        outputs = (total_loss, span_loss, utter_loss, speaker_loss) if training else [answer_list, (correct_num, all_num)]
        if not training and output_attentions:
            outputs.append(
                {'common_attentions': transformer_outputs[2 if args.model_type=='electra' else 3],\
                 'speaker_attentions': speaker_attentions
                }
            )
        return outputs
    
    def load_mha_params(self):
        for i in range(self.speaker_mha_layers):
            mha = getattr(self, "MHA_{}".format(str(i)))
            rtv = mha.load_state_dict(getattr(self, self.transformer_name).encoder.layer[i-self.speaker_mha_layers].state_dict().copy())
            print(rtv)


class PoolerStartLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU())
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor,
        question_emb: torch.FloatTensor, # (bsz, hsz)
        p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        question_emb = question_emb.unsqueeze(1).expand_as(hidden_states)
        x = self.fusion(torch.cat([hidden_states, question_emb, hidden_states*question_emb], dim=-1))
        x = self.dense(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1-p_mask)
            else:
                x = x * p_mask - 1e30 * (1-p_mask)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1-p_mask)
            else:
                x = x * p_mask - 1e30 * (1-p_mask)

        return x
