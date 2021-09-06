import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import XLNetModel, XLNetConfig, XLNetPreTrainedModel, XLNetTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
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
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.cls_index = CLS_INDEXES[args.model_type]

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'xlnet':
            self.transformer = XLNetModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        p_mask=None,
        context=None,
        utterance_ids_dict=None,
        offset_mapping=None,
        qid=None,
        start_pos=None,
        end_pos=None,
        output_attentions=False
    ):
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        hidden_states = transformer_outputs[0] # (bsz, seqlen, hsz)
        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        start_logits = self.start_predictor(hidden_states, p_mask=p_mask) # (bsz, seqlen)

        if training:
            end_logits = self.end_predictor(hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

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
            utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
            utterance_indices = torch.tensor([i for i in range(128)]).unsqueeze(0).expand(bsz, -1) # (bsz, utter_num)
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
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob, "span_pred_uid": span_pred_uid}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs


class PoolerStartLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor,
        p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        x = self.dense(hidden_states).squeeze(-1)
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
