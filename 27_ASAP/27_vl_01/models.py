# based on https://www.kaggle.com/code/tascj0/a-text-span-detector

from os.path import join as opj
import re
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from transformers import (AutoModelForTokenClassification, AutoModel, AutoTokenizer,
                          AutoConfig, AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from public_metric_2021 import score_feedback_comp


LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')
TYPE2LABEL = {t:l for l, t in enumerate(LABEL2TYPE)}
LABEL2TYPE = {l:t for t,l in TYPE2LABEL.items()}


def to_gpu(data):
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep


class TextSpanDetector(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_classes=7,
                ):
        super().__init__()
        self._current_epoch = 1
        self.num_labels = 1 + 2 + num_classes
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=1 + 2 + num_classes,
            local_files_only=False, #local_files_only
        )
            
    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        logits = self.model(input_ids=data['input_ids'],
                            attention_mask=data['attention_mask'])['logits']
        logits = aggregate_tokens_to_words(logits, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]
        return obj_pred, reg_pred, cls_pred

    def predict(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return dict(text_id=data['text_id'])

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(text_id=data['text_id'],
                    start=to_np(start),
                    end=to_np(end),
                    score=to_np(pr_score),
                    label=to_np(pr_label))
    
    def test_step(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        res = self.predict(data, test_score_thr)
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for c,start,end in zip(res['label'],res['start'],res['end']):
                pred.append([text_id, LABEL2TYPE[c], ' '.join(np.arange(start,end+1).astype(str))])
            pred_df = pd.DataFrame(pred, columns=['essay_id','class','predictionstring'])
        else:
            pred_df = pd.DataFrame([], columns=['essay_id','class','predictionstring'])
        return pred_df    

class DatasetTest(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.samples = sorted(self.df['essay_id'].unique())
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id = self.samples[index]
        text = self.df.loc[self.df['essay_id']==text_id,'essay'].values[0]

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        return dict(text=text,
                    text_id=text_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes)

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class TestCollator(object):
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
#         if self.attention_window is not None:
#             attention_window = self.attention_window
#             padded_length = (attention_window -
#                              max_seq_length % attention_window
#                              ) % attention_window + max_seq_length
#         else:
#             padded_length = max_seq_length
        padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes)