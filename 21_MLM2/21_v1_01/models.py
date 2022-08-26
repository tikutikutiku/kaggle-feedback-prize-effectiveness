import torch

def to_gpu(data):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
import sys
# sys.path.append('../../../../../COCO-LM-main/huggingface')
# from cocolm.modeling_cocolm import COCOLMModel
# from cocolm.configuration_cocolm import COCOLMConfig
# from focal_loss import FocalLoss
from sift import hook_sift_layer, AdversarialLearner

from scipy.stats import pearsonr
def calc_pearson_corr(pred, target):
    score = pearsonr(pred, target)[0]
    return score

class LSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                             hidden_size=out_channels,
                             num_layers=num_layers,
                             dropout=p_drop,
                             batch_first=True, 
                             bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
class GRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.GRU(input_size=in_channels,
                           hidden_size=out_channels,
                           num_layers=num_layers,
                           dropout=p_drop,
                           batch_first=True, 
                           bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_layers=1, nhead=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=in_channels,nhead=nhead),
                                                 num_layers=num_layers)
    def forward(self, x):
        x = self.transformer(x)
        return x
    
    
import audtorch
class CCCLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        if self.reduction=='mean':
            return -1 * audtorch.metrics.functional.pearsonr(inputs, targets)
        elif self.reduction=='none':
            return -1 * torch.ones_like(targets) * audtorch.metrics.functional.pearsonr(inputs, targets)
        else:
            raise Exception()
            
            
class CustomLoss(nn.Module):
    def __init__(self, loss_1, loss_2, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        if loss_1=='smoothl1':
            self.loss_fn_1 = nn.SmoothL1Loss(reduction=reduction)
        if loss_2=='xentropy':
            self.loss_fn_2 = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(self, inputs_1, inputs_2, targets_1, targets_2):
        loss = 0.5*self.loss_fn_1(inputs_1, targets_1) + 0.5*self.loss_fn_2(inputs_2, targets_2)
        return loss
    
        
class Model(nn.Module):
    def __init__(self, 
                 model_name, 
                 tokenizer,
                 num_labels, 
                 hidden_dropout_prob=0, #0.1, 
                 learning_rate=1e-5,
                 head_learning_rate=1e-3,
                 num_train_steps=0,
                 p_drop=0, #0.5,
                 warmup_ratio = 0,
                 model_pretraining=None,
                 rnn='none',
                 loss='mse',
                 head='simple',
                 msd='false',
                 multi_layers=1,
                 aug='none',
                 mixup_alpha=1.0,
                 aug_stop_epoch=999,
                 p_aug=0.5,
                 adv_sift='false',
                 l2norm='false',
                 s=30,
                 **kwargs,
                ):
        super().__init__()
        self._current_epoch = 1
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_ratio = warmup_ratio 
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.loss = loss
        self.msd = msd
        self.multi_layers = multi_layers
        self.aug = aug
        self.mixup_alpha = mixup_alpha
        self.aug_stop_epoch = aug_stop_epoch
        self.p_aug = p_aug
        self.adv_sift = adv_sift
        self.l2norm = l2norm
        self.s = s
        
        if model_pretraining is not None:
            self.transformer = model_pretraining.transformer
            self.config = model_pretraining.config
        elif 'cocolm' in model_name:
            self.config = COCOLMConfig.from_pretrained(model_name)
            self.config.update(
                {
                    "output_hidden_states": False,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "add_pooling_layer": False,
                    "num_labels": self.num_labels,
                }
            )
            self.transformer = COCOLMModel.from_pretrained(model_name, config=self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.update(
                {
                    "output_hidden_states": True,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "add_pooling_layer": False,
                    "num_labels": self.num_labels,
                }
            )
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
            
        if 'cocolm' not in model_name:
            self.transformer.resize_token_embeddings(len(tokenizer))
        
        
        if rnn=='none':
            self.rnn = nn.Identity()
        elif rnn=='lstm':
            self.rnn = LSTMBlock(self.config.hidden_size*self.multi_layers, (self.config.hidden_size*self.multi_layers)//2, num_layers=1, p_drop=p_drop)
        elif rnn=='gru':
            self.rnn = GRUBlock(self.config.hidden_size*self.multi_layers, (self.config.hidden_size*self.multi_layers)//2, num_layers=1, p_drop=p_drop)
        elif rnn=='transformer':
            self.rnn = TransformerBlock(self.config.hidden_size*self.multi_layers, num_layers=1, nhead=8)
        else:
            raise Exception()
    
        if self.msd=='true':
            self.dropout_1 = nn.Dropout(0.1)
            self.dropout_2 = nn.Dropout(0.2)
            self.dropout_3 = nn.Dropout(0.3)
            self.dropout_4 = nn.Dropout(0.4)
            self.dropout_5 = nn.Dropout(0.5)
            
        if head=='simple':
            self.head = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(self.config.hidden_size*self.multi_layers, self.num_labels)
            )
        else:
            raise Exception()
        self._init_weights(self.head)
        
        if loss=='mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss=='l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss=='smoothl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss=='bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif loss=='focal':
            self.loss_fn = FocalLoss(alpha=1, gamma=0.5, reduction='none')
        elif loss=='ccc':
            self.loss_fn = CCCLoss(reduction='none')
        elif loss=='xentropy':
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif loss=='smoothl1+xentropy':
            self.loss_fn = CustomLoss('smoothl1','xentropy',reduction='none')
        else:
            raise Exception()
            
        self.map_tensor = torch.Tensor([0, 0.25, 0.5, 0.75, 1.0])[None]
        
        if self.adv_sift=='true':
            adv_modules = hook_sift_layer(self, hidden_size=self.config.hidden_size)
            self.adv = AdversarialLearner(self, adv_modules)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward_logits(self, input_ids, attention_mask, anchor_target_input_ids, aug=False):
        hidden_states = self.transformer(input_ids=input_ids, 
                                         attention_mask=attention_mask).hidden_states[-self.multi_layers:] # list of (bs,num_tokens,hidden_size)
        hidden_states = torch.cat(hidden_states, dim=2) # (bs,num_tokens,hidden_size*multi_layers)
        hidden_states = self.rnn(hidden_states)
        logits_list = []
        for tmp_logits, anchor_target_input_ids in zip(hidden_states, anchor_target_input_ids):
            tmp_logits = tmp_logits[:len(anchor_target_input_ids[:-1])] # (anchor_target_num_tokens,hidden_size*multi_layers)
            tmp_logits = tmp_logits.mean(dim=0) # (hidden_size*multi_layers,)
            logits_list.append(tmp_logits)
        logits = torch.stack(logits_list) # (bs,hidden_size*multi_layers)
        
        if aug:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = logits.size()[0]
            index = torch.randperm(batch_size).cuda()
            logits = lam * logits + (1 - lam) * logits[index, :]
        
        if self.msd=='true' and self.training:
            logits_1 = self.head(self.dropout_1(logits))
            logits_2 = self.head(self.dropout_2(logits))
            logits_3 = self.head(self.dropout_3(logits))
            logits_4 = self.head(self.dropout_4(logits))
            logits_5 = self.head(self.dropout_5(logits))
            logits = (logits_1 + logits_2 + logits_3 + logits_4 + logits_5) / 5.0
        else:
            logits = self.head(logits) # (bs,num_labels)
        
        if self.l2norm=='true':
            logits = self.s * F.normalize(logits, dim=-1)
            
        if aug:
            return logits, index, lam
        else:
            return logits
    
    
    def logits_fn(self, *wargs, **kwargs):
        logits = self.forward_logits(**kwargs)
        return logits


    def training_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
            'anchor_target_input_ids':data['anchor_target_input_ids'],
            'aug':False,
        }
        
        # get loss
        if self.loss in ['xentropy']:
            if self.aug=='mixup' and np.random.random()<self.p_aug and self._current_epoch<self.aug_stop_epoch:
                input_data['aug'] = True
                logits, index, lam = self.forward_logits(**input_data)
                loss_a = self.get_losses(logits, data['target_map']).mean()
                loss_b = self.get_losses(logits, data['target_map'][index]).mean()
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                logits = self.forward_logits(**input_data)
                loss = self.get_losses(logits, data['target_map']).mean()
        elif self.loss in ['smoothl1+xentropy']:
            if self.aug=='mixup' and np.random.random()<self.p_aug and self._current_epoch<self.aug_stop_epoch:
                input_data['aug'] = True
                logits, index, lam = self.forward_logits(**input_data)
                loss_a = self.get_losses(logits, data['target'], data['target_map']).mean()
                loss_b = self.get_losses(logits, data['target'], data['target_map'][index]).mean()
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                logits = self.forward_logits(**input_data)
                loss = self.get_losses(logits, data['target'], data['target_map']).mean()
        else:
            if self.aug=='mixup' and np.random.random()<self.p_aug and self._current_epoch<self.aug_stop_epoch:
                input_data['aug'] = True
                logits, index, lam = self.forward_logits(**input_data)
                loss_a = self.get_losses(logits, data['target']).mean()
                loss_b = self.get_losses(logits, data['target'][index]).mean()
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                logits = self.forward_logits(**input_data)
                loss = self.get_losses(logits, data['target']).mean()
           
        if self.adv_sift=='true':
            input_data['aug'] = False
            loss = loss + self.adv.loss(logits, self.logits_fn, **input_data)
        
        # get score
        if self.loss in ['bce','focal']:
            score = self.get_scores(logits.sigmoid().detach().cpu().numpy(), data['target'].detach().cpu().numpy())
        elif self.loss in ['xentropy','smoothl1+xentropy']:
            pred = (logits.softmax(-1) * self.map_tensor.to(logits.device)).sum(-1).detach().cpu().numpy()
            score = self.get_scores(pred, data['target'].detach().cpu().numpy())
        else:
            score = self.get_scores(logits.detach().cpu().numpy(), data['target'].detach().cpu().numpy())
            
        return loss, score
    
    def validation_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
            'anchor_target_input_ids':data['anchor_target_input_ids'],
            'aug':False,
        }
        
        # get loss
        if self.loss in ['xentropy']:
            logits = self.forward_logits(**input_data)
            loss  = self.get_losses(logits, data['target_map']).detach().cpu().numpy()
        elif self.loss in ['smoothl1+xentropy']:
            logits = self.forward_logits(**input_data)
            loss = self.get_losses(logits, data['target'], data['target_map']).detach().cpu().numpy()
        else:
            logits = self.forward_logits(**input_data)
            loss  = self.get_losses(logits, data['target']).detach().cpu().numpy()
            
        # get pred
        if self.loss in ['bce','focal']:
            pred = logits.sigmoid().detach().cpu().numpy().reshape(-1,)
        elif self.loss in ['xentropy','smoothl1+xentropy']:
            pred = (logits.softmax(-1) * self.map_tensor.to(logits.device) ).sum(-1).detach().cpu().numpy().reshape(-1,)
        else:
            pred = logits.detach().cpu().numpy().reshape(-1,)
            
        target = data['target'].detach().cpu().numpy().reshape(-1,)
        return {
            'loss':loss,
            'pred':pred,
            'target':target,
            'data_id':data['data_id'],
            'anchor_text':data['anchor_text'],
            'target_text':data['target_text'],
            'context_text':data['context_text'],
        }
    
    def validation_epoch_end(self, outputs):
        losses = []
        preds = []
        targets = []
        for o in outputs:
            losses.append(o['loss'])
            preds.append(o['pred'])
            targets.append(o['target'])
        losses = np.hstack(losses).mean()
        preds = np.hstack(preds)
        targets = np.hstack(targets)
        scores = self.get_scores(preds, targets)
        self._current_epoch += 1
        return losses, scores
        
    def configure_optimizers(self):
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler(optimizer)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def fetch_optimizer(self):
        head_params = list(self.head.named_parameters())
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n,p in head_params], 
                "weight_decay": 0.01,
                "lr": self.head_learning_rate,
            }
        ]
        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        return optimizer

    def fetch_scheduler(self, optimizer):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=0.5,
            last_epoch=-1,
        )
        return scheduler
    
    def get_losses(self, logits, target, target_map=None):
        if self.loss in ['xentropy']:
            loss = self.loss_fn(logits.reshape(-1,self.num_labels), target.reshape(-1,))
        elif self.loss in ['smoothl1+xentropy']:
            loss = self.loss_fn((logits.reshape(-1,self.num_labels).softmax(-1) * self.map_tensor.to(logits.device) ).sum(-1), 
                                logits.reshape(-1,self.num_labels), 
                                target.reshape(-1,), target_map.reshape(-1,))
        else:
            loss = self.loss_fn(logits.reshape(-1,), target.reshape(-1,))
        return loss
    
    def get_scores(self, pred, target):
        score = calc_pearson_corr(pred.reshape(-1,), target.reshape(-1,))
        return score
    
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re

def replace_rn(text):
    return text.replace('\r\n', '[]')

class DatasetTrain(Dataset):
    def __init__(self, df, tokenizer, mask_prob=0, mask_ratio=0, aug='false', loss='mse'):
        self.df = df
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.aug = aug
        self.aug_dict = None
        self.loss = loss
        
        self.df['anchor'] = self.df['anchor'].apply(lambda x:x.lower())
        self.df['target'] = self.df['target'].apply(lambda x:x.lower())
        self.df['context'] = self.df['context'].apply(lambda x:x.lower())
        self.df['anchor_context'] = self.df['anchor'] + '_' + self.df['context']
        
        print('aug = ', aug)
        
        if aug=='true':
            score_1_df = df[df['score']==1].reset_index(drop=True)
            score_1_df = score_1_df[score_1_df['anchor']!=score_1_df['target']].reset_index(drop=True)
            aug_df = score_1_df[['anchor_context','target']].groupby('anchor_context')['target'].apply(list).reset_index()
            self.aug_dict = {}
            for ac,t in aug_df[['anchor_context','target']].values:
                self.aug_dict.update({ac:t})
        
        if 'deberta-v2' in self.tokenizer.name_or_path or 'deberta-v3' in self.tokenizer.name_or_path:
            self.df['anchor'] = self.df['anchor'].apply(lambda x:replace_rn(x))
            self.df['target'] = self.df['target'].apply(lambda x:replace_rn(x))
            self.df['context_text'] = self.df['context_text'].apply(lambda x:replace_rn(x))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        data_id = sample['id']
        anchor_text = sample['anchor']#.lower()
        target_text = sample['target']#.lower()
        context_text = sample['context_text']#.lower()
        anchor_context = sample['anchor_context']
        
        if (self.aug=='true') and (np.random.random()<0.5):
            if anchor_context in self.aug_dict.keys():
                swap_text_list = self.aug_dict[anchor_context]
                anchor_text = np.random.choice(swap_text_list)
        
        text = anchor_text + '[SEP]' + target_text + '[SEP]' + context_text
        
        tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        anchor_target_tokens = self.tokenizer.encode_plus(anchor_text + '[SEP]' + target_text, add_special_tokens=True)
        anchor_target_input_ids = torch.LongTensor(anchor_target_tokens['input_ids'])
        
#         # random masking (on pn only for full_inputs)
#         if np.random.random() < self.mask_prob:
#             all_inds = np.arange(1, len(pn_input_ids)-1)
#             n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
#             np.random.shuffle(all_inds)
#             mask_inds = all_inds[:n_mask]
#             full_input_ids[mask_inds] = self.tokenizer.mask_token_id
            
        return dict(
            data_id = data_id,
            anchor_text = anchor_text,
            target_text = target_text,
            context_text = context_text,
            text = text,
            input_ids = input_ids,
            attention_mask = attention_mask,
            anchor_target_input_ids = anchor_target_input_ids,
            target = sample['score'],
            target_map = sample['score_map']
        )
    
class CustomCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, samples):
        output = dict()
        
        for k in samples[0].keys():
            output[k] = [sample[k] for sample in samples]
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        
        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s.tolist() + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist() for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s.tolist() for s in output["attention_mask"]]
            
        output["input_ids"] = torch.LongTensor(output["input_ids"])
        output["attention_mask"] = torch.LongTensor(output["attention_mask"])
        
        if "target" in output.keys():
            output["target"] = torch.FloatTensor(output["target"])
            output["target_map"] = torch.LongTensor(output["target_map"])
        return output