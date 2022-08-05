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
sys.path.append('../../../../../COCO-LM-main/huggingface')
from cocolm.modeling_cocolm import COCOLMModel
from cocolm.configuration_cocolm import COCOLMConfig
from sklearn.metrics import f1_score
import numpy as np

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
        
        if 'cocolm' in model_name:
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
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.num_labels)
        )
        self._init_weights(self.head)
        
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
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
        
    def forward_logits(self, data):
        last_hidden_state = self.transformer(input_ids=data['input_ids'], 
                                             attention_mask=data['attention_mask']).last_hidden_state # (bs,num_tokens,hidden_size)
        logits = self.head(last_hidden_state) # (bs,num_tokens,num_labels)
        return logits
    
    def training_step(self, batch):
        '''
        For RTD pretraining, input data comes from the output of the generator.
        The input data shape will be (bs,num_tokens,num_labels=1)
        '''
        data = to_gpu(batch)
        logits = self.forward_logits(data)
        targets = data['rtd_label']
        mask = targets!=-100
        loss = self.get_losses(logits[mask].reshape(-1,), targets[mask].reshape(-1,).float()).mean()
        score = self.get_score(logits[mask].sigmoid().detach().cpu().numpy().argmax(-1).reshape(-1,),
                               targets[mask].detach().cpu().numpy().reshape(-1,))
        return loss, score
    
    def validation_step(self, batch):
        '''
        For RTD pretraining, input data comes from the output of the generator.
        The input data shape will be (bs,num_tokens,num_labels=1)
        '''
        data = to_gpu(batch)
        logits = self.forward_logits(data)
        targets = data['rtd_label']
        mask = targets!=-100
        loss = self.get_losses(logits[mask].reshape(-1,), targets[mask].reshape(-1,).float())
        return {
            'loss':loss.detach().cpu().numpy(),
            'pred':logits.sigmoid().detach().cpu().numpy().argmax(-1),
            'target':targets.detach().cpu().numpy(), 
            'text_list':data['text'],
            'data_id_list':data['data_id']
        }
    
    def validation_epoch_end(self, outputs):
        losses = []
        preds = []
        targets = []
        for o in outputs:
            losses.append(o['loss'].reshape(-1,))
            preds.append(o['pred'].reshape(-1,))
            targets.append(o['target'].reshape(-1,))
        loss = np.hstack(losses).mean()
        preds = np.hstack(preds)
        targets = np.hstack(targets)
        
        mask = (targets!=-100)
        preds = preds[mask]
        targets = targets[mask]
        
        score = self.get_score(preds, targets)
        self._current_epoch += 1
        return loss, score
    
    
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
    
    def get_losses(self, logits, target):
        loss = self.loss_fn(logits, target)
        return loss
    
    def get_score(self, pred, target):
        score = f1_score(target, pred, average='micro')
        return score