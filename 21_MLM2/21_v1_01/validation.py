import torch
print(torch.__name__, torch.__version__)

import argparse
import os
from os.path import join as opj
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_path", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1, required=True)
    parser.add_argument("--input_path", type=str, default='../../input/us-patent-phrase-to-phrase-matching/', required=False)
    parser.add_argument("--val_batch_size", type=int, default=32, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    parser.add_argument("--num_labels", type=int, default=1, required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    parser.add_argument("--ho_fold", type=int, default=-1, required=False)
    return parser.parse_args()

    
from models import Model, DatasetTrain, CustomCollator
import sys
# sys.path.append('../../../../../COCO-LM-main/huggingface')
# from cocolm.tokenization_cocolm import COCOLMTokenizer
    
if __name__=='__main__':
#if True:
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
        
    train_df = pd.read_csv(opj(args.input_path, 'train.csv'))
    test_df = pd.read_csv(opj(args.input_path, 'test.csv'))
    sub_df = pd.read_csv(opj(args.input_path, 'sample_submission.csv'))

    print('train_df.shape = ', train_df.shape)
    print('test_df.shape = ', test_df.shape)
    print('sub_df.shape = ', sub_df.shape)

    ID = 'id'
    LABEL = 'score'
    
    from preprocessing import get_context_text
    train_df = get_context_text(train_df)
    test_df = get_context_text(test_df)
    train_df['score_map'] = train_df['score'].map({0:0, 0.25:1, 0.50:2, 0.75:3, 1.0:4})

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if args.ho_fold==-1:
        output_path = opj(f'./result', args.version)
    else:
        output_path = opj(f'./result', args.version+'_ho')
    
    os.makedirs(output_path, exist_ok=True)
    fold_path = args.fold_path
    import joblib
    print('load folds...')
    trn_ids_list = joblib.load(opj(fold_path,f'trn_ids_list.joblib'))
    val_ids_list = joblib.load(opj(fold_path,f'val_ids_list.joblib'))
    
    if args.ho_fold==-1:
        trn_df = train_df[train_df['id'].isin(trn_ids_list[args.fold])].reset_index(drop=True)
        val_df = train_df[train_df['id'].isin(val_ids_list[args.fold])].reset_index(drop=True)
    else:
        trn_df = train_df[train_df['id'].isin(trn_ids_list[args.ho_fold])].reset_index(drop=True)
        val_df = train_df[train_df['id'].isin(val_ids_list[args.ho_fold])].reset_index(drop=True)
    
    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    
    if 'deberta-v2' in args.model or  'deberta-v3' in args.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model)
        special_tokens_dict = {'additional_special_tokens': ['[]']}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    elif 'cocolm' in args.model:
        tokenizer = COCOLMTokenizer.from_pretrained(args.model)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
       
    val_dataset = DatasetTrain(val_df, tokenizer)
    from torch.utils.data import DataLoader
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
    
    #model
    model = Model(args.model, 
                  tokenizer,
                  num_labels=args.num_labels, 
                  rnn=args.rnn,
                  loss=args.loss,
                  head=args.head,
                  multi_layers=args.multi_layers,
                 )
    weight_path = f'./result/{args.version}/model_seed{args.seed}_fold{args.fold}.pth'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    
    from tqdm import tqdm
    outputs = []
    for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        with torch.no_grad():
            output = model.validation_step(batch)
            outputs.append(output)
    val_loss, val_score = model.validation_epoch_end(outputs)
    print('val_loss={:.4f}, val_score={:.4f}'.format(val_loss, val_score))
            
    preds = []
    targets = []
    losses = []
    data_ids = []
    anchor_texts = []
    target_texts = []
    context_texts = []
    for o in outputs:
        preds.extend(o['pred'])
        targets.extend(o['target'])
        losses.extend(o['loss'])
        data_ids.extend(o['data_id'])
        anchor_texts.extend(o['anchor_text'])
        target_texts.extend(o['target_text'])
        context_texts.extend(o['context_text'])
    
    raw_oof = {
        data_id:{
            'pred':pred, 
            'target':target, 
            'loss':loss,
            'anchor_text':anchor_text,
            'target_text':target_text,
            'context_text':context_text
        } for data_id,pred,target,loss,anchor_text,target_text,context_text in zip(
            data_ids, preds, targets, losses, anchor_texts, target_texts, context_texts
        )
    }
    
    import joblib
    print(f'save raw_oof_fold{args.fold}...')
    if args.ho_fold==-1:
        joblib.dump(raw_oof, f'./result/{args.version}/raw_oof_fold{args.fold}.joblib')
    else:
        joblib.dump(raw_oof, f'./result/{args.version}/raw_oof_fold{args.fold}_ho.joblib')
    print(f'save raw_oof_fold{args.fold}, done')
    print('\n')