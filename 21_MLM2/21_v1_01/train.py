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
    #parser.add_argument("--input_path", type=str, default='../../input/us-patent-phrase-to-phrase-matching/', required=False)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-2021/', required=False)
    
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--mask_prob", type=float, default=0.0, required=False)
    parser.add_argument("--mask_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--trn_batch_size", type=int, default=8, required=False)
    parser.add_argument("--val_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, required=False)
    parser.add_argument("--gradient_clip_val", type=int, default=1, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    
    parser.add_argument("--restart_epoch", type=int, default=0, required=False)
    parser.add_argument("--early_stopping", type=str, default='true', required=False)
    parser.add_argument("--patience", type=int, default=100, required=False)
    parser.add_argument("--hidden_drop_prob", type=float, default=0.1, required=False)
    parser.add_argument("--p_drop", type=float, default=0.5, required=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.0, required=False)
    
    parser.add_argument("--pretrain_path", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    parser.add_argument("--aug", type=str, default='false', required=False)
    parser.add_argument("--msd", type=str, default='false', required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    parser.add_argument("--eval_step", type=int, default=-1, required=False)
    parser.add_argument("--stop_epoch", type=int, default=999, required=False)
    
    parser.add_argument("--num_labels", type=int, default=1, required=False)
    parser.add_argument("--mixup_alpha", type=float, default=1.0, required=False)
    parser.add_argument("--aug_stop_epoch", type=int, default=999, required=False)
    parser.add_argument("--p_aug", type=float, default=0.5, required=False)
    parser.add_argument("--adv_sift", type=str, default='false', required=False)
    parser.add_argument("--l2norm", type=str, default='false', required=False)
    parser.add_argument("--sampling", type=str, default='simple', required=False)
    
    return parser.parse_args()

    
    
if __name__=='__main__':
#if True:
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
        
    train_df = pd.read_csv(opj(args.input_path, 'train.csv')).rename(columns={'id':'essay_id'})
    #test_df = pd.read_csv(opj(args.input_path, 'test.csv'))
    #sub_df = pd.read_csv(opj(args.input_path, 'sample_submission.csv'))

    print('train_df.shape = ', train_df.shape)
    #print('test_df.shape = ', test_df.shape)
    #print('sub_df.shape = ', sub_df.shape)

#     from preprocessing import get_context_text
#     train_df = get_context_text(train_df)
#     test_df = get_context_text(test_df)
#     train_df['score_map'] = train_df['score'].map({0:0, 0.25:1, 0.50:2, 0.75:3, 1.0:4})

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    output_path = opj(f'./result', args.version)
    os.makedirs(output_path, exist_ok=True)
    fold_path = args.fold_path
    import joblib
    print('load folds...')
    trn_ids_list = joblib.load(opj(fold_path,f'trn_ids_list.joblib'))
    val_ids_list = joblib.load(opj(fold_path,f'val_ids_list.joblib'))
    
    #trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold])].reset_index(drop=True)
    #val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold])].reset_index(drop=True)
    trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold][:30])].reset_index(drop=True)
    val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold][:30])].reset_index(drop=True)
    
    trn_df = trn_df.rename(columns={'essay_id':'id'})
    val_df = val_df.rename(columns={'essay_id':'id'})
    
    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    from run import run
    run(args, trn_df, val_df, pseudo_df=None)