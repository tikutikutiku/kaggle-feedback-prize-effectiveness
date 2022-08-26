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

import transformers
transformers.logging.set_verbosity_error()


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
    #parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1, required=True)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-2021/', required=False)
    
    parser.add_argument("--val_batch_size", type=int, default=8, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--l2norm", type=str, default='false', required=False)
    
    parser.add_argument("--weight_path", type=str, default='none', required=False)
    
    parser.add_argument("--window_size", type=int, default=512, required=False)
    parser.add_argument("--inner_len", type=int, default=384, required=False)
    parser.add_argument("--edge_len", type=int, default=64, required=False)
    
    parser.add_argument("--text_dir", type=str, default='../../input/feedback-prize-2021/train', required=False)
    parser.add_argument("--max_length", type=int, default=-100, required=False)
    parser.add_argument("--ratio_masking", type=float, default=0.25, required=False)
    
    return parser.parse_args()

    
#from models import Model, DatasetTrain, CustomCollator
from models_pretrain_generator import DatasetTrain, CustomCollator, Model as Generator

# import sys
# sys.path.append('../../../../../COCO-LM-main/huggingface')
# from cocolm.tokenization_cocolm import COCOLMTokenizer
    
    
discourse_type_list = [
    'Lead',
    'Position',
    'Claim',
    'Counterclaim',
    'Rebuttal',
    'Evidence',
    'Concluding Statement'
]

if __name__=='__main__':
#if True:
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
        
    train_df = pd.read_csv(opj(args.input_path, 'train.csv')).rename(columns={'id':'essay_id'})
    print('train_df.shape = ', train_df.shape)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    output_path = opj(f'./result', args.version)
    os.makedirs(output_path, exist_ok=True)
    fold_path = args.fold_path
    import joblib
    print('load folds...')
    trn_ids_list = joblib.load(opj(fold_path,f'trn_ids_list.joblib'))
    val_ids_list = joblib.load(opj(fold_path,f'val_ids_list.joblib'))
    
    trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold])].reset_index(drop=True)
    val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold])].reset_index(drop=True)
    #trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold][:30])].reset_index(drop=True)
    #val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold][:30])].reset_index(drop=True)
    
    #trn_df = trn_df.rename(columns={'essay_id':'id'})
    #val_df = val_df.rename(columns={'essay_id':'id'})

    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    #from preprocessing import relation_mapper
    if 'deberta-v2' in args.generator or 'deberta-v3' in args.generator:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.generator, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in discourse_type_list]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.generator, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in discourse_type_list]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    
    val_dataset = DatasetTrain(
        val_df, 
        tokenizer, 
        ratio_masking=args.ratio_masking, 
        max_length=args.max_length,
        text_dir=args.text_dir,
        mode='valid'
    )

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
    model_pretraining = None
    model_gen = Generator(args.generator, 
                          tokenizer,
                          num_labels=len(tokenizer), 
                          #hidden_dropout_prob=args.hidden_drop_prob, 
                          #p_drop=args.p_drop,
                          #learning_rate=args.lr,
                          #head_learning_rate=args.head_lr,
                          #num_train_steps=num_train_steps,
                          #warmup_ratio=args.warmup_ratio,
                          ratio_masking=args.ratio_masking,
                          #freeze_layers=args.freeze_layers,
                          #scheduler=args.scheduler,
                          #num_cycles=args.num_cycles,
                          #with_cp=(args.check_pointing=='true'),
                          window_size=args.window_size,
                          inner_len=args.inner_len,
                          edge_len=args.edge_len,
                         )
    if args.weight_path!='none':
        weight_path = args.weight_path
    else:
        weight_path = f'./result/{args.version}/model_gen_seed{args.seed}_fold{args.fold}.pth'
    model_gen.load_state_dict(torch.load(weight_path))
    model_gen = model_gen.cuda()
    model_gen.eval()
    
    from tqdm import tqdm
    outputs = []
    for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        with torch.no_grad():
            output,_ = model_gen.validation_step(batch)
            outputs.append(output)
    val_loss, val_score = model_gen.validation_epoch_end(outputs)
    print('val_loss={:.4f}, val_score={:.4f}'.format(val_loss, val_score))