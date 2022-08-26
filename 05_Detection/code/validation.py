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
    #parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-effectiveness/', required=False)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-2021/', required=False)
    #parser.add_argument("--input_path", type=str, default='../../00_EDA/00_v2_04/result/', required=False)
    
    parser.add_argument("--val_batch_size", type=int, default=1, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    #parser.add_argument("--num_labels", type=int, default=1, required=False)
    parser.add_argument("--num_classes", type=int, default=7, required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    parser.add_argument("--max_length", type=int, default=1024, required=False)
    
    parser.add_argument("--train_text_dir", type=str, default='../../input/feedback-prize-2021/train/', required=False)
    parser.add_argument("--test_score_thr", type=float, default=0.5, required=False)
    parser.add_argument("--weight_path", type=str, default='none', required=False)
    
    return parser.parse_args()

    
from models import TextSpanDetector, DatasetTrain, CustomCollator, TYPE2LABEL
# import sys
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
        
    train_df = pd.read_csv(opj(args.input_path, 'train.csv')).rename(columns={'id':'essay_id'})
    #train_df = pd.read_csv(opj(args.input_path, 'unlabeled.csv'))
    #test_df = pd.read_csv(opj(args.input_path, 'test.csv'))
    #sub_df = pd.read_csv(opj(args.input_path, 'sample_submission.csv'))

    print('train_df.shape = ', train_df.shape)
    #print('test_df.shape = ', test_df.shape)
    #print('sub_df.shape = ', sub_df.shape)

    #LABEL = 'discourse_effectiveness'
    
    #from preprocessing import generate_text
    #train_df =generate_text(train_df)
    #train_df['label'] = train_df[LABEL].map({'Ineffective':0, 'Adequate':1, 'Effective':2})

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
    
    trn_df = trn_df.rename(columns={'essay_id':'id'})
    val_df = val_df.rename(columns={'essay_id':'id'})
    
    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    if 'deberta-v2' in args.model or  'deberta-v3' in args.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
       
    from torch.utils.data import DataLoader
    val_dataset = DatasetTrain(
            val_df,
            text_dir=args.train_text_dir,
            tokenizer=tokenizer,
            mask_prob=0,
            mask_ratio=0,
        )
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
    model = TextSpanDetector(
            args.model,
            tokenizer,
            #num_labels=args.num_labels, 
            #num_labels_2=args.num_labels_2,
            num_classes=args.num_classes,
            
            dynamic_positive=True,
            with_cp=False,
            #local_files_only=True,
            #init_cfg=None,
            
            #hidden_dropout_prob=args.hidden_drop_prob, 
            #p_drop=args.p_drop,
            #learning_rate=args.lr,
            #head_learning_rate=args.head_lr,
            #num_train_steps=num_train_steps,
            #warmup_ratio=args.warmup_ratio,
            #model_pretraining=model_pretraining,
            rnn=args.rnn,
            loss=args.loss,
            head=args.head,
            #msd=args.msd,
            multi_layers=args.multi_layers,
            #aug=args.aug,
            #mixup_alpha=args.mixup_alpha,
            #aug_stop_epoch=args.aug_stop_epoch,
            #p_aug=args.p_aug,
            #adv_sift=args.adv_sift,
            #l2norm=args.l2norm,
            #weight_decay=args.weight_decay,
            #freeze_layers=args.freeze_layers,
            #max_length=args.max_length,
            #mt=args.mt,
            #w_mt=args.w_mt,
        )
    if args.weight_path!='none':
        weight_path = args.weight_path
    else:
        weight_path = f'./result/{args.version}/model_seed{args.seed}_fold{args.fold}.pth'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    
    from tqdm import tqdm
    outputs = []
    for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        with torch.no_grad():
            outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
    val_res = model.validation_epoch_end(outputs)
    val_loss, val_score = val_res['loss'], val_res['score']
    print('val_loss={:.4f}, val_score={:.4f}'.format(val_loss, val_score))
    print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}'.format(
        val_res['obj_loss'], val_res['reg_loss'], val_res['cls_loss']
    ))    
        
    losses = []
    obj_losses = []
    reg_losses = []
    cls_losses = []
    pred_dfs = []
    gt_dfs = []
    data_ids = []
    for o in outputs:
        losses.append(o['loss'])
        obj_losses.append(o['obj_loss'])
        reg_losses.append(o['reg_loss'])
        cls_losses.append(o['cls_loss'])
        pred_dfs.append(o['pred_df'])
        gt_dfs.append(o['gt_df'])
        data_ids.append(o['data_id'])
        
    raw_oof = {
        data_id:{
            'pred_df':pred_df, 
            'gt_df':gt_df, 
            'loss':loss,
            'obj_loss':obj_loss,
            'reg_loss':reg_loss,
            'cls_loss':cls_loss,
        } for data_id, pred_df, gt_df, loss, obj_loss, reg_loss, cls_loss in zip(
            data_ids, pred_dfs, gt_dfs, losses, obj_losses, reg_losses, cls_losses
        )
    }
    
    import joblib
    print(f'save raw_oof_fold{args.fold}...')
    joblib.dump(raw_oof, f'./result/{args.version}/raw_oof_fold{args.fold}.joblib')
    print(f'save raw_oof_fold{args.fold}, done')
    print('\n')