#!/bin/bash -x

# Common packages are already installed on the compute server
# Need an additional package? Install it here via:
#  pip3 install package-name

# Edit the line below to run your experiment (this is just an example). Note:
#  - This script will be run from your output directory
#  - Imported Data is accessible via the relative path ../input/

VERSION='26_Pseudo2/26_vl_01'
MODEL='../input/deberta-xlarge'
LR=6e-6 #8e-6 #2e-5
HEAD_LR=6e-6 #8e-6 #2e-5
TRN_BS=1
VAL_BS=1
ACCUM_STEP=1
EPOCHS=3
STOP_EPOCH=3
RESTART=1
HIDDEN_DROP_PROB=0
P_DROP=0
RNN='none'
WARMUP_RATIO=0.1
LOSS='xentropy'
HEAD='simple'
AUG='false' #'mixup'
MIXUP_ALPHA=1.0
P_AUG=0
AUG_STOP_EPOCH=2
MSD='true'
MULTI_LAYERS=1
EVAL_STEP=-1 #100
NUM_LABELS=3
NUM_LABELS_2=7
ADV_SIFT='false'
FP16='true' #'false'
WD=0.01
FREEZE='false'
#MAX_LENGTH = 1024
MULTI_TASK='false' #'true'
W_MT=1.0 #0.5
#PREPROCESSED_DATA_PATH = '../../00_EDA/00_v2_01/result/train.csv'
#'../../00_EDA/00_v1_09/result/train.csv'
AWP='true'
AWP_LR=1e-4 #1e-2 #1.0
AWP_EPS=1e-3 #0.01
AWP_START_EPOCH=1

PRETRAINED_DETECTOR_PATH='../input/tascj/result/deberta_xlarge_fold0.pth'
#PRETRAINED_DETECTOR_PATH = f'../../05_Detection/exp/result/05_v1_04/model_seed100_fold0_swa.pth'

MASK_PROB=0.8
MASK_RATIO=0.3

SCHEDULER='cosine_hard'
NUM_CYCLES=$EPOCHS

CP='true'

WINDOW_SIZE=1024 #512
INNER_LEN=768 #384
EDGE_LEN=128 #64

GRAD_CLIP=1000

SEED=100

# train with pseudo label
FOLD=0
INPUT_PATH=../${VERSION}/result/pseudo_label_fold${FOLD}.csv
FOLD_PATH=../00_EDA/00_v1_13/result/ # for pseudo-label
MODE='pseudo'
LOSS='bce' #'xentropy'
/usr/bin/time -f "Time taken: %E" python3 ../code/${VERSION}/train.py \
--model $MODEL --version $VERSION --fold_path $FOLD_PATH --fold $FOLD --seed $SEED \
--lr $LR --head_lr $HEAD_LR --trn_batch_size $TRN_BS --val_batch_size $VAL_BS \
--epochs $EPOCHS --hidden_drop_prob $HIDDEN_DROP_PROB --p_drop $P_DROP \
--accumulate_grad_batches $ACCUM_STEP --rnn $RNN --warmup_ratio $WARMUP_RATIO --loss $LOSS --aug $AUG --head $HEAD \
--mixup_alpha $MIXUP_ALPHA --p_aug $P_AUG --aug_stop_epoch $AUG_STOP_EPOCH \
--msd $MSD --multi_layers $MULTI_LAYERS --eval_step $EVAL_STEP --stop_epoch $STOP_EPOCH \
--num_labels $NUM_LABELS --num_labels_2 $NUM_LABELS_2 \
--restart_epoch $RESTART --adv_sift $ADV_SIFT --fp16 $FP16 --weight_decay $WD --freeze_layers $FREEZE \
--mt $MULTI_TASK --w_mt $W_MT \
--awp $AWP --awp_lr $AWP_LR --awp_eps $AWP_EPS --awp_start_epoch $AWP_START_EPOCH \
--pretrained_detector_path $PRETRAINED_DETECTOR_PATH --mask_prob $MASK_PROB --mask_ratio $MASK_RATIO \
--scheduler $SCHEDULER --num_cycles $NUM_CYCLES --check_pointing $CP \
--window_size $WINDOW_SIZE --inner_len $INNER_LEN --edge_len $EDGE_LEN \
--gradient_clip_val $GRAD_CLIP \
--input_path $INPUT_PATH --mode $MODE


# finetune
FOLD=0
INPUT_PATH='../input/feedback-prize-effectiveness/'
FOLD_PATH='../00_EDA/00_v2_07/result/' #'../../00_EDA/00_v1_01/result/'
MODE='train'
LOSS='xentropy'
PRETRAIN_PATH=result/${VERSION}/model_seed${SEED}_fold${FOLD}_epoch${EPOCHS}_pseudo.pth
/usr/bin/time -f "Time taken: %E" python3 ../code/${VERSION}/train.py \
--model $MODEL --version $VERSION --fold_path $FOLD_PATH --fold $FOLD --seed $SEED \
--lr $LR --head_lr $HEAD_LR --trn_batch_size $TRN_BS --val_batch_size $VAL_BS \
--epochs $EPOCHS --hidden_drop_prob $HIDDEN_DROP_PROB --p_drop $P_DROP \
--accumulate_grad_batches $ACCUM_STEP --rnn $RNN --warmup_ratio $WARMUP_RATIO --loss $LOSS --aug $AUG --head $HEAD \
--mixup_alpha $MIXUP_ALPHA --p_aug $P_AUG --aug_stop_epoch $AUG_STOP_EPOCH \
--msd $MSD --multi_layers $MULTI_LAYERS --eval_step $EVAL_STEP --stop_epoch $STOP_EPOCH \
--num_labels $NUM_LABELS --num_labels_2 $NUM_LABELS_2 \
--restart_epoch $RESTART --adv_sift $ADV_SIFT --fp16 $FP16 --weight_decay $WD --freeze_layers $FREEZE \
--mt $MULTI_TASK --w_mt $W_MT \
--awp $AWP --awp_lr $AWP_LR --awp_eps $AWP_EPS --awp_start_epoch $AWP_START_EPOCH \
--pretrained_detector_path $PRETRAINED_DETECTOR_PATH --mask_prob $MASK_PROB --mask_ratio $MASK_RATIO \
--scheduler $SCHEDULER --num_cycles $NUM_CYCLES --check_pointing $CP \
--window_size $WINDOW_SIZE --inner_len $INNER_LEN --edge_len $EDGE_LEN \
--gradient_clip_val $GRAD_CLIP \
--input_path $INPUT_PATH --mode $MODE --pretrain_path $PRETRAIN_PATH


# swa
#!python ../$VERSION/swa.py --work_dir ./result/$VERSION/ --fold 0 --seed $SEED --epochs '1 3'