#!/bin/bash
# EDIT THESE, then:  bash train.sh
JUDGE=/PATH/TO/indicwav2vec-hindi
G=/PATH/G_PRETRAIN.pth
D=/PATH/D_PRETRAIN.pth
EPOCHS=500
BATCH=8
SR=48000
GPU=0

export PHON_JUDGE=$JUDGE
export C_PHON=2.0
export LAM_FRAG=4.0
export AUX=1
export CONTENT=1
export V2=0
export GTA=0
export CUDA_VISIBLE_DEVICES=$GPU

nohup python rvc/train/train.py hindi_phonloss 25 $EPOCHS $G $D $GPU $BATCH $SR False True False False 50 False HiFi-GAN False > train.log 2>&1 &
echo "started. watch:  tail -f train.log"
echo "good: '[PCL] ... fragile ids 12/12' then '[aux] ... phon=' ~2-8 falling"
