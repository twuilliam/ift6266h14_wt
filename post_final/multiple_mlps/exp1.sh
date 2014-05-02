EXP=aa
mkdir experiences/${EXP}
DATASET=acoustic/timit_${EXP}_train_aug.npz

#SdA

PREEPOCHS=15

THEANO_FLAGS=mode=FAST_RUN,device=gpu0 python26 SdA.py -b 16 -e $PREEPOCHS -i $DATASET -o experiences/${EXP}

#MLP

EPOCHS=500
UNITS="[300, 300]"

THEANO_FLAGS=mode=FAST_RUN,device=gpu0 python26 deep_mlp_sda.py -b 16 -e $EPOCHS -n "$UNITS" -o experiences/${EXP} -i $DATASET -l 0.0