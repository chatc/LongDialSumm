#!/usr/bin/env bash

TASK=QMSum
INPUT=QMSum
OUTPUT=EXP/QMSum

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

mkdir "$OUTPUT"
for SPLIT in train val
do
for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUT/$SPLIT.$LANG" \
    --outputs "$OUTPUT/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${OUTPUT}/train.bpe" \
  --validpref "${OUTPUT}/val.bpe" \
  --destdir "EXP/${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4

# for bart large
#wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
#tar -xzvf bart.large.tar.gz
# BART_PATH=your_path/bart.large/model.pt

# for bart base
#wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
#tar -xzvf bart.base.tar.gz
#BART_PATH=your_path/bart.base/model.pt

# for bart cnn
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz
tar -xzvf bart.large.cnn.tar.gz
BART_PATH=./bart.large.cnn/model.pt

# for bart xsum
#wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz
#tar -xzvf bart.large.xsum.tar.gz
# BART_PATH=your_path/bart.large.xsum/model.pt

# for bart-cnn -> MediaSum 2w -> QMSum
# BART_PATH=your_path/checkpoint_last_cnn_media_epoch7.pt

# for Squad
#BART_PATH=your_path/bart_large_squad.pt

CUDA_VISIBLE_DEVICES='0' fairseq-train "EXP/${TASK}-bin" \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --arch bart_large;
    # --arch bart_base \
