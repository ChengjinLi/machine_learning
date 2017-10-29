#!/bin/sh

python train.py \
    --data-set movie_review_data \
    --model-name text_rnn \
    --num-classes 2 \
    --embedding-dim 128 \
    --dropout-keep-prob 0.5 \
    --batch-size 64 \
    --optimizer adam \
    --max-sentence-length 56 \
    --learning-rate 0.001 \
    --init-scale 0.1 \
    --num-epochs 10 \
    --valid-num 1000 \
    --show-freq 10 \
    --valid-freq 100 \
    --save-freq 100 \
    --model-dir ./model \
    --allow-soft-placement \
    --bidirectional \
    --cell-type gru \
    --hidden-layer-num 3 \
    --hidden-neural-size 128 \
    --max-grad-norm 5 \
    --output-method mean