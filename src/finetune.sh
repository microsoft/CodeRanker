#!/bin/bash -x
MODEL=microsoft/codebert-base

if [ "$#" -eq 2 ]; then
    DATASET=$1
    TASK=$2
    # default args to run locally
    DATA_DIR=ranker_datasets/$DATASET/
    MODEL_CACHE_DIR=~/ranker_model_for_${DATASET}_cache
    MODEL_DIR=~/ranker_model_for_$DATASET
    TRAIN_FILE_SUFFIX=train.json
    VAL_FILE_SUFFIX=val.json
    LABELS_SUFFIX=labels_$TASK.txt
    WEIGHTS_SUFFIX=weights_$TASK.txt 
    LABEL_KEY=${TASK}_label
else
    # arguments
    DATA_DIR=$1
    MODEL_DIR=$2
    MODEL_CACHE_DIR=$3
    TASK=$4
    TRAIN_FILE_SUFFIX=train.json
    VAL_FILE_SUFFIX=val.json
    LABELS_SUFFIX=labels_$TASK.txt
    WEIGHTS_SUFFIX=weights_$TASK.txt
    LABEL_KEY=${TASK}_label

fi

if [ $TASK != "execution_error_with_line" ]; then 
    python3 run_seq_classification.py \
        --output_dir $MODEL_DIR \
        --cache_dir $MODEL_CACHE_DIR \
        --model_name_or_path $MODEL \
        --train_file $DATA_DIR/$TRAIN_FILE_SUFFIX \
        --validation_file $DATA_DIR/$VAL_FILE_SUFFIX \
        --sentence1_key prompt \
        --sentence2_key completion \
        --label_key $LABEL_KEY \
        --labels_file $DATA_DIR/$LABELS_SUFFIX \
        --weights_file $DATA_DIR/$WEIGHTS_SUFFIX \
        --grouped_indices_file $DATA_DIR/val_grouped_indices.npy \
        --grouped_labels_file $DATA_DIR/val_grouped_labels.npy \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --learning_rate 1e-4 \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --gradient_accumulation_steps 32 \
        --num_train_epochs 30 \
        --evaluation_strategy steps \
        --save_strategy steps \
        --logging_steps 10 \
        --load_best_model_at_end \
        --metric_for_best_model top1_accuracy \
        --logging_first_step \
        --eval_steps 10 \
        --save_steps 10

else
    LABELS_SUFFIX=labels_execution_error.txt
    WEIGHTS_SUFFIX=weights_execution_error.txt
    LABEL_KEY=execution_error_label

    python3 run_seq_classification_and_line_prediction.py \
        --output_dir $MODEL_DIR \
        --cache_dir $MODEL_CACHE_DIR \
        --model_name_or_path $MODEL \
        --train_file $DATA_DIR/$TRAIN_FILE_SUFFIX \
        --validation_file $DATA_DIR/$VAL_FILE_SUFFIX \
        --sentence1_key prompt \
        --sentence2_key completion \
        --label_key $LABEL_KEY \
        --labels_file $DATA_DIR/$LABELS_SUFFIX \
        --weights_file $DATA_DIR/$WEIGHTS_SUFFIX \
        --grouped_indices_file $DATA_DIR/val_grouped_indices.npy \
        --grouped_labels_file $DATA_DIR/val_grouped_labels.npy \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --learning_rate 1e-4 \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --gradient_accumulation_steps 32 \
        --num_train_epochs 30 \
        --evaluation_strategy steps \
        --save_strategy steps \
        --logging_steps 10 \
        --load_best_model_at_end \
        --metric_for_best_model top1_accuracy \
        --logging_first_step \
        --eval_steps 10 \
        --save_steps 10 \
        --overwrite_output_dir \

fi
        