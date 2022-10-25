#!/bin/bash -x
MODEL=microsoft/codebert-base

if [ "$#" -eq 2 ]; then
    DATASET=$1
    TASK=$2
    # default args to run locally
    DATA_DIR=../ranker_datasets/$DATASET/
    MODEL_CACHE_DIR=~/ranker_model_for_${DATASET}_cache
    MODEL_DIR=~/ranker_model_for_$DATASET
    TEST_FILE_SUFFIX=val.json
    LABELS_SUFFIX=labels_$TASK.txt
    LABEL_KEY=${TASK}_label
    PREDICT_FILE_SUFFIX=test
else
    # arguments
    DATA_DIR=$1
    MODEL_DIR=$2
    MODEL_CACHE_DIR=$3
    TASK=$4
    TEST_FILE=$5
    PREDICT_FILE_SUFFIX=$6
    LABELS_SUFFIX=labels_$TASK.txt
    LABEL_KEY=${TASK}_label
fi

if [ $TASK != "execution_error_with_line" ]; then 
    python run_seq_classification.py \
        --output_dir $MODEL_DIR \
        --cache_dir $MODEL_CACHE_DIR \
        --model_name_or_path $MODEL \
        --test_file $DATA_DIR/$TEST_FILE \
        --sentence1_key prompt \
        --sentence2_key completion \
        --label_key $LABEL_KEY \
        --labels_file $DATA_DIR/$LABELS_SUFFIX \
        --max_seq_length 512 \
        --do_predict \
        --per_device_eval_batch_size 32 \
        --predict_suffix $PREDICT_FILE_SUFFIX \
        --overwrite_cache \

else
    LABELS_SUFFIX=labels_execution_error.txt
    LABEL_KEY=execution_error_label

    python3 run_seq_classification_and_line_prediction.py \
        --output_dir $MODEL_DIR \
        --cache_dir $MODEL_CACHE_DIR \
        --model_name_or_path $MODEL \
        --test_file $DATA_DIR/$TEST_FILE \
        --sentence1_key prompt \
        --sentence2_key completion \
        --label_key $LABEL_KEY \
        --labels_file $DATA_DIR/$LABELS_SUFFIX \
        --max_seq_length 512 \
        --do_predict \
        --per_device_eval_batch_size 32 \
        --predict_suffix $PREDICT_FILE_SUFFIX \
        --overwrite_cache \
        
fi 