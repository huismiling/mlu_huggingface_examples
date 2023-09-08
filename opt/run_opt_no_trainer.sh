
set -xe 

pretrained_model=$PRETRAINED_MODELS/opt-1.3b/
trainset=$data_root/wikitext-103/wiki.train.txt
validset=$data_root/wikitext-103/wiki.valid.txt

export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1

export MLU_VISIBLE_DEVICES=7
export MLU_VISIBLE_DEVICES=4,5,6,7

# unset MLU_VISIBLE_DEVICES

export MASTER_PORT=12332

accelerate-launch --config_file configs/default_config.yaml --num_processes 4\
    --main_process_port $MASTER_PORT \
    run_clm_no_trainer.py \
    --model_name_or_path $pretrained_model \
    --train_file $trainset \
    --validation_file $validset \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --with_tracking \
    --block_size 512 \
    --max_train_steps 40 \
    --seed 7 \
    --output_dir $PWD/outputs


