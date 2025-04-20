export CUDA_VISIBLE_DEVICES=3


python train_gsm8k.py \
    --model_name_or_path model_zoo/loftq/Llama-2-7b-hf-2bit-64rank \
    --num_bits 2 \
    --learning_rate 3e-4 \
    --seed 11 \
    --expt_name gsm8k_llama2_7b_2bit_64rank_loftq \
    --output_dir exp_results/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --report_to tensorboard