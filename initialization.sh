# export CUDA_VISIBLE_DEVICES=2,3

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SAVE_DIR="model_zoo/loftq/"
# accelerate launch --multi_gpu quantize_save.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --token $HF_TOKEN \
#     --bits 8 \
#     --iter 5 \
#     --rank 64 \
#     --save_dir $SAVE_DIR

export CUDA_VISIBLE_DEVICES=3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 

SAVE_DIR="model_zoo/loftq2/"
python quantize_save.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token $HF_TOKEN \
    --bits 4 \
    --iter 5 \
    --rank 64 \
    --save_dir $SAVE_DIR