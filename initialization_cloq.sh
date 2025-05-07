export CUDA_VISIBLE_DEVICES=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR="model_zoo/cloq/"


cd /home/esthersong/LoftQ

PYTHONPATH=$(pwd) python quantize_save_cloq.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --quant_model_path "model_zoo/gptq/Llama-2-7b-hf-4bit" \
    --token $HF_TOKEN \
    --bits 4 \
    --iter 1 \
    --rank 64 \
    --save_dir $SAVE_DIR
    