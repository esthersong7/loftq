export CUDA_VISIBLE_DEVICES=3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



SAVE_DIR="model_zoo/gptq/"
python gptq_quantization.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token $HF_TOKEN \
    --bits 4 \
    --save_dir $SAVE_DIR