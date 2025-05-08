export CUDA_VISIBLE_DEVICES=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## loftq
# python activation_difference.py \
#   --ft True \
#   --baseline_model_path meta-llama/Llama-2-7b-hf \
#   --baseline_model_adapter_path LoftQ/Llama-2-7b-hf-fp16-64rank-gsm8k \
#   --quant_model_path model_zoo/loftq/Llama-2-7b-hf-8bit-64rank \
#   --quant_adapter_path exp_results/gsm8k_llama2_7b_8bit_64rank_loftq/Llama-2-7b-hf-8bit-64rank/ep_6/lr_0.0003/seed_11/ \
#   --quant_adapter_subfolder checkpoint-2802 \
#   --num_inputs 20 \
#   --metric mse \
#   --output_csv activation_diff_llama_fp_ft_vs_loftq_8bit_ft.csv


## cloq
python activation_difference.py \
  --ft False \
  --baseline_model_path meta-llama/Llama-2-7b-hf \
  --quant_model_path model_zoo/gptq/Llama-2-7b-hf-4bit \
  --quant_adapter_path model_zoo/cloq/Llama-2-7b-hf-4bit-64rank \
  --quant_adapter_subfolder cloq_init \
  --num_inputs 20 \
  --metric mse \
  --output_csv activation_diff_llama_fp_vs_cloq_4bit_init.csv