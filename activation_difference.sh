python activation_difference.py \
  --baseline_model_path meta-llama/Llama-2-7b-hf \
  --quant_model_path model_zoo/loftq/Llama-2-7b-hf-4bit-64rank \
  --quant_adapter_subfolder loftq_init \
  --num_inputs 20 \
  --metric mse \
  --output_csv activation_diff_llama_fp_vs_loftq.csv
