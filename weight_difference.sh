python weight_difference.py \
  --fp_model_path meta-llama/Llama-2-7b-hf \
  --loftq_model_path model_zoo/loftq/Llama-2-7b-hf-4bit-64rank \
  --loftq_subfolder loftq_init \
  --output_csv weight_diff_loftq_vs_fp.csv
