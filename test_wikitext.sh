# test 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using one A100

python test_wikitext.py \
  --task "wikitext2" \
  --model_dir  \
  --device "cuda" \
  --peft
