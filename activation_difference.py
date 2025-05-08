import torch
import argparse
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel, LlamaForCausalLM, GPTQConfig
# from auto_gptq import AutoGPTQForCausalLM
from peft import PeftModel,PeftModelForCausalLM
from datasets import load_dataset
import pandas as pd
from typing import Dict, List
import os



def compute_diff_metrics(a: torch.Tensor, b: torch.Tensor, metric: str = "mse") -> float:
    a = a.view(-1)
    b = b.view(-1)
    if metric == "mse":
        return F.mse_loss(a, b).item()
    elif metric == "cosine":
        return F.cosine_similarity(a, b, dim=0).item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """
    Add missing special tokens to tokenizer and resize model embeddings accordingly.
    If new tokens are added, their embeddings are initialized as the mean of existing embeddings.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_avg
        output_embeddings[-num_new_tokens:] = output_avg



TARGET_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

def normalize_layer_name(name: str) -> str:
    return (
        name.replace("base_model.model.model.", "")
            .replace("model.model.", "")
            .replace("model.", "")
    )


# 한 번만 호출되는 함수고,그 안에서 모델의 각 레이어를 순회하면서 forward hook을 등록해주는 함수
def register_hooks(model, activations_dict: Dict[str, List[torch.Tensor]]):
    # 지금 레이어 이름(name)에 해당하는 forward hook 함수를 리턴해주는 함수
    def hook_fn(norm_name):
        # PyTorch forward hook의 본체 함수
        # model.generate(...) 호출할 때마다, 각 레이어 통과할 때 fn(...)이 실행됨
        def fn(module, inp, out):
            # Handle tuple (block), tensor (linear), or dict-like (top-level)
            if isinstance(out, torch.Tensor):
                activations_dict[norm_name].append(out.detach().float().cpu())
            elif isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
                activations_dict[norm_name].append(out[0].detach().float().cpu())  # only first tensor (e.g., hidden_state)
            else:
                activations_dict[norm_name].append(torch.tensor(float('nan')))  # fallback
        return fn

    for name, module in model.named_modules():
        norm_name = normalize_layer_name(name)

        # Only attach to top-level projection modules (not lora_A, base_layer, etc.)
        if any(norm_name.endswith(proj) for proj in TARGET_NAMES) and \
           not any(x in norm_name for x in ["lora_", "base_layer", "embedding", "dropout"]):

            if norm_name not in activations_dict:
                activations_dict[norm_name] = []
            module.register_forward_hook(hook_fn(norm_name))
            # print(f"Registering hook on {norm_name}")

    if isinstance(model, PeftModelForCausalLM):
        layers = model.base_model.model.model.layers
    elif isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    for i, block in enumerate(layers):
        name = f"block_{i}"
        activations_dict[name] = []
        block.register_forward_hook(hook_fn(name))
        # import pdb; pdb.set_trace();``




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", type=bool, default=False)
    parser.add_argument("--baseline_model_path", type=str, required=True)
    parser.add_argument("--baseline_model_adapter_path", type=str, default=None)
    parser.add_argument("--quant_model_path", type=str, required=True)
    parser.add_argument("--quant_adapter_path", type=str, required=True)
    parser.add_argument("--quant_adapter_subfolder", type=str, default="loftq_init")
    parser.add_argument("--num_inputs", type=int, default=20)
    parser.add_argument("--metric", type=str, choices=["mse", "cosine"], default="mse")
    parser.add_argument("--output_csv", type=str, default="activation_diff.csv")
    args = parser.parse_args()

    # Load tokenizer from baseline
    tokenizer = AutoTokenizer.from_pretrained(args.baseline_model_path)

    # Define special tokens to ensure pad_token is set
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"


    # Load models

    # load baseline (FP LLama2-7b) model

    model_fp = AutoModelForCausalLM.from_pretrained(
        args.baseline_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    if args.ft and args.baseline_model_adapter_path is not None:
        model_fp = PeftModel.from_pretrained(model_fp, args.baseline_model_adapter_path, is_trainable=False)


    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model_fp)


    # load LoftQ / CloQ  model
    model_q = AutoModelForCausalLM.from_pretrained(
        args.quant_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        ),
        device_map={"": "cuda:0"},
    )

    # base = AutoGPTQForCausalLM.from_quantized(
    #     model_path,
    #     device="cuda:0",
    #     use_safetensors=True,
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    # )

    # base = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     # device_map="auto",
    #     device_map={"": "cuda:0"},
    #     trust_remote_code=True,
    #     torch_dtype="auto"
    # )

    # base = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     # device_map="auto",
    #     quantization_config = GPTQConfig(
    #         bits=4,
    #         group_size=128,
    #         dataset="wikitext2",
    #         desc_act=False,
    #     ),

    #     device_map={"": "cuda:0"},
    #     trust_remote_code=True,
    #     torch_dtype="auto"
    # )

    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model_q)


    model_q = PeftModel.from_pretrained(model_q, args.quant_adapter_path, subfolder=args.quant_adapter_subfolder, is_trainable=False)

    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model_q)


    # Register hooks
    fp_activations, q_activations = {}, {}
    register_hooks(model_fp, fp_activations)
    register_hooks(model_q, q_activations)



    # # debugging
    print(fp_activations.keys())
    print(q_activations.keys())

    

    


    # Calibration inputs
    ds = load_dataset("gsm8k", "main")["test"]
    questions = [f"{ex['question']} The answer is" for ex in ds.select(range(args.num_inputs))]
    inputs = tokenizer(questions, return_tensors="pt", padding=True).to("cuda:0")

    # Run forward pass
    with torch.no_grad():
        _ = model_fp(**inputs)
        _ = model_q(**inputs)


    # import pdb; pdb.set_trace()


    # Compute difference
    records = []
    for name in fp_activations:
        fp_list = fp_activations[name]
        q_list = q_activations[name.replace("fp/", "q/")]
        for i in range(len(fp_list)):
            a = fp_list[i]
            b = q_list[i]
            score = compute_diff_metrics(a, b, metric=args.metric)
            records.append({
                "layer": name.replace("fp/", ""),
                "sample_idx": i,
                args.metric: score
            })

    # Save
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved activation difference to {args.output_csv}")


if __name__ == "__main__":
    main()
