import torch
import os
import argparse
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from safetensors.torch import load_file
from peft import PeftModel

import bitsandbytes as bnb



def flatten_state_dict(state_dict):
    flat_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # Strip common PEFT/bnb prefixes
            k = k.replace("base_model.model.model.", "")
            k = k.replace(".base_layer", "")  # bnb의 내부 구조
            flat_dict[k] = v
    return flat_dict


# full precision: model.layers.0.self_attn.q_proj.weight
# quantized    : layers.0.self_attn.q_proj.weight
# adapter keys : layers.0.self_attn.q_proj.lora_A.weight

# key 동기화
def normalize_key(name):
    return name.replace("model.", "").replace("base_model.model.model.", "").replace(".base_layer", "")


def dequantize_q(q, absmax, quant_map, out_shape, dtype=torch.bfloat16, blocksize=64, quant_type='nf4'):
    out = torch.empty(out_shape, dtype=dtype, device=q.device)
    q = q.contiguous()
    absmax = absmax.contiguous()
    quant_map = quant_map.contiguous()

    # monkeypatch quant_map into global BNB codebook
    if quant_type == 'nf4':
        bnb.functional.nf4_codebook = quant_map.to(q.device)
    elif quant_type == 'fp4':
        bnb.functional.fp4_codebook = quant_map.to(q.device)
    else:
        raise ValueError("Unknown quant_type")

    deq = bnb.functional.dequantize_4bit(
        A=q,
        absmax=absmax,
        out=out,
        blocksize=blocksize,
        quant_type=quant_type
    )
    return deq


def compute_weight_diff(fp_weights, quant_state_dict, adapter_weights):
    results = []


    for k in quant_state_dict:
        if "down" in k:
            print(k, quant_state_dict[k].shape)



    # print("[DEBUG] Available keys in fp:", list(fp_weights.keys())[:5])
    # print("[DEBUG] Available keys in quant:", list(quant_state_dict.keys())[:5])
    # print("[DEBUG] Available keys in adapter:", list(adapter_weights.keys())[:5])



    for name, fp_weight in fp_weights.items():
        if not name.endswith("weight"):
            continue

        name_norm = normalize_key(name)


        q_key = name_norm  # ex: layers.0.self_attn.q_proj.weight
        absmax_key = f"{name_norm}.absmax"
        quantmap_key = f"{name_norm}.quant_map"
        quantstate_key = f"{name_norm}.quant_state.bitsandbytes__nf4"  # key 그대로!



        q = quant_state_dict.get(q_key)
        absmax = quant_state_dict.get(absmax_key)
        quant_map = quant_state_dict.get(quantmap_key)
        quant_state = quant_state_dict  .get(quantstate_key)

        A = adapter_weights.get(name_norm.replace(".weight", ".lora_A.weight"))
        B = adapter_weights.get(name_norm.replace(".weight", ".lora_B.weight"))
        scaling = adapter_weights.get(name_norm.replace(".weight", ".scaling"), torch.tensor(1.0))


        if A is None or B is None:
            print(f"[SKIP] {name_norm} - No LoRA adapter attached")
            continue



        # target shape is usually (out_dim, in_dim) → from A @ B
        target_shape = (B.shape[0], A.shape[-1])

        Q = dequantize_q(q, absmax, quant_map, target_shape)


        # if q is not None and absmax is not None and quant_map is not None:
        #     Q = bnb.functional.dequantize_4bit(q, quant_state)
        #     Q = Q.reshape(4096, 4096)  # ⬅️ reshape 필수!!!

        # import pdb; pdb.set_trace()

        if Q is not None and A is not None and B is not None:
            Q = Q.float().cpu()
            A = A.cpu()
            B = B.cpu()
            scaling = scaling.cpu()
            Q_plus_AB = Q + scaling * (B @ A)

            if Q_plus_AB.shape == fp_weight.shape:
                fp = fp_weight.float().cpu()
                mse = F.mse_loss(Q_plus_AB.view(-1), fp.view(-1)).item()
                cosine = F.cosine_similarity(Q_plus_AB.view(-1), fp.view(-1), dim=0).item()
                results.append({
                    "parameter": name,
                    "mse": mse,
                    "cosine_similarity": cosine
                })

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_model_path", type=str, required=True)
    parser.add_argument("--loftq_model_path", type=str, required=True)
    parser.add_argument("--loftq_subfolder", type=str, default="loftq_init")
    parser.add_argument("--output_csv", type=str, default="weight_diff_loftq_vs_fp.csv")
    args = parser.parse_args()

    print("🔹 Loading full-precision model...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    fp_state = flatten_state_dict(fp_model.state_dict())

    print("🔹 Loading LoFTQ quantized model (bnb 4bit)...")
    quant_model = AutoModelForCausalLM.from_pretrained(
        args.loftq_model_path,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        ),
    )
    quant_peft = PeftModel.from_pretrained(quant_model, args.loftq_model_path, subfolder=args.loftq_subfolder, is_trainable=False)

    print("🔹 Loading LoRA adapter weights from safetensor...")
    safetensor_path = os.path.join(args.loftq_model_path, args.loftq_subfolder, "adapter_model.safetensors")
    adapter_sd = load_file(safetensor_path)
    adapter_sd = flatten_state_dict(adapter_sd)

    # # Mapping: name → Linear module
    # print("🔹 Mapping quantized modules...")
    # quant_modules = {}
    # for name, module in quant_peft.named_modules():
    #     if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
    #         quant_modules[name] = module

    quant_state = flatten_state_dict(quant_peft.state_dict())


    print("🔹 Computing differences...")
    diffs = compute_weight_diff(fp_state, quant_state, adapter_sd)

    print(f"🔹 Saving to {args.output_csv}...")
    pd.DataFrame(diffs).to_csv(args.output_csv, index=False)
    print("✅ Done! Comparison complete.")


if __name__ == "__main__":
    main()
