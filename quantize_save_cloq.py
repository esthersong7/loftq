# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import LoftQConfig, TaskType, get_peft_model
# from safetensors import save_open
from safetensors import safe_open

# from accelerate import Accelerator

from transformers import GPTQConfig

from config import CloQConfig, LoraConfig       ## ??? iimport 어디서?





class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the full precision model.",
    )
    parser.add_argument(
        "--quant_model_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the GPTQ quantized model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in CloQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/cloq/",
        help="Lora model save directory",
    )





    args = parser.parse_args()
    return args








def get_calibration_loader(tokenizer):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch
    from torch.utils.data import DataLoader

    # 1. Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # # 2. Load tokenizer (LLaMA tokenizer로 교체 가능)
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 예시: LLaMA tokenizer로 변경 가능
    # tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize first 128 samples with context length 2048
    def tokenize(example):
        return tokenizer(
            example["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,        # l= 2048
        )

    dataset = dataset.filter(lambda x: len(x["text"]) > 0).shuffle(seed=42).select(range(128))
    tokenized = dataset.map(tokenize, batched=False)

    # 4. Convert to TensorDataset
    input_ids = torch.stack([torch.tensor(x["input_ids"]).squeeze(0) for x in tokenized])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"]).squeeze(0) for x in tokenized])

    calibration_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    calibration_loader = DataLoader(calibration_dataset, batch_size=1)  # b=128, l=2048

    # → 이 loader를 forward에 넣으면 X ∈ ℝᵇˡˣᵐ 에 해당하는 activation을 hook으로 수집할 수 있어
    return calibration_loader





def register_activation_hooks(model, hessian_dict, target_module_names):
    def get_hook(name):
        def hook(module, input, output):
            
            X = input[0].detach().reshape(-1, input[0].shape[-1])       #(b,l,m) -> (b*l,m)=(1*2048,m)
            H = X.T @ X                                                 #(m,m)
            H_cpu = H.cpu()

            if name not in hessian_dict:
                hessian_dict[name] = H_cpu
            else:
                hessian_dict[name] += H_cpu

            del H
            del X
            torch.cuda.empty_cache()
        
        return hook

    for name, module in model.named_modules():
        if any(target in name for target in target_module_names):
            module.register_forward_hook(get_hook(name))




def quantize_and_save():


    import sys
    print("🔍 PYTHONPATH check:", sys.path[0])

    args = arg_parse()

    device = torch.device("cuda:0")

    # preprocess?>?
    # MagR ??



    

    # load GPTQ quantized model

    # quantization_config = GPTQConfig(
    #     bits=args.bits,
    #     group_size=args.group_size,
    #     dataset=args.dataset,
    #     desc_act=False,
    # )


    quant_model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, device_map={"": device})
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    
    hessian_filename = f"Llama-2-7b-hf_{args.bits}bit_hessian_dict.pt"
    # hessian_path = os.path.join(args.save_dir, hessian_filename)


    if not os.path.exists(hessian_filename):
        print("No saved Hessian found — starting calibration...")

        # collect activation data X of Quantized model using forward hooks
        hessian_dict = {}
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        register_activation_hooks(quant_model, hessian_dict, target_modules)


        calibration_loader = get_calibration_loader(tokenizer)

        print("start calibration")

        quant_model.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                _ = quant_model(input_ids=input_ids, attention_mask=attention_mask)

                del input_ids
                del attention_mask
                torch.cuda.empty_cache()


        print("done calibration")

        torch.save(hessian_dict, hessian_filename)
        print(f"Saved hessian_dict to {hessian_filename}")


    # load hessian dict to CPU
    hessian_dict = torch.load(hessian_filename, map_location="cpu")
    print(f"Loaded hessian_dict from {hessian_filename}")



   
    # Q data - CPU
    quantized_weights = {name: param.data.clone().to("cpu") for name, param in quant_model.named_parameters() if param.ndim == 2}


    del quant_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    


    # Download Full Precision weights and configure LoRA
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            token=args.token,
            trust_remote_code=True,
            # device_map="auto",
            device_map={"": device},
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_CLS
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # embeddings not supported by peft
    else:
        raise NotImplementedError("Other models not supported yet.")




    # import pdb; pdb.set_trace()


    # Config of LoftQ
    cloq_config = CloQConfig(loftq_bits=args.bits, loftq_iter=args.iter, quant_dict=quantized_weights, activation_dict=hessian_dict)
    # quant dict, activation dict - both cpu


    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=args.rank,
        lora_alpha=16 if task_type is TaskType.CAUSAL_LM and args.bits == 4 else args.rank,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="loftq",
        loftq_config=cloq_config,
    )



    # Obtain LoftQ model
    # import pdb; pdb.set_trace()
    lora_model = get_peft_model(model, lora_config)         # Q, A, B with CloQ initialization
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit" + f"-{args.rank}rank"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "cloq_init")

    import pdb; pdb.set_trace()

    adapter_cfg = lora_model.peft_config["default"]
    adapter_cfg.loftq_config["quant_dict"] = {}
    adapter_cfg.loftq_config["activation_dict"] = {}



    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    print_model(base_model, "base_model")

    # convert safetensor to bin
    tensors = {}
    with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  # Don't apply LoftQ when loading again
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_dir, lora_dir = quantize_and_save()

# example command:
# python quantize_save_load.py \
# --model_name_or_path meta-llama/Llama-2-7b-hf \
# --token XXX \
# --bits 4 --iter 5 --rank 16 \
# --save_dir ./model_zoo/loftq/
