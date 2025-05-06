import argparse
import os


import torch
from transformers import AutoModelForCausalLM, GPTQConfig







def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with GPTQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
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
        "--save_dir",
        type=str,
        default="./model_zoo/gptq/",
        help="model save directory",
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group_size (`int`, *optional*, defaults to 128):The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='wikitext2',
        help="dataset (`Union[List[str]]`, *optional*):The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper ['wikitext2','c4','c4-new']",
    )
    args = parser.parse_args()
    return args


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




def gptq_quantization():

    args = arg_parse()


    ## preprocess를...해야하나.....

    quantization_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        dataset=args.dataset,
        desc_act=False,
    )


    quant_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=quantization_config, device_map='auto')


    # save GPTQ quant model
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit"
    quant_model_dir = os.path.join(args.save_dir, model_name)


    quant_model.save_pretrained(quant_model_dir)
    # tokenizer.save_pretrained(save_dir)  # tokenizer도 같이 저장해두면 편함
    print_model(quant_model,"GPTQ quantization model")






if __name__ == "__main__":
    gptq_quantization()