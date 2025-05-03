from peft import LoftQConfig

# /home/esthersong/miniconda3/envs/loftq/lib/python3.10/site-packages/peft/tuners/lora/config.py


class CloQConfig(LoftQConfig):
    def __init__(self, activation_dict: dict, **kwargs):
        super().__init__(**kwargs)
        self.activation_dict = activation_dict

        # calibration dataset? or activation dict?








@torch.no_grad()
def cloq_init(delta_W: torch.Tensor, num_bits: int, reduced_rank: int, activation: torch.Tensor):
    """
    weight: (out_features, in_features)
    activation: (batch_size, in_features)
    """
    device = delta_W.device
    dtype = delta_W.dtype

    # # 1. Quantization
    # quantizer = NFQuantizer(num_bits=num_bits, device=device, method="normal", block_size=64)
    # quantized_weight, max_abs, shape = quantizer.quantize_block(weight)
    # dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)

    # # 2. Residual
    # delta_W = weight - dequantized_weight

    # 3. Gram matrix H = X^T X
    H = activation.T @ activation
    m = H.size(0)
    lam = 0.01 * torch.trace(H) / m
    H += lam * torch.eye(m, device=H.device, dtype=H.dtype)

    # 4. SVD on H
    U_H, Sigma_H, _ = torch.linalg.svd(H)
    R = torch.diag(Sigma_H.sqrt()) @ U_H.T

    # 5. SVD on R @ delta_W
    RAW = R @ delta_W
    U_r, Sigma_r, Vh_r = torch.linalg.svd(RAW, full_matrices=False)
    U_r = U_r[:, :reduced_rank]
    Sigma_r = Sigma_r[:reduced_rank]
    V_r = Vh_r[:reduced_rank, :]

    A = torch.linalg.solve(R, U_r @ torch.diag(Sigma_r))
    B = V_r.T

    return A, B
