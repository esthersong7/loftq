import torch

@torch.no_grad()
def cloq_init(delta_W: torch.Tensor, num_bits: int, reduced_rank: int, hessian: torch.Tensor):
    """
    delta_W: (out_features, in_features)
    hessian: (batch_size, in_features)
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
    # H = activation.T @ activation
    m = hessian.size(0)
    lam = 0.01 * torch.trace(hessian) / m
    hessian += lam * torch.eye(m, device=hessian.device, dtype=hessian.dtype)

    # 4. SVD on H
    U_H, Sigma_H, _ = torch.linalg.svd(hessian)
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
