import torch
from mamba_ssm import Mamba2

m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64).cuda()
for T in [257, 513, 1025, 2049]:
    x = torch.randn(2, T, 256).cuda()
    with torch.no_grad():
        out = m(x)
    print(f"T={T}, out.shape={out.shape}")
