import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
from kernels.fp16_gemm import fp16_gemm_kernel

os.environ['TRITON_CPU_BACKEND'] = '1'

M, N, K = 512, 512, 512
a = torch.randn((M, K), dtype=torch.float16, device='cpu')
b = torch.randn((K, N), dtype=torch.float16, device='cpu')
c = torch.empty((M, N), dtype=torch.float16, device='cpu')

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32

grid = lambda META: (
    triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
)

# 触发一次 JIT 编译并获取 compiled kernel 对象
compiled_fn = fp16_gemm_kernel[grid](
    a, b, c,
    M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
)

print("\n=== JIT 编译完成 ===")
out_dir = os.path.join(os.path.dirname(__file__), "ir_dumps")
os.makedirs(out_dir, exist_ok=True)

extension_map = {
    'source': '01_source.txt',
    'ttir': '02_ttir.mlir',
    'ttcir': '03_ttcir.mlir',
    'tttcir': '04_tttcir.mlir',
    'llir': '05_llvm.ll',
    'asm': '06_arm64.s',
    'so': '07_kernel.so',
}

for dev, cache_info in fp16_gemm_kernel.device_caches.items():
    print(f"Device: {dev}")
    kernel_dict = cache_info[0]
    for key, compiled_k in kernel_dict.items():
        print(f"Compiled kernel found: {type(compiled_k)}")
        if hasattr(compiled_k, 'asm'):
            print("asm keys:", list(compiled_k.asm.keys()))
            for ir_name, ir_code in compiled_k.asm.items():
                filename = extension_map.get(ir_name, f"dump_{ir_name}.txt")
                out_file = os.path.join(out_dir, filename)
                with open(out_file, "w") as f:
                    f.write(str(ir_code))
                print(f"  -> 已成功保存 {ir_name} 至 {out_file} (长度: {len(str(ir_code))} 字符)")



