"""
测试 INT8 在真实 LLM 场景下的表现

结论: INT8 的优势不在于计算速度，而在于:
1. 减少内存占用
2. 减少内存带宽 (对于 memory-bound 的 decode 阶段)
"""
import torch
import time

print("=" * 70)
print("INT8 Real Scenario Test")
print("=" * 70)

device = 'cuda'
n_iter = 100

# LLM 常见尺寸
test_cases = [
    # (batch*seq, in_features, out_features, 场景)
    (1, 256, 1024, "Decode: bs=1, rank=256, head_dim*n_heads=1024"),
    (1, 512, 2048, "Decode: bs=1, rank=512, head_dim*n_heads=2048"),
    (512, 256, 1024, "Prefill: seq=512, rank=256"),
    (1024, 256, 1024, "Prefill: seq=1024, rank=256"),
    (2048, 256, 1024, "Prefill: seq=2048, rank=256"),
    (4096, 256, 1024, "Prefill: seq=4096, rank=256"),
    (8192, 512, 2048, "Prefill: seq=8192, rank=512 (大矩阵)"),
]

print(f"\n{'Scenario':<50} {'FP16(ms)':<12} {'INT8(ms)':<12} {'Speedup':<10}")
print("-" * 90)

for M, K, N, desc in test_cases:
    # FP16
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(x_fp16, w_fp16.T)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(x_fp16, w_fp16.T)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iter * 1000
    
    # INT8
    x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    w_int8_T = w_int8.T.contiguous()
    
    # Warmup
    for _ in range(10):
        _ = torch._int_mm(x_int8, w_int8_T)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch._int_mm(x_int8, w_int8_T)
    torch.cuda.synchronize()
    int8_time = (time.perf_counter() - start) / n_iter * 1000
    
    speedup = fp16_time / int8_time
    print(f"{desc:<50} {fp16_time:<12.4f} {int8_time:<12.4f} {speedup:<10.2f}x")

print("\n" + "=" * 70)
print("Analysis")
print("=" * 70)
print("""
观察结果:
1. 小矩阵 (decode 场景): INT8 可能更慢，因为 kernel launch 开销
2. 大矩阵 (prefill 场景): INT8 可能持平或略快

INT8 的真正价值:
1. 内存节省: INT8 权重占用减半
2. KV Cache 压缩: 存储 INT8 latent 而不是 FP16
3. Memory-bound 场景: 当瓶颈是内存带宽时，INT8 有优势

对于你的场景 (低秩 Value 重建):
- 如果计算是瓶颈 (compute-bound): FP16 可能更快
- 如果内存是瓶颈 (memory-bound): INT8 可能有优势
- 最大的收益来自于 KV Cache 压缩，而不是计算加速
""")

# 测试内存带宽场景
print("\n" + "=" * 70)
print("Memory Bandwidth Test (模拟 decode 阶段)")
print("=" * 70)
print("在 decode 阶段，每次只处理 1 个 token，但需要读取整个 KV Cache")
print("这时内存带宽是瓶颈，INT8 可以减少一半的内存读取")

# 模拟 decode: 需要读取整个权重矩阵，但只计算 1 行
for seq_len in [512, 1024, 2048, 4096]:
    K, N = 256, 1024
    
    # FP16: 读取 FP16 权重
    w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)
    x_fp16 = torch.randn(1, K, device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(x_fp16, w_fp16.T)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iter * 1000
    
    # INT8: 读取 INT8 权重 (一半的内存带宽)
    w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    w_int8_T = w_int8.T.contiguous()
    x_int8 = torch.randint(-128, 127, (1, K), dtype=torch.int8, device=device)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch._int_mm(x_int8, w_int8_T)
    torch.cuda.synchronize()
    int8_time = (time.perf_counter() - start) / n_iter * 1000
    
    # 计算理论内存读取量
    fp16_mem = (N * K * 2 + 1 * K * 2) / 1024 / 1024  # MB
    int8_mem = (N * K * 1 + 1 * K * 1) / 1024 / 1024  # MB
    
    print(f"Decode (seq={seq_len}): FP16={fp16_time:.4f}ms, INT8={int8_time:.4f}ms, "
          f"Speedup={fp16_time/int8_time:.2f}x, Mem: FP16={fp16_mem:.2f}MB vs INT8={int8_mem:.2f}MB")
