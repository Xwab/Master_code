"""
INT8 量化操作模块

包含:
1. INT8Quantizer - 返回量化值和 scale 的量化器
2. int8_linear - INT8 线性层运算
3. CUDA kernel (如果可用)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import os

# ============================================================================
# 1. INT8 量化器 (返回量化值，不反量化)
# ============================================================================

class INT8Quantizer:
    """
    INT8 对称量化器
    
    与 Quantizer 的区别:
    - Quantizer: 返回反量化后的 float 值 (fake quant)
    - INT8Quantizer: 返回 int8 值和 scale (真正的量化)
    """
    
    @staticmethod
    @torch.no_grad()
    def quantize_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-token 量化 (用于 Value latent)
        
        Args:
            x: (batch, seq, hidden) float tensor
        
        Returns:
            x_int8: (batch, seq, hidden) int8 tensor
            scale: (batch, seq, 1) float tensor
        """
        # 计算每个 token 的最大绝对值
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 127.0
        
        # 量化
        x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
        
        return x_int8, scale
    
    @staticmethod
    @torch.no_grad()
    def quantize_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-channel 量化 (用于重建矩阵)
        
        Args:
            w: (out_features, in_features) float tensor
        
        Returns:
            w_int8: (out_features, in_features) int8 tensor
            scale: (out_features, 1) float tensor
        """
        # 计算每个输出通道的最大绝对值
        amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 127.0
        
        # 量化
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        
        return w_int8, scale
    
    @staticmethod
    @torch.no_grad()
    def dequantize(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return x_int8.float() * scale


# ============================================================================
# 2. INT8 矩阵乘法 (Python 实现)
# ============================================================================

def int8_linear_python(
    x: torch.Tensor,           # (batch, seq, in_features) float16
    w: torch.Tensor,           # (out_features, in_features) float16
    bias: Optional[torch.Tensor] = None,
    use_native_int8: bool = True
) -> torch.Tensor:
    """
    INT8 线性层: 在线量化 + INT8 matmul + 反量化
    
    流程:
    1. 量化 x (per-token)
    2. 量化 w (per-channel) [可以预计算]
    3. INT8 matmul
    4. 反量化
    
    Args:
        x: 输入 (batch, seq, in_features)
        w: 权重 (out_features, in_features)
        bias: 偏置 (out_features)
        use_native_int8: 是否使用 torch._int_mm
    
    Returns:
        output: (batch, seq, out_features)
    """
    batch_size, seq_len, in_features = x.shape
    out_features = w.shape[0]
    
    # Step 1: 量化输入 (per-token)
    x_int8, x_scale = INT8Quantizer.quantize_per_token(x.float())
    
    # Step 2: 量化权重 (per-channel)
    w_int8, w_scale = INT8Quantizer.quantize_per_channel(w.float())
    
    # Step 3: INT8 矩阵乘法
    if use_native_int8 and hasattr(torch, '_int_mm') and x.is_cuda:
        # 使用 torch._int_mm (需要 2D 输入)
        x_2d = x_int8.view(-1, in_features).contiguous()
        w_T = w_int8.T.contiguous()
        
        try:
            out_int32 = torch._int_mm(x_2d, w_T)
            out = out_int32.view(batch_size, seq_len, out_features).float()
        except:
            # Fallback
            out = torch.matmul(x_int8.float(), w_int8.T.float())
    else:
        # Fallback: float32
        out = torch.matmul(x_int8.float(), w_int8.T.float())
    
    # Step 4: 反量化
    # out = out * x_scale * w_scale^T
    out = out * x_scale * w_scale.T
    
    # Step 5: 加偏置
    if bias is not None:
        out = out + bias
    
    return out.to(x.dtype)


class INT8LinearWithPrequantizedWeight(nn.Module):
    """
    预量化权重的 INT8 线性层
    
    权重在初始化时量化一次，推理时只需要量化输入
    """
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        
        # 预量化权重
        w_int8, w_scale = INT8Quantizer.quantize_per_channel(weight.float())
        
        self.register_buffer('w_int8', w_int8.contiguous())
        self.register_buffer('w_scale', w_scale.contiguous())
        
        if bias is not None:
            self.register_buffer('bias', bias.clone())
        else:
            self.register_buffer('bias', None)
        
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, in_features)
        Returns:
            out: (batch, seq, out_features)
        """
        batch_size, seq_len, _ = x.shape
        
        # 量化输入 (per-token)
        x_int8, x_scale = INT8Quantizer.quantize_per_token(x.float())
        x_int8 = x_int8.contiguous()
        
        # INT8 matmul
        if hasattr(torch, '_int_mm') and x.is_cuda:
            try:
                x_2d = x_int8.view(-1, self.in_features)
                out_int32 = torch._int_mm(x_2d, self.w_int8.T.contiguous())
                out = out_int32.view(batch_size, seq_len, self.out_features).float()
            except:
                out = torch.matmul(x_int8.float(), self.w_int8.T.float())
        else:
            out = torch.matmul(x_int8.float(), self.w_int8.T.float())
        
        # 反量化
        out = out * x_scale * self.w_scale.T
        
        if self.bias is not None:
            out = out + self.bias
        
        return out.to(x.dtype)


# ============================================================================
# 3. CUDA Kernel (Triton 实现)
# ============================================================================

# 检查 Triton 是否可用
TRITON_AVAILABLE = False
TRITON_ERROR = None
try:
    import triton
    import triton.language as tl
    # 测试基本功能
    _ = tl.constexpr
    TRITON_AVAILABLE = True
except ImportError as e:
    TRITON_ERROR = f"ImportError: {e}"
except Exception as e:
    TRITON_ERROR = f"Error: {e}"

def get_triton_status():
    """获取 Triton 状态"""
    return {
        'available': TRITON_AVAILABLE,
        'error': TRITON_ERROR,
        'version': getattr(triton, '__version__', 'unknown') if TRITON_AVAILABLE else None
    }


if TRITON_AVAILABLE:
    
    @triton.jit
    def _int8_quantize_per_token_kernel(
        x_ptr,           # 输入指针 (float)
        x_int8_ptr,      # 输出 int8 指针
        scale_ptr,       # 输出 scale 指针
        n_cols,          # 列数
        BLOCK_SIZE: tl.constexpr,
    ):
        """Per-token 量化 kernel"""
        row_idx = tl.program_id(0)
        
        # 加载一行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        row_start = row_idx * n_cols
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # 计算 scale
        amax = tl.max(tl.abs(x))
        amax = tl.maximum(amax, 1e-8)
        scale = amax / 127.0
        
        # 量化 (使用 floor(x + 0.5) 代替 rint)
        x_scaled = x / scale
        x_int8 = tl.floor(x_scaled + 0.5)  # 四舍五入
        x_int8 = tl.minimum(tl.maximum(x_int8, -128.0), 127.0)
        
        # 存储
        tl.store(x_int8_ptr + row_start + col_offsets, x_int8.to(tl.int8), mask=mask)
        tl.store(scale_ptr + row_idx, scale)
    
    
    @triton.jit
    def _int8_matmul_kernel(
        # 指针
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,
        # 矩阵维度
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Block sizes
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        INT8 矩阵乘法 kernel
        
        C = A @ B * a_scale * b_scale^T
        
        A: (M, K) int8, a_scale: (M, 1)
        B: (K, N) int8, b_scale: (N, 1)  # 注意: B 已经是转置后的
        C: (M, N) float
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # 计算 block 起始位置
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # 初始化累加器
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        
        # 循环累加
        for k in range(0, K, BLOCK_SIZE_K):
            k_offs = k + offs_k
            
            # 加载 A block
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0)
            
            # 加载 B block (B 是 K x N 格式)
            b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0)
            
            # INT8 累加
            acc += tl.dot(a, b)
        
        # 加载 scales
        a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
        
        # 反量化: c = acc * a_scale[:, None] * b_scale[None, :]
        c = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        
        # 存储结果
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    
    
    def int8_quantize_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 Triton 的 per-token 量化
        
        Args:
            x: (batch * seq, hidden) float tensor
        Returns:
            x_int8: (batch * seq, hidden) int8 tensor
            scale: (batch * seq,) float tensor
        """
        assert x.is_cuda, "Triton kernel requires CUDA"
        
        M, K = x.shape
        
        x_int8 = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # 选择 block size
        BLOCK_SIZE = triton.next_power_of_2(K)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)
        
        grid = (M,)
        _int8_quantize_per_token_kernel[grid](
            x, x_int8, scale,
            K,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return x_int8, scale.unsqueeze(1)
    
    
    def int8_matmul_triton(
        a_int8: torch.Tensor,   # (M, K) int8
        b_int8: torch.Tensor,   # (N, K) int8 (权重, 需要转置)
        a_scale: torch.Tensor,  # (M, 1) float
        b_scale: torch.Tensor,  # (N, 1) float
    ) -> torch.Tensor:
        """
        使用 Triton 的 INT8 矩阵乘法
        
        Returns:
            c: (M, N) float
        """
        assert a_int8.is_cuda, "Triton kernel requires CUDA"
        
        M, K = a_int8.shape
        N = b_int8.shape[0]
        
        # B 需要从 (N, K) 转为 (K, N)
        b_T = b_int8.T.contiguous()
        
        c = torch.empty((M, N), dtype=torch.float32, device=a_int8.device)
        
        # Block sizes
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        _int8_matmul_kernel[grid](
            a_int8, b_T, c,
            a_scale.squeeze(-1), b_scale.squeeze(-1),
            M, N, K,
            a_int8.stride(0), a_int8.stride(1),
            b_T.stride(0), b_T.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return c


# ============================================================================
# 4. 统一接口
# ============================================================================

class INT8LinearFunction:
    """INT8 线性层的统一接口"""
    
    @staticmethod
    def apply(
        x: torch.Tensor,
        w_int8: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        backend: str = "auto"
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, in_features) float
            w_int8: (out_features, in_features) int8 (预量化)
            w_scale: (out_features, 1) float
            bias: (out_features,) float
            backend: "triton", "torch", "auto"
        
        Returns:
            out: (batch, seq, out_features)
        """
        batch_size, seq_len, in_features = x.shape
        out_features = w_int8.shape[0]
        
        # 选择后端
        if backend == "auto":
            if TRITON_AVAILABLE and x.is_cuda:
                backend = "triton"
            elif hasattr(torch, '_int_mm') and x.is_cuda:
                backend = "torch"
            else:
                backend = "fallback"
        
        # 量化输入
        x_flat = x.view(-1, in_features).float()
        
        if backend == "triton" and TRITON_AVAILABLE:
            x_int8, x_scale = int8_quantize_triton(x_flat)
            out = int8_matmul_triton(x_int8, w_int8, x_scale, w_scale)
        elif backend == "torch" and hasattr(torch, '_int_mm'):
            x_int8, x_scale = INT8Quantizer.quantize_per_token(x_flat)
            x_int8 = x_int8.contiguous()
            try:
                out_int32 = torch._int_mm(x_int8, w_int8.T.contiguous())
                out = out_int32.float() * x_scale * w_scale.T
            except:
                out = torch.matmul(x_int8.float(), w_int8.T.float()) * x_scale * w_scale.T
        else:
            x_int8, x_scale = INT8Quantizer.quantize_per_token(x_flat)
            out = torch.matmul(x_int8.float(), w_int8.T.float()) * x_scale * w_scale.T
        
        out = out.view(batch_size, seq_len, out_features)
        
        if bias is not None:
            out = out + bias
        
        return out.to(x.dtype)


# ============================================================================
# 5. 便捷模块
# ============================================================================

def disable_triton():
    """禁用 Triton (如果有兼容性问题)"""
    global TRITON_AVAILABLE
    TRITON_AVAILABLE = False
    print("Triton disabled. Using torch backend.")


class INT8Linear(nn.Module):
    """
    完整的 INT8 线性层
    
    使用方法:
        layer = INT8Linear(weight, bias)
        output = layer(input)
    
    如果 Triton 有问题，可以禁用:
        from int8_ops import disable_triton
        disable_triton()
    """
    
    def __init__(
        self, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None,
        backend: str = "auto"
    ):
        super().__init__()
        
        self.backend = backend
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]
        
        # 预量化权重
        w_int8, w_scale = INT8Quantizer.quantize_per_channel(weight.float())
        self.register_buffer('w_int8', w_int8.contiguous())
        self.register_buffer('w_scale', w_scale.contiguous())
        
        if bias is not None:
            self.register_buffer('bias', bias.clone())
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return INT8LinearFunction.apply(
            x, self.w_int8, self.w_scale, self.bias, self.backend
        )
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, backend={self.backend}'


# ============================================================================
# 6. 测试函数
# ============================================================================

def test_int8_ops():
    """测试 INT8 操作"""
    print("=" * 70)
    print("INT8 Operations Test")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print(f"torch._int_mm available: {hasattr(torch, '_int_mm')}")
    
    # 测试参数
    batch_size, seq_len = 2, 512
    in_features, out_features = 256, 1024
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float16)
    w = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
    bias = torch.randn(out_features, device=device, dtype=torch.float16)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    
    # 测试 1: INT8Quantizer
    print("\n--- Test INT8Quantizer ---")
    x_int8, x_scale = INT8Quantizer.quantize_per_token(x)
    print(f"x_int8 dtype: {x_int8.dtype}, shape: {x_int8.shape}")
    print(f"x_scale shape: {x_scale.shape}")
    
    w_int8, w_scale = INT8Quantizer.quantize_per_channel(w)
    print(f"w_int8 dtype: {w_int8.dtype}, shape: {w_int8.shape}")
    print(f"w_scale shape: {w_scale.shape}")
    
    # 测试 2: INT8Linear
    print("\n--- Test INT8Linear ---")
    layer = INT8Linear(w, bias, backend="auto")
    out = layer(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    
    # 测试 3: 与 FP16 对比精度
    print("\n--- Accuracy vs FP16 ---")
    out_fp16 = torch.nn.functional.linear(x, w, bias)
    
    mse = ((out - out_fp16) ** 2).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out.flatten(), out_fp16.flatten(), dim=0
    ).item()
    
    print(f"MSE: {mse:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    # 测试 4: 速度对比
    print("\n--- Speed Benchmark ---")
    n_iter = 100
    
    # Warmup
    for _ in range(10):
        _ = layer(x)
        _ = torch.nn.functional.linear(x, w, bias)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # INT8
    import time
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = layer(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    int8_time = (time.perf_counter() - start) / n_iter * 1000
    
    # FP16
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.nn.functional.linear(x, w, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"INT8 time: {int8_time:.4f} ms")
    print(f"FP16 time: {fp16_time:.4f} ms")
    print(f"Speedup: {fp16_time / int8_time:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_int8_ops()
