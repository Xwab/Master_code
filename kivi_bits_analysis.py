"""
KIVI 量化实际精度分析

分析对于序列长度 L 的 KV 缓存，使用 KIVI-2bit 量化后，
由于存储量化系数和保留全精度 residual token 的影响，
实际平均每个元素的数据精度是多少 bit。
"""

import math

def calculate_kivi_avg_bits(
    seq_len: int,
    n_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 32,
    original_bits: int = 16,  # fp16
) -> dict:
    """
    计算 KIVI 量化后的实际平均精度。
    
    Args:
        seq_len: 序列长度 L
        n_bits: 量化位数 (2, 3, 4, etc.)
        group_size: 量化分组大小
        residual_length: 保持全精度的最近 token 数
        original_bits: 原始精度 (fp16 = 16)
    
    Returns:
        包含详细分析的字典
    """
    L = seq_len
    G = group_size
    R = residual_length
    
    # 计算量化和残差的 token 数
    if L <= R:
        # 所有 token 都在 residual 中，不量化
        return {
            "seq_len": L,
            "n_bits": n_bits,
            "group_size": G,
            "residual_length": R,
            "L_quant": 0,
            "L_res": L,
            "avg_bits": float(original_bits),
            "compression_ratio": 1.0,
            "effective_ratio": original_bits / original_bits,
            "note": "All tokens in residual, no quantization"
        }
    
    # 需要量化的 token 数 (对齐到 group_size)
    L_quant = ((L - R) // G) * G
    L_res = L - L_quant
    
    if L_quant == 0:
        return {
            "seq_len": L,
            "n_bits": n_bits,
            "group_size": G,
            "residual_length": R,
            "L_quant": 0,
            "L_res": L,
            "avg_bits": float(original_bits),
            "compression_ratio": 1.0,
            "effective_ratio": original_bits / original_bits,
            "note": "Quantized tokens < group_size, no quantization"
        }
    
    # 量化部分的组数
    n_groups = L_quant // G
    
    # =========================================
    # 存储分析 (per element, 假设特征维度 D)
    # =========================================
    
    # 量化数据存储: L_quant × n_bits
    quant_data_bits = L_quant * n_bits
    
    # scale 和 zero_point 存储:
    # - Key (per-channel): D 个 channel，每个 channel 有 n_groups 组
    #   每组需要 scale (16bit) + zero_point (16bit) = 32bit
    #   总共: D × n_groups × 32 bit
    #   平均到每个元素: n_groups × 32 / L_quant = 32 / G
    #
    # - Value (per-token): L_quant 个 token，每个 token 有 D/G 组
    #   每组需要 scale + zero_point = 32bit
    #   总共: L_quant × (D/G) × 32 bit
    #   平均到每个元素: 32 / G
    #
    # 两者平均到每元素的 overhead 相同: 32/G
    
    quant_overhead_per_element = 32.0 / G  # scale + zero_point 的开销
    
    # 量化部分每元素的实际 bit
    bits_per_quant_element = n_bits + quant_overhead_per_element
    
    # 残差部分: 全精度
    residual_bits = L_res * original_bits
    
    # 总 bit 数 (per D features)
    total_bits = L_quant * bits_per_quant_element + residual_bits
    
    # 平均每元素的 bit
    avg_bits = total_bits / L
    
    # 压缩比
    original_total = L * original_bits
    compression_ratio = total_bits / original_total
    
    return {
        "seq_len": L,
        "n_bits": n_bits,
        "group_size": G,
        "residual_length": R,
        "L_quant": L_quant,
        "L_res": L_res,
        "n_groups": n_groups,
        "bits_per_quant_element": bits_per_quant_element,
        "quant_overhead_per_element": quant_overhead_per_element,
        "avg_bits": avg_bits,
        "compression_ratio": compression_ratio,
        "memory_saving": 1 - compression_ratio,
    }


def print_analysis(result: dict):
    """打印分析结果"""
    print("=" * 60)
    print(f"序列长度 L = {result['seq_len']}")
    print(f"量化位数 = {result['n_bits']} bit")
    print(f"Group size = {result['group_size']}")
    print(f"Residual length = {result['residual_length']}")
    print("-" * 60)
    print(f"量化的 token 数: {result['L_quant']}")
    print(f"全精度残差 token 数: {result['L_res']}")
    if 'n_groups' in result:
        print(f"量化分组数: {result['n_groups']}")
        print(f"量化 overhead (scale+zero/G): {result['quant_overhead_per_element']:.4f} bit/element")
        print(f"量化部分每元素实际 bit: {result['bits_per_quant_element']:.4f}")
    print("-" * 60)
    print(f"★ 平均每元素精度: {result['avg_bits']:.4f} bit")
    print(f"★ 压缩比: {result['compression_ratio']:.4f} ({result['compression_ratio']*100:.2f}%)")
    print(f"★ 内存节省: {result.get('memory_saving', 0)*100:.2f}%")
    print("=" * 60)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KIVI-2bit 量化实际精度分析")
    print("=" * 60 + "\n")
    
    # 默认参数: 2bit, group_size=128, residual_length=32
    print("【默认参数】n_bits=2, group_size=128, residual_length=32\n")
    
    # 不同序列长度
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    print("-" * 80)
    print(f"{'序列长度':<10} {'量化token':<10} {'残差token':<10} {'平均bit':<12} {'压缩比':<12} {'节省':<10}")
    print("-" * 80)
    
    for L in seq_lengths:
        result = calculate_kivi_avg_bits(L, n_bits=2, group_size=128, residual_length=32)
        print(f"{L:<10} {result['L_quant']:<10} {result['L_res']:<10} "
              f"{result['avg_bits']:<12.4f} {result['compression_ratio']*100:<11.2f}% "
              f"{result.get('memory_saving', 0)*100:<9.2f}%")
    
    print("-" * 80)
    print()
    
    # 详细分析几个典型长度
    print("\n【详细分析】\n")
    for L in [512, 2048, 8192]:
        result = calculate_kivi_avg_bits(L, n_bits=2, group_size=128, residual_length=32)
        print_analysis(result)
    
    # 公式总结
    print("\n" + "=" * 60)
    print("【公式总结】")
    print("=" * 60)
    print("""
对于序列长度 L，使用 KIVI n-bit 量化:

  L_quant = floor((L - R) / G) × G   (量化的 token 数)
  L_res = L - L_quant                (全精度残差 token 数)

  平均每元素精度 = [L_quant × (n + 32/G) + L_res × 16] / L

其中:
  - n = n_bits (量化位数)
  - G = group_size (分组大小)
  - R = residual_length (残差长度)
  - 32/G 是每个量化元素的 scale+zero_point 开销
  
当 L → ∞ 时:
  avg_bits → n + 32/G = 2 + 32/128 = 2.25 bit (对于 2bit, G=128)
""")
    
    # 不同 group_size 的影响
    print("\n【不同 group_size 的影响】(L=4096, n_bits=2, residual=32)\n")
    print("-" * 60)
    print(f"{'group_size':<15} {'overhead':<15} {'avg_bits':<15} {'压缩比':<15}")
    print("-" * 60)
    for G in [32, 64, 128, 256, 512]:
        result = calculate_kivi_avg_bits(4096, n_bits=2, group_size=G, residual_length=32)
        overhead = 32.0 / G
        print(f"{G:<15} {overhead:<15.4f} {result['avg_bits']:<15.4f} {result['compression_ratio']*100:.2f}%")
    print("-" * 60)
    
    # 不同 n_bits 的影响
    print("\n【不同 n_bits 的影响】(L=4096, group_size=128, residual=32)\n")
    print("-" * 60)
    print(f"{'n_bits':<15} {'理论最低':<15} {'实际avg_bits':<15} {'压缩比':<15}")
    print("-" * 60)
    for n in [2, 3, 4, 8]:
        result = calculate_kivi_avg_bits(4096, n_bits=n, group_size=128, residual_length=32)
        theoretical = n + 32/128  # 忽略 residual 的理论最低
        print(f"{n:<15} {theoretical:<15.4f} {result['avg_bits']:<15.4f} {result['compression_ratio']*100:.2f}%")
    print("-" * 60)
    
    # 不同 residual_length 的影响
    print("\n【不同 residual_length 的影响】(L=2048, n_bits=2, group_size=128)\n")
    print("-" * 60)
    print(f"{'residual':<15} {'L_quant':<15} {'L_res':<15} {'avg_bits':<15}")
    print("-" * 60)
    for R in [0, 32, 64, 128, 256, 512]:
        result = calculate_kivi_avg_bits(2048, n_bits=2, group_size=128, residual_length=R)
        print(f"{R:<15} {result['L_quant']:<15} {result['L_res']:<15} {result['avg_bits']:<15.4f}")
    print("-" * 60)
