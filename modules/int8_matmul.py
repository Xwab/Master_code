"""
INT8 Matrix Multiplication for Value Reconstruction

Implements W8A8 quantization for ALinear weight and value_states,
using int8 matmul for faster reconstruction.

Supported backends:
1. torch._int_mm (PyTorch 2.0+, CUDA)
2. Fallback to fake quantization for CPU/older PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def check_int8_support():
    """Check if int8 matmul is supported."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    # Check PyTorch version
    major, minor = torch.__version__.split('.')[:2]
    if int(major) < 2:
        return False, f"PyTorch {torch.__version__} < 2.0, _int_mm not available"
    
    # Check if _int_mm exists
    if not hasattr(torch, '_int_mm'):
        return False, "torch._int_mm not found"
    
    return True, "INT8 matmul supported"


# Global flag for int8 support
INT8_SUPPORTED, INT8_STATUS = check_int8_support()
print(f"[INT8] {INT8_STATUS}")


class Int8Quantizer(nn.Module):
    """
    Symmetric INT8 quantizer for activations and weights.
    
    Quantization formula:
        x_int8 = round(x / scale)
        x_dequant = x_int8 * scale
    
    Where scale = max(|x|) / 127
    """
    
    def __init__(self, per_tensor: bool = True):
        super().__init__()
        self.per_tensor = per_tensor
    
    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8.
        
        Args:
            x: Input tensor (float16/float32)
        
        Returns:
            (x_int8, scale): Quantized tensor and scale factor
        """
        if self.per_tensor:
            # Per-tensor quantization
            scale = x.abs().max() / 127.0
            scale = scale.clamp(min=1e-5)
        else:
            # Per-channel quantization (along last dim)
            scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
            scale = scale.clamp(min=1e-5)
        
        x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
        
        return x_int8, scale
    
    @torch.no_grad()
    def dequantize(self, x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor back to float."""
        return x_int8.float() * scale


class Int8Linear(nn.Module):
    """
    INT8 Linear layer with W8A8 quantization.
    
    Weight is quantized to INT8 during initialization.
    Activation is quantized to INT8 during forward pass.
    Uses INT8 matmul for computation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_per_channel: bool = True,
        activation_per_tensor: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_per_channel = weight_per_channel
        self.activation_per_tensor = activation_per_tensor
        
        # Original weight placeholder (will be replaced with quantized)
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.activation_quantizer = Int8Quantizer(per_tensor=activation_per_tensor)
    
    @torch.no_grad()
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to INT8."""
        if self.weight_per_channel:
            # Per-channel (per output channel)
            scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
            scale = scale.clamp(min=1e-5)
        else:
            # Per-tensor
            scale = weight.abs().max() / 127.0
            scale = torch.tensor([[scale.item()]], device=weight.device)
            scale = scale.clamp(min=1e-5)
        
        weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
        
        self.weight_int8.copy_(weight_int8)
        self.weight_scale.copy_(scale)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, weight_per_channel: bool = True):
        """Create Int8Linear from existing nn.Linear."""
        int8_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            weight_per_channel=weight_per_channel,
        )
        
        # Move to same device
        int8_linear = int8_linear.to(linear.weight.device)
        
        # Quantize weight
        int8_linear.quantize_weight(linear.weight.data)
        
        # Copy bias
        if linear.bias is not None:
            int8_linear.bias.data.copy_(linear.bias.data)
        
        return int8_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with INT8 matmul.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
        
        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Flatten to 2D for matmul
        x_2d = x.view(-1, self.in_features)
        
        if INT8_SUPPORTED and x.is_cuda:
            # Use real INT8 matmul
            output = self._int8_matmul(x_2d)
        else:
            # Fallback to fake quantization
            output = self._fake_int8_matmul(x_2d)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output.to(original_dtype)
    
    def _int8_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Real INT8 matmul using torch._int_mm."""
        # Quantize activation
        x_int8, x_scale = self.activation_quantizer.quantize(x)
        
        # INT8 matmul: (M, K) @ (K, N) -> (M, N)
        # torch._int_mm expects (M, K) @ (K, N)
        # weight is (out_features, in_features), need to transpose
        weight_t = self.weight_int8.t().contiguous()  # (in_features, out_features)
        
        # Perform int8 matmul
        # Result is int32
        output_int32 = torch._int_mm(x_int8, weight_t)
        
        # Dequantize: output = output_int32 * x_scale * weight_scale
        # x_scale: scalar or (M, 1)
        # weight_scale: (out_features, 1) -> (1, out_features)
        output = output_int32.float() * x_scale * self.weight_scale.t()
        
        return output
    
    def _fake_int8_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Fake INT8 matmul (quantize-dequantize, then float matmul)."""
        # Quantize and dequantize activation
        x_int8, x_scale = self.activation_quantizer.quantize(x)
        x_dequant = x_int8.float() * x_scale
        
        # Dequantize weight
        weight_dequant = self.weight_int8.float() * self.weight_scale
        
        # Float matmul
        output = F.linear(x_dequant, weight_dequant)
        
        return output


class Int8ValueReconstructor(nn.Module):
    """
    INT8 Value Reconstructor for attention.
    
    Performs: output = value_states @ ALinear_weight.T
    Using W8A8 INT8 quantization for both value_states and ALinear weight.
    """
    
    def __init__(
        self,
        latent_dim: int,  # rank or out_features
        out_features: int,  # head_dim * num_heads
        bias: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_features = out_features
        
        # INT8 quantized ALinear
        self.int8_linear = Int8Linear(
            latent_dim,
            out_features,
            bias=bias,
            weight_per_channel=True,
            activation_per_tensor=False,  # Per-token for value_states
        )
    
    @classmethod
    def from_linear(cls, linear: nn.Linear):
        """Create from existing ALinear."""
        reconstructor = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        reconstructor.int8_linear = Int8Linear.from_linear(linear, weight_per_channel=True)
        return reconstructor
    
    def forward(self, value_states: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full value from latent.
        
        Args:
            value_states: [batch, seq_len, latent_dim]
        
        Returns:
            [batch, seq_len, out_features]
        """
        return self.int8_linear(value_states)


# ============================================================================
# Utility Functions
# ============================================================================

def create_int8_value_reconstructor(alinear: nn.Linear) -> Int8ValueReconstructor:
    """Create INT8 value reconstructor from ALinear."""
    return Int8ValueReconstructor.from_linear(alinear)


def convert_alinear_to_int8(model: nn.Module, verbose: bool = True):
    """
    Convert all v_proj.ALinear layers to INT8.
    
    Args:
        model: The model to convert
        verbose: Print conversion info
    
    Returns:
        Number of layers converted
    """
    converted = 0
    
    for name, module in model.named_modules():
        if name.endswith('v_proj') and hasattr(module, 'ALinear'):
            if verbose:
                print(f"[INT8] Converting {name}.ALinear to INT8...")
            
            original_alinear = module.ALinear
            int8_reconstructor = Int8ValueReconstructor.from_linear(original_alinear)
            
            # Replace ALinear with INT8 reconstructor
            module.int8_reconstructor = int8_reconstructor
            module.use_int8_reconstruction = True
            
            converted += 1
    
    if verbose:
        print(f"[INT8] Converted {converted} ALinear layers to INT8")
    
    return converted


# ============================================================================
# Integration with Attention
# ============================================================================

class Int8ValueReconstructionMixin:
    """
    Mixin class to add INT8 value reconstruction to attention layers.
    
    Usage:
        class MyAttention(Int8ValueReconstructionMixin, LlamaAttention):
            pass
    """
    
    def setup_int8_reconstruction(self):
        """Setup INT8 reconstruction for value."""
        if hasattr(self, 'v_proj') and hasattr(self.v_proj, 'ALinear'):
            self.int8_value_reconstructor = Int8ValueReconstructor.from_linear(
                self.v_proj.ALinear
            )
            self.use_int8_reconstruction = True
            print(f"[INT8] Setup INT8 value reconstruction for layer {getattr(self, 'layer_idx', '?')}")
    
    def reconstruct_value_int8(self, value_states: torch.Tensor) -> torch.Tensor:
        """Reconstruct value using INT8 matmul."""
        if hasattr(self, 'int8_value_reconstructor') and self.use_int8_reconstruction:
            return self.int8_value_reconstructor(value_states)
        else:
            # Fallback to original
            return self.v_proj.ALinear(value_states)
