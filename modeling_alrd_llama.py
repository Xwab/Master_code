from transformers import LlamaForCausalLM
from .configuration_alrd_llama import ALRDLlamaConfig
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,  LlamaFlashAttention2, LlamaSdpaAttention, LlamaConfig, LlamaModel, \
    repeat_kv, apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import torch
from typing import Optional, Tuple, Union
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast
)
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import DynamicCache, StaticCache
from modules.quant_utils import (
    Quantizer, Quantizer2, quantize_tensor_forward,
    KIVIKeyQuantizer, KIVIValueQuantizer, KIVIMixedQuantizer,
    kivi_quantize_per_channel, kivi_quantize_per_token
)
from modules.hadamard_utils import apply_hadamard
from modules.kivi_cache import KIVILatentCache, create_kivi_cache
from modules.kivi_mixed_cache import (
    KIVIMixedPrecisionQuantizer,
    ALRDLinear_KIVI_Value_FullRank_Mixed,
    KIVIMixedPrecisionCache,
    create_mixed_precision_cache,
    calculate_mixed_precision_split,
)

# Union type for all KIVI-style caches
KIVI_CACHE_TYPES = (KIVILatentCache, KIVIMixedPrecisionCache)
logger = logging.get_logger(__name__)



class CustomLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        # KIVI-style quantizer for ALinear weight
        # Use per-channel quantization (each output channel has its own scale)
        self.a_weight_bits = getattr(config, 'a_weight_bits', 8)
        self.a_weight_group_size = getattr(config, 'a_weight_group_size', 128)
        self.quantizer = KIVIValueQuantizer(
            n_bits=self.a_weight_bits, 
            group_size=self.a_weight_group_size
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        cached_position_embeddings = None
        if "cached_position_embeddings" in kwargs:
            cached_position_embeddings = kwargs["cached_position_embeddings"]
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        
        # Check if using KIVI cache (handles quantization internally)
        # Supports both standard KIVILatentCache and KIVIMixedPrecisionCache
        use_kivi_cache = isinstance(past_key_value, KIVI_CACHE_TYPES)

        # Compute low-rank latent representations
        key_latent = self.k_proj.BLinear(hidden_states)  # [bsz, q_len, rank]
        value_latent = self.v_proj.BLinear(hidden_states)  # [bsz, q_len, rank]
        
        if use_kivi_cache:
            # KIVI cache handles quantization internally with per-channel/per-token scheme
            # and maintains residual (recent tokens in full precision)
            key_states, value_states = past_key_value.update(key_latent, value_latent, self.layer_idx)
        else:
            # Legacy path: use external quantization
            key_states = self.k_proj.quantize_latent_mixed(key_latent)
            value_states = self.v_proj.quantize_latent_mixed(value_latent)
            
            if past_key_value is not None:
                # Update with standard DynamicCache
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Restore full dimension from low-rank latent
        key_states = self.k_proj.ALinear(key_states)
        quantized_value_A_Linear = self.quantizer(self.v_proj.ALinear.weight.data)
        value_states = torch.matmul(value_states, quantized_value_A_Linear.t())

        #value_states_high = value_states[:, :, :high_bit_rank_num]
        #value_states_high = torch.matmul(value_states_high.to(torch.float32), quantized_value_A_Linear_high.t().to(torch.float32))
        #value_states_high = torch.matmul(value_states_high, quantized_value_A_Linear_high.t())
        #value_states_high = torch.clamp(value_states_high, min=self.q_min, max=self.q_max)
        #value_states_high = value_states_high * B_sacles_high.view(1, -1, 1)
        #value_states_high = (value_states_high * A_scales.view(1, 1, -1))#.to(torch.float16)
     
        #value_states_low = value_states[:, :, high_bit_rank_num:]
        #value_states_low = torch.matmul(value_states_low.to(torch.float32), quantized_value_A_Linear_low.t().to(torch.float32))
        #value_states_low = torch.matmul(value_states_low, quantized_value_A_Linear_low.t())
        #value_states_low = torch.clamp(value_states_low, min=self.q_min, max=self.q_max)
        #value_states_low = value_states_low * B_sacles_low.view(1, -1, 1)
        #value_states_low = (value_states_low * A_scales.view(1, 1, -1))#.to(torch.float16)
        
        #value_states = value_states_high + value_states_low

        #quantized_value_A_Linear = self.quantizer(self.v_proj.ALinear.weight.data)
        #value_states = torch.matmul(value_states, quantized_value_A_Linear.t())

        #value_states = self.v_proj.ALinear(value_states)
        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = cached_position_embeddings
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:,-1,:], sin[:,-1,:])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomLlamaFlashAttention2(LlamaFlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = True#not is_flash_attn_greater_or_equal_2_10()
        
        # KIVI-style quantizer for ALinear weight
        self.a_weight_bits = getattr(config, 'a_weight_bits', 8)
        self.a_weight_group_size = getattr(config, 'a_weight_group_size', 128)
        self.quantizer = KIVIValueQuantizer(
            n_bits=self.a_weight_bits, 
            group_size=self.a_weight_group_size
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        cached_position_embeddings = None
        if "cached_position_embeddings" in kwargs:
            cached_position_embeddings = kwargs["cached_position_embeddings"]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        
        # Check if using KIVI cache (handles quantization internally)
        # Supports both standard KIVILatentCache and KIVIMixedPrecisionCache
        use_kivi_cache = isinstance(past_key_value, KIVI_CACHE_TYPES)
        
        # Compute low-rank latent representations
        key_latent = self.k_proj.BLinear(hidden_states)  # [bsz, q_len, rank]
        value_latent = self.v_proj.BLinear(hidden_states)  # [bsz, q_len, rank]
        
        if use_kivi_cache:
            # KIVI cache handles quantization internally with per-channel/per-token scheme
            key_states, value_states = past_key_value.update(key_latent, value_latent, self.layer_idx)
        else:
            # Legacy path: use external quantization
            key_states = self.k_proj.quantize_latent(key_latent)
            value_states = self.v_proj.quantize_latent(value_latent)
            
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Restore full dimension from low-rank latent
        key_states = self.k_proj.ALinear(key_states)
        quantized_value_A_Linear = self.quantizer(self.v_proj.ALinear.weight.data)
        value_states = torch.matmul(value_states, quantized_value_A_Linear.t())

        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = cached_position_embeddings
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:,-1,:], sin[:,-1,:])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        

        

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换 Attention 为自定义的 CustomLlamaSdpaAttention
        #self.self_attn = CustomLlamaSdpaAttention(config, layer_idx)
        self.self_attn = CustomLlamaFlashAttention2(config, layer_idx)


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        # 替换解码层为自定义的 CustomLlamaDecoderLayer
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            print(f"rope_type:{self.rope_type}")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        cached_position = torch.arange(
            0, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)


        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cached_position_embeddings = self.rotary_emb(hidden_states, cached_position)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    cached_position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    cached_position_embeddings = cached_position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


'''class ALRDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):

        return self.ALinear(self.BLinear(input))'''

class ALRDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, input):
        return self.ALinear(self.BLinear(input))


class ALRDLinear_quant_key_v2(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        #self.BLinear = nn.Linear(in_features, rank, bias=False)
        #self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        ratio = rank / out_features
        if ratio < 0.5:
            #8 * out_features * ratio = 4 * out_features - 4 * x + 3 * x
            #4 * out_features * ratio = 3 *
            self.quantizer = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
            self.quantizer2 = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
            #self.quant_rank =  int(4 * out_features - 8 * out_features * ratio) #25%

        else:
            self.quantizer = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
            self.quantizer2 = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
            # self.quant_rank = int((8 * out_features - 8 * out_features * ratio) / 4) #25%

        #16 * out_features * ratio = 16 * out_features - 16 * x + 2 * x
        #14 * x = 16*out_features*(1-ratio)
        #8 * out_features * ratio = 8 * out_features - 8 * x + 2 * x
        #6 * x = 8*out_features*(1-ratio)
        # fullbit * x + (fullbit + 1) * (out_features - x) = 8 * out_features * ratio
        #
        #full_bit = int(8 * ratio)
        #x0 = 8 * out_features * (1-ratio) / (8 - full_bit)
        #self.quant_rank0 = x0
        #self.quant_rank0 = int(full_bit * out_features + out_features - 8 * out_features * ratio)
        #self.quant_rank1 = out_features - self.quant_rank0


        #x = 8 * out_features * (1-ratio) / 4
        #times = round(x // 32)
        #self.quant_rank = times * 32


        #x = 16 * out_features * (1-ratio) / 12 #x是待量化的维度 #2bit
        #times = round(x // 16)
        self.quant_rank = rank
        #print('quant rank', self.quant_rank)
        #self.quantizer_8 = Quantizer(n_bits = 8, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_4 = Quantizer(n_bits = 4, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_3 = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_2 = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
        self.rank = rank
        self.rank_lists = self.split_rank_for_hada(self.quant_rank)

    def split_rank_for_hada(self, rank):

        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        #hada_list = [244, 180, 172, 156, 140, 108, 92, 84, 76, 68, 60, 52, 44, 40, 36, 28, 20, 12, 1]
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if (is_pow2(times)):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list

    def quantize_latent(self, latents):
        fake_quant_latent = self.quantizer(latents)
        return fake_quant_latent
    
    def quantize_latent_mixed(self, latents):
        #latents_high_bit = latents[:,:,:-self.quant_rank]
        #latents_low_bit = latents[:,:,-self.quant_rank:]
        #latents_high_bit, scales_high_bit = self.quantizer3(latents_high_bit)
        #latents_low_bit, scales_low_bit = self.quantizer(latents_low_bit)
        #latents_high_bit = self.quantizer(latents_high_bit)
        #latents_low_bit = self.quantizer2(latents_low_bit)
        #return torch.cat([latents_high_bit, latents_low_bit], dim = -1)
        #return latents_high_bit, latents_low_bit, scales_high_bit, scales_low_bit
        #return torch.cat([latents_high_bit, latents_low_bit], dim = -1), self.quant_rank, scales_high_bit, scales_low_bit
        fake_quant_latent = self.quantizer(latents)
        return fake_quant_latent

    def fuse_hadamard(self):
        def hadamard_transform(x):
            """
            计算一维向量的哈达玛变换。
            参数:
                x (Tensor): 一维张量，其长度必须为 2^n。
            返回:
                Tensor: 哈达玛变换后的结果。
            """
            n = x.size(1)
            # 检查输入长度是否为2的幂次
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")

            # 构建哈达玛矩阵
            H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(x.device)
            # 通过Kronecker积递归构建高阶哈达玛矩阵
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(H.device))

            # 执行变换：H * x
            return torch.matmul(x, H) / torch.tensor(n).sqrt()
        #VT_weight = self.BLinear.weight.data
        #U_weight = self.ALinear.weight.data
        #结合量化
        full_rank = self.ALinear.weight.data.shape[0]
        start = 0
        VT_weight = self.BLinear.weight.data#[start:, :]
        U_weight = self.ALinear.weight.data#[:, start:]
        total_rank = 0
        print(VT_weight.shape[0], sum(self.rank_lists), self.rank_lists)
        for rank in self.rank_lists:
            VT_chunk = VT_weight[start + total_rank:start + total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()          # shape: [in_dim, rank]
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            #VT_chunk = apply_hadamard(VT_chunk)
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[start + total_rank:start + total_rank + rank, :] = VT_chunk.t()

            U_chunk = U_weight[:, start + total_rank:start + total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            #U_chunk = apply_hadamard(U_chunk)
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank]
            )

            total_rank += rank

    def forward(self, input):

       #y = self.BLinear(input)[:, :self.rank]
       #weight_A_partial = self.ALinear.weight.data[:, :self.rank]
       #y = torch.matmul(y, weight_A_partial.t())
       #return y
       y = self.BLinear(input)
       
       y_quant = self.quantizer(y)
       return self.ALinear(y_quant)
       #y = self.quantize_latent(y)
       #return self.ALinear(y)

       #return self.ALinear(self.BLinear(input))

class ALRDLinear_quant_key(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        #self.BLinear = nn.Linear(in_features, rank, bias=False)
        #self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.BLinear = nn.Linear(in_features, out_features, bias=False)
        self.ALinear = nn.Linear(out_features, out_features, bias=bias)
        ratio = rank / out_features
        if ratio < 0.5:
            #8 * out_features * ratio = 4 * out_features - 4 * x + 3 * x
            #4 * out_features * ratio = 3 *
            self.quantizer = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
            self.quantizer2 = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
            #self.quant_rank =  int(4 * out_features - 8 * out_features * ratio) #25%

        else:
            self.quantizer = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
            self.quantizer2 = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
            # self.quant_rank = int((8 * out_features - 8 * out_features * ratio) / 4) #25%

        #16 * out_features * ratio = 16 * out_features - 16 * x + 2 * x
        #14 * x = 16*out_features*(1-ratio)
        #8 * out_features * ratio = 8 * out_features - 8 * x + 2 * x
        #6 * x = 8*out_features*(1-ratio)
        # fullbit * x + (fullbit + 1) * (out_features - x) = 8 * out_features * ratio
        #
        #full_bit = int(8 * ratio)
        #x0 = 8 * out_features * (1-ratio) / (8 - full_bit)
        #self.quant_rank0 = x0
        #self.quant_rank0 = int(full_bit * out_features + out_features - 8 * out_features * ratio)
        #self.quant_rank1 = out_features - self.quant_rank0


        #x = 8 * out_features * (1-ratio) / 4
        #times = round(x // 32)
        #self.quant_rank = times * 32


        #x = 16 * out_features * (1-ratio) / 12 #x是待量化的维度 #2bit
        #times = round(x // 16)
        self.quant_rank = rank
        #print('quant rank', self.quant_rank)
        #self.quantizer_8 = Quantizer(n_bits = 8, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_4 = Quantizer(n_bits = 4, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_3 = Quantizer(n_bits = 3, group_size = 0, sym = True, clip_ratio = 1.0)
        #self.quantizer_2 = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
        self.rank = rank
        self.rank_lists = self.split_rank_for_hada(self.quant_rank)

    def split_rank_for_hada(self, rank):

        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        #hada_list = [244, 180, 172, 156, 140, 108, 92, 84, 76, 68, 60, 52, 44, 40, 36, 28, 20, 12, 1]
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if (is_pow2(times)):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list

    def quantize_latent(self, latents):
        fake_quant_latent = self.quantizer(latents)
        return fake_quant_latent
    
    def quantize_latent_mixed(self, latents):
        latents_high_bit = latents[:,:,:-self.quant_rank]
        latents_low_bit = latents[:,:,-self.quant_rank:]
        #latents_high_bit, scales_high_bit = self.quantizer3(latents_high_bit)
        #latents_low_bit, scales_low_bit = self.quantizer(latents_low_bit)
        latents_high_bit = self.quantizer(latents_high_bit)
        latents_low_bit = self.quantizer2(latents_low_bit)
        return torch.cat([latents_high_bit, latents_low_bit], dim = -1)
        #return latents_high_bit, latents_low_bit, scales_high_bit, scales_low_bit
        #return torch.cat([latents_high_bit, latents_low_bit], dim = -1), self.quant_rank, scales_high_bit, scales_low_bit

    def fuse_hadamard(self):
        def hadamard_transform(x):
            """
            计算一维向量的哈达玛变换。
            参数:
                x (Tensor): 一维张量，其长度必须为 2^n。
            返回:
                Tensor: 哈达玛变换后的结果。
            """
            n = x.size(1)
            # 检查输入长度是否为2的幂次
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")

            # 构建哈达玛矩阵
            H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(x.device)
            # 通过Kronecker积递归构建高阶哈达玛矩阵
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(H.device))

            # 执行变换：H * x
            return torch.matmul(x, H) / torch.tensor(n).sqrt()
        #VT_weight = self.BLinear.weight.data
        #U_weight = self.ALinear.weight.data
        #结合量化
        full_rank = self.ALinear.weight.data.shape[0]
        start = 0
        VT_weight = self.BLinear.weight.data#[start:, :]
        U_weight = self.ALinear.weight.data#[:, start:]
        total_rank = 0
        print(VT_weight.shape[0], sum(self.rank_lists), self.rank_lists)
        for rank in self.rank_lists:
            VT_chunk = VT_weight[start + total_rank:start + total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()          # shape: [in_dim, rank]
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            #VT_chunk = apply_hadamard(VT_chunk)
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[start + total_rank:start + total_rank + rank, :] = VT_chunk.t()

            U_chunk = U_weight[:, start + total_rank:start + total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            #U_chunk = apply_hadamard(U_chunk)
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank]
            )

            total_rank += rank

    def forward(self, input):

       #y = self.BLinear(input)[:, :self.rank]
       #weight_A_partial = self.ALinear.weight.data[:, :self.rank]
       #y = torch.matmul(y, weight_A_partial.t())
       #return y
       y = self.BLinear(input)
       
       y_quant = self.quantizer(y)
       return self.ALinear(y_quant)
       #y = self.quantize_latent(y)
       #return self.ALinear(y)

       #return self.ALinear(self.BLinear(input))


class ALRDLinear_quant(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        #self.BLinear = nn.Linear(in_features, rank, bias=False)
        #self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.BLinear = nn.Linear(in_features, out_features, bias=False)
        self.ALinear = nn.Linear(out_features, out_features, bias=bias)
        ratio = rank / out_features
        #16 * out_features * ratio = 16 * out_features - 16 * x + 2 * x
        #14 * x = 16*out_features*(1-ratio)
        #8 * out_features * ratio = 8 * out_features - 8 * x + 2 * x
        #6 * x = 8*out_features*(1-ratio)
        # fullbit * x + (fullbit + 1) * (out_features - x) = 8 * out_features * ratio
        #
        full_bit = int(8 * ratio)
        #x0 = 8 * out_features * (1-ratio) / (8 - full_bit)
        #self.quant_rank0 = x0
        self.quant_rank0 = int(full_bit * out_features + out_features - 8 * out_features * ratio)
        self.quant_rank1 = out_features - self.quant_rank0
  
        #4 * out * ratio = 4 * out - 4 * x + 2 * x
        x = 4 * out_features * (1-ratio) / 2


        #x = 8 * out_features * (1-ratio) / 4
        #times = round(x // 32)
        #self.quant_rank = times * 32
        self.quant_rank = int(x)


        #x = 16 * out_features * (1-ratio) / 12 #x是待量化的维度 #2bit
        #times = round(x // 16)
        #self.quant_rank = times * 16
        print('value quant rank', self.quant_rank)
        self.quantizer = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
        self.quantizer2 = Quantizer(n_bits=full_bit, group_size= 0, sym = True, clip_ratio = 1.0)
        self.quantizer2_2 = Quantizer(n_bits=full_bit+1, group_size= 0, sym = True, clip_ratio = 1.0)
        self.quantizer3 = Quantizer(n_bits=4, group_size= 0, sym = True, clip_ratio = 1.0)
        self.rank = rank
        self.rank_lists = self.split_rank_for_hada(out_features)

    def split_rank_for_hada(self, rank):

        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        #hada_list = [244, 180, 172, 156, 140, 108, 92, 84, 76, 68, 60, 52, 44, 40, 36, 28, 20, 12, 1]
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if (is_pow2(times)):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list

    def quantize_latent(self, latents):
        fake_quant_latent = self.quantizer2(latents)
        return fake_quant_latent
    
    def quantize_latent_full(self, latents):
        fake_quant_latent = self.quantizer3(latents)
        return fake_quant_latent
    
    def quantize_latent_mixed_2(self, latents):
        latents_full_bit = latents[:,:,:self.quant_rank1]
        latents_quant = latents[:,:,self.quant_rank1:]
        return torch.cat([self.quantizer2_2(latents_full_bit), self.quantizer2(latents_quant)], dim = -1)
    
    def quantize_latent_mixed(self, latents):
        latents_high_bit = latents[:,:,:-self.quant_rank]
        latents_low_bit = latents[:,:,-self.quant_rank:]
        #latents_high_bit, scales_high_bit = self.quantizer3(latents_high_bit)
        #latents_low_bit, scales_low_bit = self.quantizer(latents_low_bit)
        latents_high_bit = self.quantizer3(latents_high_bit)
        latents_low_bit = self.quantizer(latents_low_bit)
        return torch.cat([latents_high_bit, latents_low_bit], dim = -1)
        #return latents_high_bit, latents_low_bit, scales_high_bit, scales_low_bit
        #return torch.cat([latents_high_bit, latents_low_bit], dim = -1), self.quant_rank, scales_high_bit, scales_low_bit
    def fuse_hadamard(self):
        def hadamard_transform(x):
            """
            计算一维向量的哈达玛变换。
            参数:
                x (Tensor): 一维张量，其长度必须为 2^n。
            返回:
                Tensor: 哈达玛变换后的结果。
            """
            n = x.size(1)
            # 检查输入长度是否为2的幂次
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")

            # 构建哈达玛矩阵
            H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(x.device)
            # 通过Kronecker积递归构建高阶哈达玛矩阵
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=torch.float16).to(H.device))

            # 执行变换：H * x
            return torch.matmul(x, H) / torch.tensor(n).sqrt()
        #VT_weight = self.BLinear.weight.data
        #U_weight = self.ALinear.weight.data
        #结合量化
        full_rank = self.ALinear.weight.data.shape[0]
        #start = full_rank - self.quant_rank
        start=0
        VT_weight = self.BLinear.weight.data#[start:, :]
        U_weight = self.ALinear.weight.data#[:, start:]
        total_rank = 0
        print(VT_weight.shape[0], sum(self.rank_lists), self.rank_lists)
        for rank in self.rank_lists:
            VT_chunk = VT_weight[start + total_rank:start + total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()          # shape: [in_dim, rank]
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            #VT_chunk = apply_hadamard(VT_chunk)
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[start + total_rank:start + total_rank + rank, :] = VT_chunk.t()

            U_chunk = U_weight[:, start + total_rank:start + total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            #U_chunk = apply_hadamard(U_chunk)
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, start + total_rank:start + total_rank + rank]
            )

            total_rank += rank

    def forward(self, input):

       #y = self.BLinear(input)[:, :self.rank]
       #weight_A_partial = self.ALinear.weight.data[:, :self.rank]
       #y = torch.matmul(y, weight_A_partial.t())
       #return y
       y = self.BLinear(input)
       y_full_bit = y[:,:,:self.quant_rank]
       y_quant = y[:,:,self.quant_rank:]
       y_quant = self.quantizer(y_quant)
       weight_A_full_bit = self.ALinear.weight.data[:, :self.quant_rank]
       weight_A_quant = self.ALinear.weight.data[:, :self.quant_rank]
       return torch.matmul(y_full_bit, weight_A_full_bit.t()) + torch.matmul(y_quant, weight_A_quant.t())
       #y = self.quantize_latent(y)
       #return self.ALinear(y)

       #return self.ALinear(self.BLinear(input))


# ============================================================================
# KIVI-style Low-Rank Linear Layers
# ============================================================================

class ALRDLinear_KIVI_Key(nn.Module):
    """
    Low-rank linear layer with KIVI-style per-channel quantization for Key.
    
    Key cache is quantized per-channel (along the token dimension), which 
    preserves the variance structure that is important for attention computation.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        bias: bool = True,
        k_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
    ):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # KIVI-style quantizer: per-channel for Key
        self.kivi_quantizer = KIVIMixedQuantizer(
            n_bits=k_bits,
            group_size=group_size,
            residual_length=residual_length,
            per_channel=True,  # Key uses per-channel quantization
        )
        
        # KIVI-style quantizer (per-channel for Key)
        self.quantizer = KIVIKeyQuantizer(n_bits=k_bits, group_size=group_size)
        
        # For Hadamard transform
        self.rank_lists = self._split_rank_for_hada(rank)
    
    def _split_rank_for_hada(self, rank):
        """Split rank into chunks suitable for Hadamard transform (power of 2)."""
        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if is_pow2(times):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list
    
    def fuse_hadamard(self):
        """Apply Hadamard transform to BLinear and ALinear weights for better quantization."""
        def hadamard_transform(x):
            n = x.size(1)
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")
            H = torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(x.device)
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(H.device))
            return torch.matmul(x, H) / torch.tensor(n, dtype=x.dtype).sqrt()
        
        VT_weight = self.BLinear.weight.data
        U_weight = self.ALinear.weight.data
        total_rank = 0
        
        print(f"ALRDLinear_KIVI_Key fuse_hadamard: rank={self.rank}, rank_lists={self.rank_lists}")
        
        for rank in self.rank_lists:
            # Transform BLinear (VT)
            VT_chunk = VT_weight[total_rank:total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[total_rank:total_rank + rank, :] = VT_chunk.t()
            
            # Transform ALinear (U)
            U_chunk = U_weight[:, total_rank:total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, total_rank:total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, total_rank:total_rank + rank]
            )
            
            total_rank += rank
    
    def quantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Simple quantization without residual."""
        x_quant, _, _ = kivi_quantize_per_channel(
            latents, 
            self.kivi_quantizer.n_bits, 
            self.kivi_quantizer.group_size
        )
        return x_quant
    
    def quantize_latent_mixed(self, latents: torch.Tensor) -> torch.Tensor:
        """KIVI-style quantization with residual (recent tokens in full precision)."""
        return self.kivi_quantizer(latents)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.BLinear(input)
        y = self.quantize_latent(y)
        return self.ALinear(y)


class ALRDLinear_KIVI_Value(nn.Module):
    """
    Low-rank linear layer with KIVI-style per-token quantization for Value.
    
    Value cache is quantized per-token (along the hidden dimension), which
    works better for the weighted sum operation in attention.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        bias: bool = True,
        v_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
    ):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # KIVI-style quantizer: per-token for Value
        self.kivi_quantizer = KIVIMixedQuantizer(
            n_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            per_channel=False,  # Value uses per-token quantization
        )
        
        # KIVI-style quantizer (per-token for Value)
        self.quantizer = KIVIValueQuantizer(n_bits=v_bits, group_size=group_size)
        
        # For Hadamard transform
        self.rank_lists = self._split_rank_for_hada(rank)
    
    def _split_rank_for_hada(self, rank):
        """Split rank into chunks suitable for Hadamard transform (power of 2)."""
        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if is_pow2(times):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list
    
    def fuse_hadamard(self):
        """Apply Hadamard transform to BLinear and ALinear weights for better quantization."""
        def hadamard_transform(x):
            n = x.size(1)
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")
            H = torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(x.device)
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(H.device))
            return torch.matmul(x, H) / torch.tensor(n, dtype=x.dtype).sqrt()
        
        VT_weight = self.BLinear.weight.data
        U_weight = self.ALinear.weight.data
        total_rank = 0
        
        print(f"ALRDLinear_KIVI_Value fuse_hadamard: rank={self.rank}, rank_lists={self.rank_lists}")
        
        for rank in self.rank_lists:
            # Transform BLinear (VT)
            VT_chunk = VT_weight[total_rank:total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[total_rank:total_rank + rank, :] = VT_chunk.t()
            
            # Transform ALinear (U)
            U_chunk = U_weight[:, total_rank:total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, total_rank:total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, total_rank:total_rank + rank]
            )
            
            total_rank += rank
    
    def quantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Simple quantization without residual."""
        x_quant, _, _ = kivi_quantize_per_token(
            latents, 
            self.kivi_quantizer.n_bits, 
            self.kivi_quantizer.group_size
        )
        return x_quant
    
    def quantize_latent_mixed(self, latents: torch.Tensor) -> torch.Tensor:
        """KIVI-style quantization with residual (recent tokens in full precision)."""
        return self.kivi_quantizer(latents)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.BLinear(input)
        y = self.quantize_latent(y)
        return self.ALinear(y)


class ALRDLinear_KIVI_Value_Mixed(nn.Module):
    """
    Low-rank linear layer with mixed-precision KIVI quantization for Value.
    
    This class implements mixed 4bit/2bit quantization based on singular value importance:
    - Features with larger singular values (more important) → 4bit
    - Features with smaller singular values (less important) → 2bit
    
    The split is determined by the target compression ratio:
    - target_ratio: user-specified target compression ratio (r1)
    - lowrank_ratio: rank / out_features (r2) 
    - final_ratio: r1 * r2 (should be >= 0.125 for 2bit minimum)
    
    Compression calculation:
    - n_4bit = out_features * (8 * final_ratio - 1)
    - n_2bit = out_features - n_4bit
    - Average bits = (4 * n_4bit + 2 * n_2bit) / out_features
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        bias: bool = True,
        target_ratio: float = 0.25,  # Target compression ratio (r1)
        group_size: int = 128,
        residual_length: int = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.group_size = group_size
        self.residual_length = residual_length
        
        # BLinear outputs full out_features (not rank)
        # This allows mixed-precision quantization on full feature space
        self.BLinear = nn.Linear(in_features, out_features, bias=False)
        self.ALinear = nn.Linear(out_features, out_features, bias=bias)
        
        # For Hadamard transform - use out_features since that's the latent dimension
        self.rank_lists = self._split_rank_for_hada(out_features)
        
        # Calculate compression ratios
        self.lowrank_ratio = rank / out_features  # r2
        self.target_ratio = target_ratio  # r1
        self.final_ratio = self.target_ratio * self.lowrank_ratio  # r = r1 * r2
        
        # Calculate 4bit/2bit split based on final compression ratio
        # Compression: (4 * n_4bit + 2 * n_2bit) / (16 * out_features) = final_ratio
        # Solving: n_4bit = out_features * (8 * final_ratio - 1)
        #          n_2bit = out_features * (2 - 8 * final_ratio)
        
        if self.final_ratio <= 0.125:
            # All 2bit (maximum compression)
            self.n_4bit = 0
            self.n_2bit = out_features
        elif self.final_ratio >= 0.25:
            # All 4bit (minimum compression for mixed mode)
            self.n_4bit = out_features
            self.n_2bit = 0
        else:
            # Mixed precision
            self.n_4bit = int(out_features * (8 * self.final_ratio - 1))
            self.n_2bit = out_features - self.n_4bit
        
        # Ensure alignment to group_size for efficient quantization
        if group_size > 0:
            self.n_4bit = (self.n_4bit // group_size) * group_size
            self.n_2bit = out_features - self.n_4bit
        
        # Create KIVI-style quantizers for each precision level (per-token for Value)
        self.quantizer_4bit = KIVIValueQuantizer(n_bits=4, group_size=group_size)
        self.quantizer_2bit = KIVIValueQuantizer(n_bits=2, group_size=group_size)
        
        # Store actual average bits for logging
        self.avg_bits = (4 * self.n_4bit + 2 * self.n_2bit) / out_features if out_features > 0 else 0
        
        print(f"ALRDLinear_KIVI_Value_Mixed: out_features={out_features}, "
              f"n_4bit={self.n_4bit}, n_2bit={self.n_2bit}, "
              f"avg_bits={self.avg_bits:.2f}, final_ratio={self.final_ratio:.4f}")
    
    def _split_rank_for_hada(self, rank):
        """Split rank into chunks suitable for Hadamard transform (power of 2)."""
        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if is_pow2(times):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list
    
    def fuse_hadamard(self):
        """Apply Hadamard transform to BLinear and ALinear weights for better quantization."""
        def hadamard_transform(x):
            n = x.size(1)
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")
            H = torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(x.device)
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(H.device))
            return torch.matmul(x, H) / torch.tensor(n, dtype=x.dtype).sqrt()
        
        VT_weight = self.BLinear.weight.data
        U_weight = self.ALinear.weight.data
        total_rank = 0
        
        print(f"ALRDLinear_KIVI_Value_Mixed fuse_hadamard: out_features={self.out_features}, rank_lists={self.rank_lists}")
        
        for rank in self.rank_lists:
            # Transform BLinear (VT)
            VT_chunk = VT_weight[total_rank:total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[total_rank:total_rank + rank, :] = VT_chunk.t()
            
            # Transform ALinear (U)
            U_chunk = U_weight[:, total_rank:total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, total_rank:total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, total_rank:total_rank + rank]
            )
            
            total_rank += rank
    
    def quantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Mixed-precision quantization: 4bit for important features, 2bit for others.
        
        Features are assumed to be ordered by singular value importance
        (largest singular values first), so we apply 4bit to the first n_4bit
        features and 2bit to the remaining.
        
        Args:
            latents: [batch, seq_len, out_features]
        
        Returns:
            Quantized latents with same shape
        """
        if self.n_4bit == self.out_features:
            # All 4bit
            return self.quantizer_4bit(latents)
        elif self.n_4bit == 0:
            # All 2bit
            return self.quantizer_2bit(latents)
        else:
            # Mixed precision: split, quantize separately, concatenate
            latents_4bit = latents[..., :self.n_4bit]
            latents_2bit = latents[..., self.n_4bit:]
            
            quant_4bit = self.quantizer_4bit(latents_4bit)
            quant_2bit = self.quantizer_2bit(latents_2bit)
            
            return torch.cat([quant_4bit, quant_2bit], dim=-1)
    
    def quantize_latent_mixed(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Mixed-precision quantization with residual support.
        Keep recent tokens in full precision.
        """
        seq_len = latents.shape[1]
        
        if seq_len <= self.residual_length:
            # All tokens fit in residual, no quantization
            return latents
        
        # Split into quantized and residual parts
        n_quant = seq_len - self.residual_length
        if self.group_size > 0:
            n_quant = (n_quant // self.group_size) * self.group_size
        
        if n_quant <= 0:
            return latents
        
        latents_to_quant = latents[:, :n_quant, :]
        latents_residual = latents[:, n_quant:, :]
        
        # Apply mixed-precision quantization
        latents_quantized = self.quantize_latent(latents_to_quant)
        
        return torch.cat([latents_quantized, latents_residual], dim=1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.BLinear(input)
        y = self.quantize_latent(y)
        return self.ALinear(y)


class ALRDLlamaForCausalLM(LlamaForCausalLM):
    
    config_class = ALRDLlamaConfig

    def __init__(self, config: ALRDLlamaConfig):
        super().__init__(config)
        self.truncation_ranks = config.truncation_ranks
        self.model = CustomLlamaModel(config)
        
        # KIVI parameters
        self.use_kivi = getattr(config, 'use_kivi', False)
        self.k_bits = getattr(config, 'k_bits', 2)
        self.v_bits = getattr(config, 'v_bits', 2)
        self.group_size = getattr(config, 'group_size', 128)
        self.residual_length = getattr(config, 'residual_length', 32)
        
        # Mixed-precision Value parameters
        self.use_mixed_precision_value = getattr(config, 'use_mixed_precision_value', False)
        self.value_target_ratios = getattr(config, 'value_target_ratios', {})
        self.default_value_target_ratio = getattr(config, 'default_value_target_ratio', 0.25)
        
        # Full-rank mixed-precision Value parameters
        # When True: keep full rank (D), use 4bit+2bit mixed to match low-rank+3bit compression
        # This replaces low-rank truncation with full-rank mixed-precision quantization
        self.use_fullrank_mixed_value = getattr(config, 'use_fullrank_mixed_value', False)

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)

        for name, module in self.named_modules():
            if name in self.truncation_ranks:
                info = linear_info[module]
                rank = self.truncation_ranks[name]
                
                if self.use_kivi:
                    # Use KIVI-style quantization
                    if name.endswith('k_proj'):
                        new_layer = ALRDLinear_KIVI_Key(
                            module.in_features, 
                            module.out_features, 
                            rank,
                            bias=module.bias is not None,
                            k_bits=self.k_bits,
                            group_size=self.group_size,
                            residual_length=self.residual_length,
                        )
                    elif name.endswith('v_proj'):
                        if self.use_fullrank_mixed_value:
                            # Full-rank + 4bit/2bit mixed to match low-rank+3bit compression
                            # This is the user's new scheme:
                            # - Keep all D features (no truncation)
                            # - Use 4bit for important features, 2bit for others
                            # - Match compression: 3r = 4*n_4bit + 2*n_2bit
                            new_layer = ALRDLinear_KIVI_Value_FullRank_Mixed(
                                in_features=module.in_features, 
                                out_features=module.out_features, 
                                original_rank=rank,  # The rank that would have been used
                                bias=module.bias is not None,
                                group_size=self.group_size,
                                residual_length=self.residual_length,
                            )
                        elif self.use_mixed_precision_value:
                            # Use mixed 4bit/2bit quantization for Value
                            target_ratio = self.value_target_ratios.get(
                                name, self.default_value_target_ratio
                            )
                            new_layer = ALRDLinear_KIVI_Value_Mixed(
                                module.in_features, 
                                module.out_features, 
                                rank,
                                bias=module.bias is not None,
                                target_ratio=target_ratio,
                                group_size=self.group_size,
                                residual_length=self.residual_length,
                            )
                        else:
                            # Use uniform quantization for Value
                            new_layer = ALRDLinear_KIVI_Value(
                                module.in_features, 
                                module.out_features, 
                                rank,
                                bias=module.bias is not None,
                                v_bits=self.v_bits,
                                group_size=self.group_size,
                                residual_length=self.residual_length,
                            )
                    else:
                        continue
                else:
                    # Use legacy quantization
                    if name.endswith('k_proj'):
                        new_layer = ALRDLinear_quant_key_v2(
                            module.in_features, 
                            module.out_features, 
                            rank,
                            bias=module.bias is not None
                        )
                    elif name.endswith('v_proj'):
                        new_layer = ALRDLinear_quant_key_v2(
                            module.in_features, 
                            module.out_features, 
                            rank,
                            bias=module.bias is not None
                        )
                    else:
                        continue
                
                setattr(info["father"], info["name"], new_layer)
    
    def create_kivi_cache(self) -> KIVILatentCache:
        """
        Create a KIVI cache for this model.
        
        Usage:
            model = ALRDLlamaForCausalLM.from_pretrained(...)
            kivi_cache = model.create_kivi_cache()
            
            # Use in generation
            outputs = model.generate(
                input_ids,
                past_key_values=kivi_cache,
                use_cache=True,
            )
        
        Returns:
            KIVILatentCache configured with model's KIVI parameters
        """
        return create_kivi_cache(
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.group_size,
            residual_length=self.residual_length,
        )
    
    def create_mixed_precision_cache(
        self,
        out_features: int = None,
    ) -> KIVIMixedPrecisionCache:
        """
        Create a mixed-precision KIVI cache for full-rank Value quantization.
        
        This cache uses:
        - Standard KIVI per-channel quantization for Key
        - Mixed 4bit/2bit quantization for Value based on singular value importance
        - Per-layer different ranks from truncation_ranks
        
        Args:
            out_features: Output dimension (D). If None, uses head_dim * num_kv_heads
        
        Returns:
            KIVIMixedPrecisionCache configured for this model with per-layer ranks
        """
        # Get out_features from model config
        if out_features is None:
            out_features = self.config.hidden_size // self.config.num_attention_heads * self.config.num_key_value_heads
        
        # Extract per-layer v_proj ranks from truncation_ranks
        # truncation_ranks format: {"model.layers.0.self_attn.v_proj": 256, ...}
        layer_original_ranks = {}
        default_original_rank = out_features // 4  # Default to 25%
        
        for name, rank in self.truncation_ranks.items():
            if 'v_proj' in name:
                # Extract layer index from name like "model.layers.0.self_attn.v_proj"
                try:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            layer_original_ranks[layer_idx] = rank
                            break
                except (ValueError, IndexError):
                    pass
        
        if layer_original_ranks:
            # Use the first layer's rank as default if available
            default_original_rank = list(layer_original_ranks.values())[0]
            print(f"[KIVI MixedPrecision] Created cache with {len(layer_original_ranks)} layer-specific ranks")
            print(f"[KIVI MixedPrecision] Sample ranks: {dict(list(layer_original_ranks.items())[:3])}...")
        else:
            print(f"[KIVI MixedPrecision] No v_proj ranks found, using default_original_rank={default_original_rank}")
        
        # Get mixed precision options from config
        match_compression = getattr(self.config, 'mixed_match_compression', True)
        high_precision_ratio = getattr(self.config, 'mixed_high_precision_ratio', 0.25)
        high_bits = getattr(self.config, 'mixed_high_bits', 4)
        low_bits = getattr(self.config, 'mixed_low_bits', 2)
        
        return create_mixed_precision_cache(
            k_bits=self.k_bits,
            out_features=out_features,
            layer_original_ranks=layer_original_ranks,
            default_original_rank=default_original_rank,
            group_size=self.group_size,
            residual_length=self.residual_length,
            match_compression=match_compression,
            high_precision_ratio=high_precision_ratio,
            high_bits=high_bits,
            low_bits=low_bits,
        )
    
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        """
        Override generate to automatically use KIVI cache when use_kivi is enabled.
        
        This makes the model compatible with lm_eval and other evaluation frameworks
        that call model.generate() without explicitly passing a KIVI cache.
        
        To use KIVI cache, the past_key_values must be an instance of KIVILatentCache
        or KIVIMixedPrecisionCache (for full-rank mixed precision mode).
        This is checked in attention via: use_kivi_cache = isinstance(past_key_value, KIVI_CACHE_TYPES)
        """
        past_key_values = kwargs.get('past_key_values')
        
        # Check if we should use KIVI cache
        # NOTE: Both KIVILatentCache and KIVIMixedPrecisionCache use fake quantization and do NOT save memory!
        # Only enable if you explicitly want to test KIVI cache behavior.
        # For memory-efficient inference, use official KIVI with Triton kernels.
        use_kivi_cache = getattr(self, 'use_kivi_cache', False)  # Default: disabled
        
        if use_kivi_cache and self.use_kivi:
            if kwargs.get('past_key_values') is None or not isinstance(kwargs.get('past_key_values'), KIVI_CACHE_TYPES):
                # Use mixed precision cache if full-rank mixed mode is enabled
                if self.use_fullrank_mixed_value:
                    kwargs['past_key_values'] = self.create_mixed_precision_cache()
                    print(f"[KIVI] Auto-created KIVIMixedPrecisionCache for generate()")
                else:
                    kwargs['past_key_values'] = self.create_kivi_cache()
                    print(f"[KIVI] Auto-created KIVILatentCache for generate()")
            
            if 'use_cache' not in kwargs:
                kwargs['use_cache'] = True
        
        # Call parent's generate
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Override forward to automatically use KIVI cache when use_kivi is enabled.
        
        To use KIVI cache, the past_key_values must be an instance of KIVILatentCache
        or KIVIMixedPrecisionCache (for full-rank mixed precision mode).
        This is checked in attention via: use_kivi_cache = isinstance(past_key_value, KIVI_CACHE_TYPES)
        """
        # Determine if we should use cache
        use_cache_flag = use_cache if use_cache is not None else self.config.use_cache
        
        # Check if we should use KIVI cache
        # NOTE: Disabled by default because KIVI caches use fake quantization
        # and does NOT save memory. Enable with model.use_kivi_cache = True
        use_kivi_cache = getattr(self, 'use_kivi_cache', False)
        
        if use_kivi_cache and self.use_kivi and use_cache_flag:
            if past_key_values is None or not isinstance(past_key_values, KIVI_CACHE_TYPES):
                # Use mixed precision cache if full-rank mixed mode is enabled
                if self.use_fullrank_mixed_value:
                    past_key_values = self.create_mixed_precision_cache()
                else:
                    past_key_values = self.create_kivi_cache()
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
