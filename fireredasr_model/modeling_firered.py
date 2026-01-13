# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch FireRed model."""

import math
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import auto_docstring, logging
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from .configuration_firered import FireredAudioConfig, FireredAudioEncoderConfig


logger = logging.get_logger(__name__)


class GlobalCMVN(torch.nn.Module):

    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x

@dataclass
class FireRedCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
        two sets of pre-computed hidden-states: key and values states in the self-attention blocks.
        The `past_key_values` are returned when `use_cache=True` is passed or when `config.use_cache=True`.
        It is a [`~cache_utils.Cache`] instance.

        If `past_key_values` are used, the user can optionally input only the last `input_ids` (those
        that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
        all `input_ids` of shape `(batch_size, sequence_length)`.
    attention_mask (`torch.FloatTensor`, *optional*):
        Attentions mask, used to update attention mask and position_ids.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None


# Copied from transformers.models.whisper.modeling_whisper.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head,
                 residual_dropout=0.1,
                 dropout_rate=0.1, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model,
                                             residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size,
                                         dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, input_size, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((input_size - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        self.subsampling = 4
        left_context = right_context = 3  # both exclude currect frame
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model*4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model*4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(pre_layer_norm,
                                 linear_expand,
                                 nonlinear,
                                 dropout_pre,
                                 linear_project,
                                 dropout_post)

    def forward(self, x):
        residual = x
        output = self.net(x)
        output = output + residual
        return output


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33, dropout_rate=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=1, bias=False)
        self.glu = F.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model*2, d_model*2,
                                        kernel_size, stride=1,
                                        padding=self.padding,
                                        groups=d_model*2, bias=False)
        self.batch_norm = nn.LayerNorm(d_model*2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model*2, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        output, attn = self.attention(q, k, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn

    def forward_qkv(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual, sz_b, len_q):
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float('inf')

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        output, attn = self.forward_attention(attn, v, mask)
        return output, attn

    def forward_attention(self, attn, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = self.dropout(attn)
        output = torch.matmul(d_attn, v)

        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__(n_head, d_model,
                         residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k ** 0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, q, k, v, pos_emb, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        
        q_with_bias_u = q_with_bias_u.to(k.dtype)
        q_with_bias_v = q_with_bias_v.to(p.dtype)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.attention.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn

class FireRedPreTrainedModel(PreTrainedModel):
    config: FireredAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of FireRed isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.audio_config.initializer_range
        )

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()




class FireRedEncoder(nn.Module):
    """FireRedConformer encoder module."""
    
    def __init__(self, config: FireredAudioEncoderConfig
        ):
        super().__init__()

        config.output_size = 1280
        self._output_size = config.output_size

        self.input_preprocessor = Conv2dSubsampling(config.input_size, config.output_size)
        self.positional_encoding = RelPositionalEncoding(config.output_size)
        self.dropout = nn.Dropout(config.residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(config.num_blocks):
            block = RelPosEmbConformerBlock(config.output_size, config.attention_heads,
                        config.residual_dropout,
                        config.dropout_rate, config.kernel_size)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths, pad=True):
        
        padded_input_place_hold = padded_input

        if pad:
            padded_input = F.pad(padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1), 'constant', 0.0)
            
            
        padded_input = padded_input.to(padded_input_place_hold.dtype)
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input, src_mask)
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                   pad_mask=src_mask)
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i]:] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)
    


# 小红书的fireredasr中的adapter
class FireRedMultiModalProjector(nn.Module):
    def __init__(self, config: FireredAudioConfig):
        super().__init__()
        
        config.audio_config.downsample_rate = 2
        config.audio_config.encoder_dim = 1280
        config.audio_config.llm_dim = 3584
        
        self.ds = config.audio_config.downsample_rate
        self.linear1 = nn.Linear(config.audio_config.encoder_dim * config.audio_config.downsample_rate, config.audio_config.llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.audio_config.llm_dim, config.audio_config.llm_dim)

    def forward(self, x, x_lens):
        '''
        以后可以在这个地方搞一点点事情
        '''
        
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens
    
    
class FireRedForConditionalGeneration(FireRedPreTrainedModel, GenerationMixin):
    def __init__(self, config: FireredAudioConfig):
        super().__init__(config)

        self.audio_tower = FireRedEncoder(config.audio_config)
        self.multi_modal_projector = FireRedMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels=None
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape                                      # 音频特征大小
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)                                                                   # 音频特征mask
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)                     # 提取音频特征有效部分
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_id
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)                                  # 音频数目

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.config.audio_token_id) & (attention_mask == 1)
        )                                                                           

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )
            
        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        feats_lengths: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, FireRedCausalLMOutputWithPast]:
        r"""
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, FireRedForConditionalGeneration

        >>> model = FireRedForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""
        
        
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        target_device = input_ids.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feats_lengths = feats_lengths.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                input_features = input_features.to(inputs_embeds.dtype)
                selected_audio_feature, audio_feat_lengths, _ = self.audio_tower(input_features, feats_lengths)
                audio_features, audio_feat_lengths = self.multi_modal_projector(selected_audio_feature, audio_feat_lengths)

                audio_features = audio_features.to(input_features.dtype)
                # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                audio_tokens = input_ids == self.config.audio_token_id
                legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                if legacy_processing:
                    logger.warning_once(
                        "Expanding inputs for audio tokens in FireRed should be done in processing."
                    )
                    
                    inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                        audio_features, audio_feat_lengths, inputs_embeds, input_ids, attention_mask, labels
                    )
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_feat_lengths.device)[None, :]
                    audio_features_mask = audio_features_mask < audio_feat_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)
        # import pdb;
        # pdb.set_trace()       
       
        # attention_mask = None
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FireRedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage
        

        input_features = kwargs.pop("input_features", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            model_inputs["input_features"] = input_features

        return model_inputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        feats_lengths: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,

    ):
        target_device = input_ids.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feats_lengths = feats_lengths.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)
            
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                input_features = input_features.to(inputs_embeds.dtype)
                selected_audio_feature, audio_feat_lengths, _ = self.audio_tower(input_features, feats_lengths)
                audio_features, audio_feat_lengths = self.multi_modal_projector(selected_audio_feature, audio_feat_lengths)

                audio_features = audio_features.to(selected_audio_feature.dtype)
                # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                audio_tokens = input_ids == self.config.audio_token_id
                legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                if legacy_processing:
                    logger.warning_once(
                        "Expanding inputs for audio tokens in FireRed should be done in processing."
                    )
                    
                    inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                        audio_features, audio_feat_lengths, inputs_embeds, input_ids, attention_mask
                    )
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_feat_lengths.device)[None, :]
                    audio_features_mask = audio_features_mask < audio_feat_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)
                    
        result = {}        
        generated_ids = self.language_model.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=500,
            use_cache=True,
            **{k: (v.to(input_ids.device) if torch.is_tensor(v) else v) for k, v in kwargs.items()},
        )
        result['sequences'] = torch.cat([input_ids, generated_ids.sequences.to(input_ids.device)], dim=-1)
        
        return result
        


__all__ = ["FireRedForConditionalGeneration", "FireRedPreTrainedModel", "FireRedEncoder"]
