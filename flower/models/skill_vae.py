import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer



###############################################################################
#
# Skill-VAE module
#
###############################################################################

def get_fsq_level(codebook_size):
    power = int(np.log2(codebook_size))
    if power == 4: # 16
        fsq_level = [5, 3]
    elif power == 6: # 64
        fsq_level = [8, 8]
    elif power == 8: # 256
        fsq_level = [8, 6, 5]
    elif power == 9: # 512
        fsq_level = [8, 8, 8]
    elif power == 10: # 1024
        fsq_level = [8, 5, 5, 5]
    elif power == 11: # 2048
        fsq_level = [8, 8, 6, 5]
    elif power == 12: # 4096
        fsq_level = [7, 5, 5, 5, 5]
    return fsq_level


class SkillVAE(nn.Module):
    def __init__(self,
                 action_dim,
                 encoder_dim,
                 decoder_dim,
 
                 skill_block_size,
                 downsample_factor, 

                 attn_pdrop,
                 use_causal_encoder,
                 use_causal_decoder,
 
                 encoder_heads,
                 encoder_layers,
                 decoder_heads,
                 decoder_layers,
 
                 vq_type,
                 fsq_level,
                 codebook_dim,
                 codebook_size,
                 input_action_dim=None,
                 output_action_dim=None,
                 ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.skill_block_size = skill_block_size
        self.downsample_factor = downsample_factor
        self.use_causal_encoder = use_causal_encoder
        self.use_causal_decoder = use_causal_decoder
        self.vq_type = vq_type
        self.fsq_level = fsq_level
        self.action_dim = action_dim
        self.input_action_dim = action_dim if input_action_dim is None else input_action_dim
        self.output_action_dim = action_dim if output_action_dim is None else output_action_dim

        assert int(np.log2(downsample_factor)) == np.log2(downsample_factor), 'downsample_factor must be a power of 2'
        strides = [2] * int(np.log2(downsample_factor)) + [1]
        kernel_sizes = [5] + [3] * int(np.log2(downsample_factor))
        if len(strides) == 1:
            kernel_sizes = [3, 2]
            strides = [1,1]

        if vq_type == 'vq':
            self.vq = VectorQuantize(dim=encoder_dim, codebook_dim=codebook_dim, codebook_size=codebook_size)
        elif vq_type == 'fsq':
            if fsq_level is None:
                fsq_level = get_fsq_level(codebook_size)
            self.vq = FSQ(dim=encoder_dim, levels=fsq_level)
        else:
            raise NotImplementedError('Unknown vq_type')
        self.action_proj = nn.Linear(self.input_action_dim, encoder_dim)
        self.action_head = nn.Linear(decoder_dim, self.output_action_dim)
        self.conv_block = ResidualTemporalBlock(
            encoder_dim, encoder_dim, kernel_size=kernel_sizes, 
            stride=strides, causal=use_causal_encoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, 
                                                   nhead=encoder_heads, 
                                                   dim_feedforward=4*encoder_dim, 
                                                   dropout=attn_pdrop, 
                                                   activation='gelu', 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, 
                                              num_layers=encoder_layers,
                                              enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim,
                                                   nhead=decoder_heads,
                                                   dim_feedforward=4*decoder_dim,
                                                   dropout=attn_pdrop,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(decoder_dim)
    
    def encode(self, act, obs_emb=None):
        x = self.action_proj(act)
        x = self.conv_block(x)
        B, H, D = x.shape
        
        if obs_emb is not None:
            x = torch.cat([obs_emb, x], dim=1)
        x = self.add_positional_emb(x)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.encoder(x, mask=mask, is_causal=True)
        else:
            x = self.encoder(x)

        x = x[:, -H:]

        return x

    def quantize(self, z):
        if self.vq_type == 'vq':
            codes, indices, commitment_loss = self.vq(z)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size, device=z.device)
        else:
            codes, indices = self.vq(z)
            commitment_loss = torch.tensor([0.0], device=z.device)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size, device=z.device)
        ## pp_sample is the average number of unique indices per sequence while pp is for the whole batch
        pp_sample = torch.tensor(np.mean([len(torch.unique(index_seq)) for index_seq in indices])/z.shape[1], device=z.device)
        return codes, indices, pp, pp_sample, commitment_loss

    def decode(self, codes, obs_emb=None):
        x = self.fixed_positional_emb(torch.zeros((codes.shape[0], self.skill_block_size, self.decoder_dim), dtype=codes.dtype, device=codes.device))
        if obs_emb is not None:
            codes = torch.cat([obs_emb, codes], dim=1)
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.decoder(x, codes, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, codes)
        x = self.action_head(x)
        return x

    def forward(self, act, obs_emb=None):
        z = self.encode(act, obs_emb=obs_emb)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes, obs_emb=obs_emb)
        return x, pp, pp_sample, commitment_loss, codes

    def get_indices(self, act, obs_emb=None):
        z = self.encode(act, obs_emb=obs_emb)
        _, indices, _, _, _ = self.quantize(z)
        return indices
    
    def decode_actions(self, indices):
        if self.vq_type == 'fsq':
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.get_output_from_indices(indices)
        x = self.decode(codes)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class AdaLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, cond_dim, dim_feedforward, dropout, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _modulate(self, x, shift, scale):
        return x * (1.0 + scale) + shift

    def forward(self, src, cond, src_mask=None, is_causal=False):
        gate_attn, shift_attn, scale_attn, gate_ffn, shift_ffn, scale_ffn = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )

        attn_in = self._modulate(self.norm1(src), shift_attn, scale_attn)
        attn_out = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=src_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        src = src + (1.0 + gate_attn) * self.dropout1(attn_out)

        ffn_in = self._modulate(self.norm2(src), shift_ffn, scale_ffn)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(ffn_in))))
        src = src + (1.0 + gate_ffn) * self.dropout2(ffn_out)
        return src


class AdaLNTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer] + [
            AdaLNTransformerEncoderLayer(
                layer.self_attn.embed_dim,
                layer.self_attn.num_heads,
                layer.adaLN_modulation[-1].in_features,
                layer.linear1.out_features,
                layer.dropout.p,
            )
            for _ in range(num_layers - 1)
        ])

    def forward(self, src, cond, mask=None, is_causal=False):
        output = src
        for layer in self.layers:
            output = layer(output, cond, src_mask=mask, is_causal=is_causal)
        return output


class SkillVAEFTAdaLN(SkillVAE):
    def __init__(self,
                 *args,
                 ft_dim=6,
                 ft_downsample_mode='avg',
                 ft_conv_strides=None,
                 ft_conv_kernel_sizes=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ft_conditioning = True
        self.ft_dim = ft_dim
        self.ft_downsample_mode = ft_downsample_mode
        self.ft_conv_strides = list(ft_conv_strides) if ft_conv_strides is not None else None
        self.ft_conv_kernel_sizes = list(ft_conv_kernel_sizes) if ft_conv_kernel_sizes is not None else None

        # common projection: (B, H, 6) or (B, H, S, 6) -> (..., D)
        self.ft_proj = nn.Linear(ft_dim, self.encoder_dim)

        if ft_downsample_mode == 'conv':
            self.ft_conv_block = ResidualTemporalBlock(
                self.encoder_dim,
                self.encoder_dim,
                kernel_size=self._conv_kernel_sizes(),
                stride=self._conv_strides(),
                causal=self.use_causal_encoder,
            )
            cond_dim = self.encoder_dim

        elif ft_downsample_mode in ['avg', 'max']:
            cond_dim = self.encoder_dim

        elif ft_downsample_mode == 'avg_max':
            # avg + max gives 2D, project back to D
            self.ft_avg_max_proj = nn.Linear(2 * self.encoder_dim, self.encoder_dim)
            cond_dim = self.encoder_dim

        else:
            raise ValueError(f"Unsupported ft_downsample_mode: {ft_downsample_mode}")

        layer = AdaLNTransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=self.encoder.layers[0].self_attn.num_heads,
            cond_dim=cond_dim,
            dim_feedforward=4 * self.encoder_dim,
            dropout=self.encoder.layers[0].dropout.p,
            activation='gelu',
        )
        self.encoder = AdaLNTransformerEncoder(layer, len(self.encoder.layers))

    def _conv_strides(self):
        if self.ft_conv_strides is not None:
            return self.ft_conv_strides
        strides = [2] * int(np.log2(self.downsample_factor)) + [1]
        if len(strides) == 1:
            strides = [1, 1]
        return strides

    def _conv_kernel_sizes(self):
        if self.ft_conv_kernel_sizes is not None:
            return self.ft_conv_kernel_sizes
        kernel_sizes = [5] + [3] * int(np.log2(self.downsample_factor))
        if len(self._conv_strides()) == 2 and self.downsample_factor == 1:
            kernel_sizes = [3, 2]
        return kernel_sizes

    def _pool_ft(self, ft, target_len):
        if ft.dim() == 4:
            B, T, samples_per_step, C = ft.shape
            ft = ft.reshape(B, T * samples_per_step, C)
            T = T * samples_per_step
            pool_width = self.downsample_factor * samples_per_step
        elif ft.dim() == 3:
            B, T, C = ft.shape
            pool_width = self.downsample_factor
        else:
            raise ValueError(f"Expected ft to have shape (B,T,C) or (B,T,S,C), got {tuple(ft.shape)}")

        target_T = target_len * pool_width
        if T < target_T:
            pad = ft[:, -1:].expand(B, target_T - T, C)
            ft = torch.cat([ft, pad], dim=1)
        elif T > target_T:
            ft = ft[:, :target_T]

        ft = ft.reshape(B, target_len, pool_width, C)
        if self.ft_downsample_mode == 'avg':
            return ft.mean(dim=2)
        if self.ft_downsample_mode == 'max':
            max_idx = ft.abs().argmax(dim=2, keepdim=True)
            return ft.gather(dim=2, index=max_idx).squeeze(2)
        if self.ft_downsample_mode == 'avg_max':
            avg = ft.mean(dim=2)
            max_idx = ft.abs().argmax(dim=2, keepdim=True)
            max_abs = ft.gather(dim=2, index=max_idx).squeeze(2)
            return self.ft_avg_max_proj(torch.cat([avg, max_abs], dim=-1))
        raise ValueError(f"Pooling is not defined for mode {self.ft_downsample_mode}")

    @staticmethod
    def _deterministic_adaptive_avg_pool1d(x, target_len):
        input_len = x.size(-1)
        if input_len == target_len:
            return x
        if input_len % target_len == 0:
            kernel_size = input_len // target_len
            return F.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_size)

        # Match adaptive average pooling's variable-width binning without using
        # adaptive_avg_pool1d, whose CUDA backward can be nondeterministic.
        pooled = []
        for i in range(target_len):
            start = int(np.floor(i * input_len / target_len))
            end = int(np.ceil((i + 1) * input_len / target_len))
            pooled.append(x[..., start:end].mean(dim=-1))
        return torch.stack(pooled, dim=-1)

    def _get_ft_cond(self, ft, target_len):
        if ft is None:
            return torch.zeros((1, target_len, self.encoder_dim), device=self.device)

        ft = self.ft_proj(ft)

        if ft.dim() == 4:
            B, T, samples_per_step, C = ft.shape

        if self.ft_downsample_mode == 'conv':
            if ft.dim() == 4:
                B, T, samples_per_step, C = ft.shape
                ft = ft.reshape(B, T * samples_per_step, C)
            cond = self.ft_conv_block(ft)
            if cond.size(1) != target_len:
                cond = cond.transpose(1, 2)
                if cond.size(-1) > target_len:
                    cond = self._deterministic_adaptive_avg_pool1d(cond, target_len)
                else:
                    cond = F.interpolate(
                        cond,
                        size=target_len,
                        mode='linear',
                        align_corners=False,
                    )
                cond = cond.transpose(1, 2)
            return cond

        return self._pool_ft(ft, target_len)

    def encode(self, act, obs_emb=None, ft=None):
        x = self.action_proj(act)
        x = self.conv_block(x)
        B, H, D = x.shape

        ft_cond = self._get_ft_cond(ft, H)
        if ft_cond.size(0) == 1 and B != 1:
            ft_cond = ft_cond.expand(B, -1, -1)

        if obs_emb is not None:
            obs_cond = torch.zeros(
                (B, obs_emb.size(1), ft_cond.size(-1)),
                dtype=ft_cond.dtype,
                device=ft_cond.device,
            )
            ft_cond = torch.cat([obs_cond, ft_cond], dim=1)
            x = torch.cat([obs_emb, x], dim=1)
        x = self.add_positional_emb(x)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.encoder(x, ft_cond, mask=mask, is_causal=True)
        else:
            x = self.encoder(x, ft_cond)

        x = x[:, -H:]
        return x

    def forward(self, act, obs_emb=None, ft=None):
        z = self.encode(act, obs_emb=obs_emb, ft=ft)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes, obs_emb=obs_emb)
        return x, pp, pp_sample, commitment_loss, codes

    def get_indices(self, act, obs_emb=None, ft=None):
        z = self.encode(act, obs_emb=obs_emb, ft=ft)
        _, indices, _, _, _ = self.quantize(z)
        return indices


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


# TODO: delete deconv modules for final release version
class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, output_padding=stride-1)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
