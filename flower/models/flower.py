import logging
import os
import sys  # MODIFIED: QueST repo path를 동적으로 추가하기 위해 사용합니다.
from pathlib import Path  # MODIFIED: checkpoint/repo path를 안전하게 처리하기 위해 사용합니다.
from typing import Any, Dict, Optional, Tuple, Collection, List
from functools import partial
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import einops
from einops import rearrange, repeat
from torch import einsum
from einops_exts import rearrange_many
import wandb
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from flower.models.networks.transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock, 
    stateless_norm
)
from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from flower.callbacks.ema import EMA
from flower.models.utils import ActionIndex, generate_policy_prompt

# MODIFIED: Frozen SkillVAE를 FLOWER의 AdaLN conditioning으로 사용하기 위한 fallback import입니다.
# checkpoint cfg가 있으면 hydra.instantiate()를 우선 사용하고, cfg가 없는 checkpoint에서만 이 fallback을 사용합니다.
try:
    from flower.models.quest_skill_vae import SkillVAE
except ImportError:
    try:
        from flower.models.skill_vae import SkillVAE
    except ImportError:
        try:
            from skill_vae import SkillVAE
        except ImportError:
            SkillVAE = None

logger = logging.getLogger(__name__)


class CausalConv1dHistoryEncoder(nn.Module):
    """Encode a fixed-length history with left-padded causal Conv1d layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "last",
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if pool not in ["last", "mean"]:
            raise ValueError(f"pool must be 'last' or 'mean', got {pool}")

        self.kernel_size = int(kernel_size)
        self.pool = pool
        self.convs = nn.ModuleList()
        in_channels = int(input_dim)
        for _ in range(int(n_layers)):
            self.convs.append(nn.Conv1d(in_channels, int(hidden_dim), kernel_size=self.kernel_size))
            in_channels = int(hidden_dim)

        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.proj = Mlp(
            in_features=int(hidden_dim),
            hidden_features=int(out_dim),
            out_features=int(out_dim),
            drop=float(dropout),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3:
            raise ValueError(f"history should be [B,T,D], got {tuple(history.shape)}")

        x = history.transpose(1, 2).contiguous()
        left_pad = self.kernel_size - 1
        for conv in self.convs:
            x = F.pad(x, (left_pad, 0))
            x = F.silu(conv(x))
            x = self.dropout(x)

        pooled = x.mean(dim=-1) if self.pool == "mean" else x[:, :, -1]
        return self.proj(self.norm(pooled))


class FLOWERVLA(pl.LightningModule):
    def __init__(
        self,
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-base",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.2,

        # Auxiliary BBox Loss
        bbox_loss: bool = False,
        bbox_loss_weight: float = 0.05,
        bbox_key: str = "bbox",
        
        # Model Structure
        multistep: int = 10,
        num_sampling_steps: int = 5,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        
        # Model flags
        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        round_robin_conditioning: bool = False,
        return_act_chunk: bool = False,
        
        # DiT Configuration
        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        
        # RoPE Configuration
        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
        
        # Optimizer Configuration
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,

        load_pretrained: bool = False,
        pretrained_model_path: str = None,

        # MODIFIED: SkillVAE 기반 AdaLN conditioning 옵션입니다.
        # SkillVAE는 checkpoint에서 로드한 뒤 frozen encoder/quantizer로만 사용합니다.
        use_skill_vae_adaln: bool = False,
        use_skill_vae_adaln_conv1d: bool = False,
        use_proprio_skill_history_mlp_adaln: bool = False,
        skill_vae_ckpt_path: Optional[str] = None,
        skill_vae_repo_path: Optional[str] = None,  # MODIFIED: flower_skill_pred.py처럼 QueST/SkillVAE repo path를 sys.path에 추가합니다.
        skill_vae_config: Optional[DictConfig] = None,
        skill_vae_action_dim: int = 7,
        skill_vae_encoder_dim: int = 256,
        skill_vae_decoder_dim: int = 256,
        skill_vae_skill_block_size: int = 32,
        skill_vae_downsample_factor: int = 4,
        skill_vae_attn_pdrop: float = 0.1,
        skill_vae_use_causal_encoder: bool = True,
        skill_vae_use_causal_decoder: bool = True,
        skill_vae_encoder_heads: int = 4,
        skill_vae_encoder_layers: int = 4,
        skill_vae_decoder_heads: int = 4,
        skill_vae_decoder_layers: int = 4,
        skill_vae_vq_type: str = "fsq",
        skill_vae_fsq_level: Optional[List[int]] = None,
        skill_vae_codebook_dim: int = 256,
        skill_vae_codebook_size: int = 1024,
        skill_vae_cond_pool: str = "mean",
        skill_vae_strict_load: bool = False,
        skill_vae_allow_zero_history: bool = False,
        # MODIFIED: action history AdaLN ablation source입니다.
        # "skill_vae"는 frozen SkillVAE code를 사용하고, "raw_action"은 [B,32,7] history를 직접 MLP로 AdaLN합니다.
        history_adaln_source: str = "skill_vae",
        raw_action_history_hidden_dim: Optional[int] = None,
        # MODIFIED: SkillVAE 입력 action history에만 gripper relabeling을 적용합니다.
        # Flow Transformer의 denoising target/actions에는 이 처리를 적용하지 않습니다.
        skill_vae_relabel_gripper: bool = True,
        skill_vae_gripper_action_idx: int = -1,
        skill_vae_gripper_state_idx: int = -1,
        skill_vae_relabel_action_threshold: float = 0.9,
        skill_vae_relabel_state_threshold: float = 0.95,
        skill_vae_relabel_require_state: bool = False,
        # MODIFIED: right_force_history [T,6]를 causal Conv1d로 AdaLN condition에 추가합니다.
        use_force_history_adaln: bool = False,
        force_history_dim: int = 6,
        force_history_hidden_dim: int = 256,
        force_history_conv_layers: int = 2,
        force_history_kernel_size: int = 3,
        force_history_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if "use_skill_vae_conv1d" in kwargs:
            use_skill_vae_adaln_conv1d = bool(kwargs.pop("use_skill_vae_conv1d")) or bool(
                use_skill_vae_adaln_conv1d
            )
        use_skill_vae_adaln_conv1d = bool(use_skill_vae_adaln_conv1d)
        use_skill_vae_adaln = bool(use_skill_vae_adaln) or use_skill_vae_adaln_conv1d
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.action_space_index = ActionIndex()
        # Initialize model flags and configurations
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            use_proprio=use_proprio,
            round_robin_conditioning=round_robin_conditioning,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
            # MODIFIED: right_force_history causal Conv1d condition 옵션입니다.
            use_force_history_adaln=use_force_history_adaln,
            force_history_dim=force_history_dim,
            force_history_hidden_dim=force_history_hidden_dim,
            force_history_conv_layers=force_history_conv_layers,
            force_history_kernel_size=force_history_kernel_size,
            force_history_dropout=force_history_dropout,
            # MODIFIED: SkillVAE code embedding을 AdaLN global condition에 더할지 결정합니다.
            use_skill_vae_adaln=use_skill_vae_adaln,
            use_skill_vae_adaln_conv1d=use_skill_vae_adaln_conv1d,
            use_proprio_skill_history_mlp_adaln=use_proprio_skill_history_mlp_adaln,
        )
        self.obs_modalities = []
        # Initialize model dimensions
        self._init_dimensions(
            dit_dim=dit_dim,
            n_heads=n_heads,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )
        self.target_modality = "actions"
        # Setup VLM and core components
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.use_bbox_loss = bool(bbox_loss)
        self.bbox_loss_weight = bbox_loss_weight
        self.bbox_key = bbox_key
        self.bbox_head = None
        if self.use_bbox_loss:
            self.bbox_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
            )
        self.action_type_adaln = action_type_adaln
        self.use_proprio = use_proprio
        # Setup DiT components
        self._setup_dit_components(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        # MODIFIED: SkillVAE checkpoint를 로드하고 frozen condition encoder로 붙입니다.
        self._setup_skill_vae_conditioner(
            skill_vae_ckpt_path=skill_vae_ckpt_path,
            skill_vae_repo_path=skill_vae_repo_path,
            skill_vae_config=skill_vae_config,
            action_dim=skill_vae_action_dim,
            encoder_dim=skill_vae_encoder_dim,
            decoder_dim=skill_vae_decoder_dim,
            skill_block_size=skill_vae_skill_block_size,
            downsample_factor=skill_vae_downsample_factor,
            attn_pdrop=skill_vae_attn_pdrop,
            use_causal_encoder=skill_vae_use_causal_encoder,
            use_causal_decoder=skill_vae_use_causal_decoder,
            encoder_heads=skill_vae_encoder_heads,
            encoder_layers=skill_vae_encoder_layers,
            decoder_heads=skill_vae_decoder_heads,
            decoder_layers=skill_vae_decoder_layers,
            vq_type=skill_vae_vq_type,
            fsq_level=skill_vae_fsq_level,
            codebook_dim=skill_vae_codebook_dim,
            codebook_size=skill_vae_codebook_size,
            cond_pool=skill_vae_cond_pool,
            strict_load=skill_vae_strict_load,
            allow_zero_history=skill_vae_allow_zero_history,
            relabel_gripper=skill_vae_relabel_gripper,
            gripper_action_idx=skill_vae_gripper_action_idx,
            gripper_state_idx=skill_vae_gripper_state_idx,
            relabel_action_threshold=skill_vae_relabel_action_threshold,
            relabel_state_threshold=skill_vae_relabel_state_threshold,
            relabel_require_state=skill_vae_relabel_require_state,
            # MODIFIED: SkillVAE code vs raw action history AdaLN을 config로 전환합니다.
            history_adaln_source=history_adaln_source,
            raw_action_history_hidden_dim=raw_action_history_hidden_dim,
        )
        
        # Initialize state tracking
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        # MODIFIED: 추론 시 실제로 실행된 최근 32개 action을 rolling buffer로 유지합니다.
        self.skill_action_history = None
        # MODIFIED: inference에서도 SkillVAE 입력 relabeling을 위해 action과 같은 길이의 gripper state history를 유지합니다.
        self.skill_gripper_state_history = None
        self.modality_scope = "lang"
        # Save optimizer config
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optimizer_type = optimizer_type

        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)

    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""
        print(f"Loading pretrained weights from {pretrained_model_path}...")
        # Determine file type and load accordingly
        if pretrained_model_path.endswith('.safetensors'):
            # Load safetensors file
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}  # Create checkpoint-like structure for compatibility
            print("Loaded safetensors file")
        else:
            # Load PyTorch checkpoint (.pt, .pth, .ckpt)
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            # Extract the state dict (handle PyTorch Lightning or plain models)
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract the state dict (handle PyTorch Lightning or plain models)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and 
                "EMA" in checkpoint["callbacks"] and 
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
                
                print("Found EMA weights in checkpoint, attempting to load them...")
                ema_weights_list = checkpoint['callbacks']['EMA']['ema_weights']
                
                # Get the original state dict to use as a reference for parameter names and shapes
                original_state_dict = checkpoint.get("state_dict", checkpoint)
                
                # Create a new state dict by matching EMA weights with original parameter names
                state_dict = {}
                ema_idx = 0
                
                for param_name, original_param in original_state_dict.items():
                    if ema_idx < len(ema_weights_list):
                        ema_weight = ema_weights_list[ema_idx]
                        
                        # Check if shapes match
                        if ema_weight.shape == original_param.shape:
                            state_dict[param_name] = ema_weight
                            ema_idx += 1
                        else:
                            # Shape mismatch - try to find the correct EMA weight by shape
                            found_match = False
                            for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                                if ema_weights_list[temp_idx].shape == original_param.shape:
                                    state_dict[param_name] = ema_weights_list[temp_idx]
                                    # Swap to maintain order
                                    ema_weights_list[temp_idx], ema_weights_list[ema_idx] = ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                    ema_idx += 1
                                    found_match = True
                                    break
                            
                            if not found_match:
                                # If no match found, use original parameter
                                print(f"Warning: No matching EMA weight found for {param_name}, using original")
                                state_dict[param_name] = original_param
                    else:
                        # No more EMA weights available, use original
                        print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                        state_dict[param_name] = original_param
                
                print(f"Successfully matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Fix key mismatches: remove 'agent.' prefix if it exists
        new_state_dict = {}
        # Handle language encoder/model naming mismatch
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")  # Remove 'agent.' if it exists
            
            # Handle language encoder/model naming mismatch
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            #elif "vlm.language_model." in new_key and "vlm.language_model.model." not in new_key:
                # If it's already language_model but missing the nested structure, add it
                #new_key = new_key.replace("vlm.language_model.", "vlm.language_model.model.encoder.")
            # Handle MLP naming mismatch
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        # Load the state dict with strict=False to handle mismatches
        #missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False) 기존 코드
        model_state = self.state_dict()
        filtered_state_dict = {}

        for key, value in new_state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                print(f"Skipping mismatched key: {key} ckpt={value.shape} model={model_state[key].shape}")
                continue
            filtered_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)


        # Log mismatches for debugging
        print(f"Pretrained weights loaded with the following issues:")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")  # Show first 30 for brevity
        if unexpected_keys:
            print(f"  ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")  # Show first 30 for brevity
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!")

        # Handle mean-resizing for missing embeddings if enabled
        if mean_resizing:
            self._initialize_new_embeddings(new_state_dict)

        return missing_keys, unexpected_keys

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
            
        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')
        
        self.use_adaln_cond = self.use_adaln_cond 
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_proprio = self.use_proprio 
        self.round_robin_conditioning = bool(self.round_robin_conditioning)
        self.use_skill_vae_adaln = bool(self.use_skill_vae_adaln)
        self.use_skill_vae_adaln_conv1d = bool(self.use_skill_vae_adaln_conv1d)
        if self.use_skill_vae_adaln_conv1d:
            self.use_skill_vae_adaln = True
        self.use_proprio_skill_history_mlp_adaln = self.use_proprio_skill_history_mlp_adaln
        if self.use_proprio_skill_history_mlp_adaln and not (self.use_proprio and self.use_skill_vae_adaln):
            raise ValueError(
                "use_proprio_skill_history_mlp_adaln requires both "
                "use_proprio=True and use_skill_vae_adaln=True."
            )
        self.use_force_history_adaln = bool(self.use_force_history_adaln)
        self.force_history_dim = int(self.force_history_dim)
        self.force_history_hidden_dim = int(self.force_history_hidden_dim)
        self.force_history_conv_layers = int(self.force_history_conv_layers)
        self.force_history_kernel_size = int(self.force_history_kernel_size)
        self.force_history_dropout = float(self.force_history_dropout)
        if self.use_force_history_adaln and self.force_history_dim <= 0:
            raise ValueError(f"force_history_dim must be positive, got {self.force_history_dim}")
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_cross_attn = self.use_cross_attn
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions"""
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")


    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components(self, **kwargs):
        """Setup DiT model components"""
        # Extract parameters
        dit_dim = kwargs['dit_dim']
        n_heads = kwargs['n_heads']
        n_layers = kwargs['n_layers']
        hidden_dim = kwargs['hidden_dim']
        use_cross_attn = kwargs['use_cross_attn']
        use_rope = kwargs['use_rope']
        use_nope = kwargs['use_nope']

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        if self.use_proprio_skill_history_mlp_adaln:
            self.proprio_skill_history_fusion = nn.ModuleDict()
        self.force_history_encoder = None
        if self.use_force_history_adaln:
            self.force_history_encoder = CausalConv1dHistoryEncoder(
                input_dim=self.force_history_dim,
                hidden_dim=self.force_history_hidden_dim,
                out_dim=dit_dim,
                kernel_size=self.force_history_kernel_size,
                n_layers=self.force_history_conv_layers,
                dropout=self.force_history_dropout,
            ).to(self.device)
            
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Core components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(dit_dim, max_actions=len(self.action_space_index.action_spaces))


        # Positional encoding if not using ROPE/NOPE
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, kwargs['act_window_size'], dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=kwargs['attn_pdrop'],
                resid_pdrop=kwargs['resid_pdrop'],
                mlp_pdrop=kwargs['mlp_pdrop'],
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=kwargs['query_seq_len'],
                rope_theta=kwargs['rope_theta'],

            ) for _ in range(n_layers)
        ])

        # Create components per action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            
            # Add encoder/decoder for this action
            self.action_encoders[action_name] =  Mlp(in_features=input_dim, hidden_features=dit_dim, out_features=dit_dim, bias=True)
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)
                
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

            if self.use_proprio:
                self.proprio_encoders[action_name] = Mlp(
                    in_features=self.lowdim_obs_dim,
                    hidden_features=dit_dim,
                    out_features=dit_dim,
                    drop=0.2,
                ).to(self.device)

            if self.use_proprio_skill_history_mlp_adaln:
                fusion_input_dim = dit_dim * (2 + int(self.use_force_history_adaln))
                self.proprio_skill_history_fusion[action_name] = Mlp(
                    in_features=fusion_input_dim,
                    hidden_features=dit_dim,
                    out_features=dit_dim,
                    drop=0.2,
                ).to(self.device)

    # ==========================================================================
    # MODIFIED: Frozen SkillVAE -> Flow Transformer AdaLN conditioning utilities
    # ==========================================================================
    def _setup_skill_vae_conditioner(
        self,
        skill_vae_ckpt_path: Optional[str],
        skill_vae_repo_path: Optional[str],
        skill_vae_config: Optional[DictConfig],
        action_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        skill_block_size: int,
        downsample_factor: int,
        attn_pdrop: float,
        use_causal_encoder: bool,
        use_causal_decoder: bool,
        encoder_heads: int,
        encoder_layers: int,
        decoder_heads: int,
        decoder_layers: int,
        vq_type: str,
        fsq_level: Optional[List[int]],
        codebook_dim: int,
        codebook_size: int,
        cond_pool: str,
        strict_load: bool,
        allow_zero_history: bool,
        relabel_gripper: bool,
        gripper_action_idx: int,
        gripper_state_idx: int,
        relabel_action_threshold: float,
        relabel_state_threshold: float,
        relabel_require_state: bool,
        # MODIFIED: action history AdaLN ablation 옵션입니다.
        history_adaln_source: str = "skill_vae",
        raw_action_history_hidden_dim: Optional[int] = None,
    ):
        """Create a frozen SkillVAE and a trainable projection into dit_dim.

        MODIFIED: flower_skill_pred.py의 QueST autoencoder loading 방식과 동일하게,
        checkpoint 내부 cfg/config가 있으면 cfg.algo.policy.autoencoder를 Hydra instantiate로
        복원합니다. cfg가 없는 checkpoint만 기존 수동 SkillVAE(**kwargs) fallback을 사용합니다.
        """
        self.skill_vae = None
        self.skill_vae_proj = None
        self.skill_vae_conv1d_encoder = None
        # MODIFIED: raw_action baseline에서 사용하는 trainable history encoder입니다.
        self.raw_action_history_proj = None
        # MODIFIED: "skill_vae" 또는 "raw_action" 중 어떤 history representation을 AdaLN에 넣을지 저장합니다.
        self.history_adaln_source = str(history_adaln_source)
        self.skill_vae_action_dim = action_dim
        self.skill_vae_encoder_dim = encoder_dim
        self.skill_vae_skill_block_size = skill_block_size
        self.skill_vae_cond_pool = cond_pool
        self.skill_vae_allow_zero_history = allow_zero_history
        # MODIFIED: 아래 relabel 옵션은 SkillVAE 입력 history에만 쓰입니다.
        self.skill_vae_relabel_gripper = bool(relabel_gripper)
        self.skill_vae_gripper_action_idx = int(gripper_action_idx)
        self.skill_vae_gripper_state_idx = int(gripper_state_idx)
        self.skill_vae_relabel_action_threshold = float(relabel_action_threshold)
        self.skill_vae_relabel_state_threshold = float(relabel_state_threshold)
        self.skill_vae_relabel_require_state = bool(relabel_require_state)

        if not self.use_skill_vae_adaln:
            return
        if self.history_adaln_source not in ["skill_vae", "raw_action"]:
            raise ValueError(
                f"history_adaln_source는 'skill_vae' 또는 'raw_action'만 지원합니다. got={self.history_adaln_source}"
            )
        if cond_pool not in ["mean", "last"]:
            raise ValueError("skill_vae_cond_pool은 'mean' 또는 'last'만 지원합니다.")
        if self.use_skill_vae_adaln_conv1d and self.history_adaln_source != "skill_vae":
            raise ValueError("use_skill_vae_adaln_conv1d=True는 history_adaln_source='skill_vae'에서만 지원합니다.")

        # MODIFIED: raw_action baseline은 SkillVAE checkpoint를 로드하지 않습니다.
        # 이미 dataloader가 만든 [B,32,7] action history를 flatten 후 MLP로 dit_dim에 projection합니다.
        # 사용자가 요청한 대로 raw action history에는 gripper relabeling 함수를 적용하지 않습니다.
        if self.history_adaln_source == "raw_action":
            hidden_dim = int(raw_action_history_hidden_dim or self.dit_dim)
            flat_dim = int(skill_block_size) * int(action_dim)
            self.raw_action_history_proj = nn.Sequential(
                nn.LayerNorm(flat_dim),
                nn.Linear(flat_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.dit_dim),
            )
            print(
                f"Using raw action history AdaLN: [B,{skill_block_size},{action_dim}] "
                f"-> flatten {flat_dim} -> hidden {hidden_dim} -> dit_dim {self.dit_dim}"
            )
            return

        if skill_vae_ckpt_path is None:
            raise ValueError("history_adaln_source='skill_vae'이면 skill_vae_ckpt_path가 필요합니다.")

        fallback_vae_kwargs = dict(
            action_dim=action_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            skill_block_size=skill_block_size,
            downsample_factor=downsample_factor,
            attn_pdrop=attn_pdrop,
            use_causal_encoder=use_causal_encoder,
            use_causal_decoder=use_causal_decoder,
            encoder_heads=encoder_heads,
            encoder_layers=encoder_layers,
            decoder_heads=decoder_heads,
            decoder_layers=decoder_layers,
            vq_type=vq_type,
            fsq_level=fsq_level,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
        )

        # MODIFIED: checkpoint cfg/config가 있으면 그 설정을 우선 사용하여 SkillVAE 구조를 정확히 복원합니다.
        self.skill_vae = self._load_skill_vae_from_quest_checkpoint(
            ckpt_path=skill_vae_ckpt_path,
            quest_repo_path=skill_vae_repo_path,
            skill_vae_config=skill_vae_config,
            fallback_vae_kwargs=fallback_vae_kwargs,
            strict_load=strict_load,
        )
        self._freeze_skill_vae()

        # MODIFIED: 실제 로드된 autoencoder의 차원을 기준으로 projection을 만듭니다.
        actual_encoder_dim = int(getattr(self.skill_vae, "encoder_dim", fallback_vae_kwargs["encoder_dim"]))
        actual_action_dim = int(getattr(getattr(self.skill_vae, "action_proj", None), "in_features", fallback_vae_kwargs["action_dim"]))
        actual_skill_block_size = int(getattr(self.skill_vae, "skill_block_size", fallback_vae_kwargs["skill_block_size"]))
        actual_downsample_factor = int(getattr(self.skill_vae, "downsample_factor", fallback_vae_kwargs["downsample_factor"]))

        if self.use_skill_vae_adaln_conv1d:
            hidden_dim = int(raw_action_history_hidden_dim or actual_encoder_dim)
            self.skill_vae_conv1d_encoder = CausalConv1dHistoryEncoder(
                input_dim=actual_encoder_dim,
                hidden_dim=hidden_dim,
                out_dim=self.dit_dim,
                kernel_size=3,
                n_layers=2,
                dropout=attn_pdrop,
                pool="mean",
            ).to(self.device)
            print(
                f"Using SkillVAE-code Conv1d AdaLN: "
                f"[B,~{actual_skill_block_size // actual_downsample_factor},{actual_encoder_dim}] "
                f"-> mean pool -> hidden {hidden_dim} -> dit_dim {self.dit_dim}"
            )
        else:
            self.skill_vae_proj = nn.Sequential(
                nn.LayerNorm(actual_encoder_dim),
                nn.Linear(actual_encoder_dim, self.dit_dim),
            )

        self.skill_vae_action_dim = actual_action_dim
        self.skill_vae_encoder_dim = actual_encoder_dim
        self.skill_vae_skill_block_size = actual_skill_block_size

    def _load_skill_vae_from_quest_checkpoint(
        self,
        ckpt_path: str,
        quest_repo_path: Optional[str],
        skill_vae_config: Optional[DictConfig],
        fallback_vae_kwargs: Dict[str, Any],
        strict_load: bool = False,
    ) -> nn.Module:
        """Load a frozen QueST SkillVAE using the same cfg-driven style as flower_skill_pred.py.

        MODIFIED: 우선순위는 다음과 같습니다.
          1) checkpoint["cfg"] 또는 checkpoint["config"]에서 cfg.algo.policy.autoencoder를 복원
          2) autoencoder prefix를 제거한 state_dict를 load
          3) cfg가 없는 checkpoint는 fallback_vae_kwargs로 SkillVAE를 만들고 suffix matching으로 load
        """
        if quest_repo_path is not None:
            quest_repo_path = str(Path(quest_repo_path).expanduser())
            if os.path.exists(quest_repo_path) and quest_repo_path not in sys.path:
                sys.path.insert(0, quest_repo_path)

        try:
            from hydra.utils import instantiate
        except ImportError as exc:
            raise ImportError(
                "hydra.utils.instantiate를 import할 수 없습니다. QueST checkpoint cfg 복원을 위해 hydra-core가 필요합니다."
            ) from exc

        ckpt_path = str(Path(ckpt_path).expanduser())
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,  # MODIFIED: PyTorch 2.6 대응. QueST ckpt는 cfg/OmegaConf 객체를 포함할 수 있습니다.
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"SkillVAE checkpoint는 dict 형태여야 합니다: {ckpt_path}")

        cfg = checkpoint.get("cfg", checkpoint.get("config"))
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError(f"SkillVAE checkpoint에서 model/state_dict를 찾지 못했습니다: {ckpt_path}")

        # MODIFIED: checkpoint 저장 방식에 따라 autoencoder prefix가 다를 수 있어 가능한 prefix를 제거합니다.
        prefix_candidates = (
            "autoencoder.",
            "policy.autoencoder.",
            "algo.policy.autoencoder.",
            "module.autoencoder.",
            "module.policy.autoencoder.",
            "skill_vae.",
            "skill_target_autoencoder.",
        )
        autoencoder_state = {}
        matched_prefix = None
        for prefix in prefix_candidates:
            stripped = {
                key[len(prefix):]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
            }
            if stripped:
                autoencoder_state = stripped
                matched_prefix = prefix
                break
        if not autoencoder_state:
            autoencoder_state = {key: value for key, value in state.items() if isinstance(key, str) and torch.is_tensor(value)}

        if cfg is not None:
            cfg = OmegaConf.create(cfg)

            # MODIFIED: QueST config에서 autoencoder 정의를 복원합니다.
            autoencoder_cfg = OmegaConf.to_container(cfg.algo.policy.autoencoder, resolve=False)
            autoencoder_cfg["action_dim"] = int(cfg.task.shape_meta.action_dim)
            autoencoder_cfg["skill_block_size"] = int(cfg.algo.skill_block_size)
            autoencoder_cfg["downsample_factor"] = int(cfg.algo.downsample_factor)
            autoencoder_cfg["codebook_size"] = int(cfg.algo.codebook_size)

            if skill_vae_config is not None:
                override = OmegaConf.to_container(skill_vae_config, resolve=True) if isinstance(skill_vae_config, DictConfig) else dict(skill_vae_config)
                autoencoder_cfg.update(override)

            # MODIFIED: flower_skill_pred.py와 동일하게 프로젝트 내부 복사본 target을 우선 사용합니다.
            original_target = str(autoencoder_cfg.get("_target_", ""))
            has_ft_history_cfg = any(
                key in autoencoder_cfg
                for key in ("ft_dim", "ft_downsample_mode", "ft_conv_strides", "ft_conv_kernel_sizes")
            )
            has_ft_history_state = any(
                key.startswith(("ft_proj.", "ft_conv_block.", "ft_avg_max_proj."))
                or "adaLN_modulation" in key
                for key in autoencoder_state.keys()
            )
            if original_target.endswith("SkillVAEFTAdaLN") or has_ft_history_cfg or has_ft_history_state:
                autoencoder_cfg["_target_"] = "flower.models.quest_skill_vae.SkillVAEFTAdaLN"
                if "ft_project_condition" not in autoencoder_cfg:
                    autoencoder_cfg["ft_project_condition"] = any(
                        key.startswith("ft_proj.") for key in autoencoder_state.keys()
                    )
            else:
                autoencoder_cfg["_target_"] = "flower.models.quest_skill_vae.SkillVAE"

            autoencoder = self._instantiate_skill_vae_with_target_fallback(autoencoder_cfg, instantiate)
            load_strict = bool(strict_load or matched_prefix is not None)
            try:
                missing_keys, unexpected_keys = autoencoder.load_state_dict(autoencoder_state, strict=load_strict)
            except RuntimeError:
                if strict_load:
                    raise
                # MODIFIED: checkpoint prefix는 맞지만 local class와 일부 key가 다르면 학습을 막지 않도록 한번 더 완화합니다.
                missing_keys, unexpected_keys = autoencoder.load_state_dict(autoencoder_state, strict=False)

            if missing_keys:
                logger.warning("SkillVAE missing keys while loading: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("SkillVAE unexpected keys while loading: %s", unexpected_keys[:20])

            logger.info(
                "Loaded frozen SkillVAE from %s as %s with skill_block_size=%s, downsample_factor=%s, encoder_dim=%s",
                ckpt_path,
                autoencoder_cfg.get("_target_", "unknown"),
                getattr(autoencoder, "skill_block_size", "unknown"),
                getattr(autoencoder, "downsample_factor", "unknown"),
                getattr(autoencoder, "encoder_dim", "unknown"),
            )
            print(
                f"Loaded frozen SkillVAE from {ckpt_path}: "
                f"target={autoencoder_cfg.get('_target_', 'unknown')}, "
                f"matched_prefix={matched_prefix}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )
            return autoencoder

        # MODIFIED: cfg/config가 없는 checkpoint에 대한 기존 호환 fallback입니다.
        if SkillVAE is None:
            raise ImportError(
                "checkpoint에 cfg/config가 없고 SkillVAE fallback import도 실패했습니다. "
                "skill_vae.py를 flower.models.quest_skill_vae, flower.models.skill_vae 또는 실행 경로에 두세요."
            )
        vae_kwargs = dict(fallback_vae_kwargs)
        if skill_vae_config is not None:
            override = OmegaConf.to_container(skill_vae_config, resolve=True) if isinstance(skill_vae_config, DictConfig) else dict(skill_vae_config)
            vae_kwargs.update(override)
        autoencoder = SkillVAE(**vae_kwargs)
        self._load_skill_vae_state_by_suffix(autoencoder, autoencoder_state, ckpt_path, strict=strict_load)
        return autoencoder

    def _instantiate_skill_vae_with_target_fallback(self, autoencoder_cfg: Dict[str, Any], instantiate):
        """Instantiate SkillVAE cfg, falling back across local module names if needed."""
        targets = [str(autoencoder_cfg.get("_target_", ""))]
        if targets[0].startswith("flower.models.quest_skill_vae"):
            targets.extend([
                targets[0].replace("flower.models.quest_skill_vae", "flower.models.skill_vae"),
                targets[0].replace("flower.models.quest_skill_vae.", "skill_vae."),
            ])
        elif targets[0].startswith("quest."):
            cls_name = targets[0].split(".")[-1]
            targets.extend([
                f"flower.models.quest_skill_vae.{cls_name}",
                f"flower.models.skill_vae.{cls_name}",
                f"skill_vae.{cls_name}",
            ])

        last_exc = None
        for target in dict.fromkeys(targets):
            if not target:
                continue
            cfg = dict(autoencoder_cfg)
            cfg["_target_"] = target
            try:
                return instantiate(OmegaConf.create(cfg))
            except Exception as exc:
                last_exc = exc
                # MODIFIED: 업로드한 skill_vae.py처럼 ft_project_condition 인자를 받지 않는 복사본도 지원합니다.
                if "ft_project_condition" in cfg:
                    cfg_without_ft_project = dict(cfg)
                    cfg_without_ft_project.pop("ft_project_condition", None)
                    try:
                        return instantiate(OmegaConf.create(cfg_without_ft_project))
                    except Exception as exc2:
                        last_exc = exc2
        raise ImportError(f"SkillVAE instantiate failed for targets={targets}") from last_exc

    def _load_skill_vae_state_by_suffix(
        self,
        autoencoder: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        ckpt_path: str,
        strict: bool = False,
    ):
        """Fallback loader for checkpoints without QueST cfg/config."""
        model_state = autoencoder.state_dict()
        filtered_state = {}
        shape_mismatch = []

        for raw_key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            parts = raw_key.split(".")
            candidates = [".".join(parts[i:]) for i in range(len(parts))]
            matched_key = next((cand for cand in candidates if cand in model_state), None)
            if matched_key is None:
                continue
            if model_state[matched_key].shape != value.shape:
                shape_mismatch.append((matched_key, tuple(value.shape), tuple(model_state[matched_key].shape)))
                continue
            filtered_state[matched_key] = value

        if len(filtered_state) == 0:
            raise ValueError(f"SkillVAE checkpoint에서 현재 SkillVAE 구조와 매칭되는 weight가 없습니다: {ckpt_path}")

        missing, unexpected = autoencoder.load_state_dict(filtered_state, strict=False)
        print(
            f"Loaded frozen SkillVAE from {ckpt_path}: "
            f"matched={len(filtered_state)}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(shape_mismatch)}"
        )
        if strict and (missing or unexpected or shape_mismatch):
            raise RuntimeError(
                "SkillVAE strict load failed. "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatch={shape_mismatch[:20]}"
            )
        return missing, unexpected, shape_mismatch

    def _freeze_skill_vae(self):
        """Freeze SkillVAE parameters and keep it in eval mode."""
        if self.skill_vae is None:
            return
        # MODIFIED: SkillVAE는 action history -> quantized code extractor로만 사용하므로 학습하지 않습니다.
        self.skill_vae.eval()
        for param in self.skill_vae.parameters():
            param.requires_grad = False

    def _normalize_skill_action_history(
        self,
        history: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        return_valid_mask: bool = False,
    ):
        """Return [B, skill_block_size, action_dim] with left padding or last-window crop.

        MODIFIED:
        - episode 초반 history padding은 단순 zero action이 아니라
        "no motion + gripper open" action으로 둡니다.
        - 즉 padding action은 [0, 0, 0, 0, 0, 0, +1]입니다.
        - valid_mask는 그대로 padding 위치 False, 실제 action 위치 True입니다.
        """
        if history is None:
            if not self.skill_vae_allow_zero_history:
                raise KeyError(
                    "SkillVAE/History AdaLN을 사용하려면 batch에 'skill_prev_actions'가 필요합니다. "
                    "형상은 [B, 32, 7]이어야 하며, episode 시작부는 왼쪽 padding으로 채우세요."
                )

            history = torch.zeros(
                batch_size,
                self.skill_vae_skill_block_size,
                self.skill_vae_action_dim,
                device=device,
                dtype=dtype,
            )

            # MODIFIED:
            # fallback으로 history가 없을 때도 padding 정책을 학습/추론과 맞춥니다.
            # gripper action dim은 마지막 차원이라고 가정합니다.
            history[..., self.skill_vae_action_dim - 1] = 1.0

        if not torch.is_tensor(history):
            history = torch.as_tensor(history)
        if history.dim() == 4 and history.size(1) == 1:
            history = history.squeeze(1)
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3:
            raise ValueError(f"skill_prev_actions는 [B,T,A]여야 합니다. got={tuple(history.shape)}")
        if history.size(-1) != self.skill_vae_action_dim:
            raise ValueError(
                f"skill_prev_actions 마지막 차원은 {self.skill_vae_action_dim}이어야 합니다. got={history.size(-1)}"
            )
        if history.size(0) == 1 and batch_size > 1:
            history = history.expand(batch_size, -1, -1)
        elif history.size(0) != batch_size:
            raise ValueError(f"skill_prev_actions batch mismatch: got={history.size(0)}, expected={batch_size}")

        history = history.to(device=device, dtype=dtype)
        T = history.size(1)
        block = self.skill_vae_skill_block_size
        valid_mask = torch.ones(batch_size, T, device=device, dtype=torch.bool)

        if T < block:
            # MODIFIED:
            # timestep 12이면 앞쪽 20개를 padding으로 채웁니다.
            # padding action은 [0,0,0,0,0,0,+1]입니다.
            pad = torch.zeros(
                batch_size,
                block - T,
                self.skill_vae_action_dim,
                device=device,
                dtype=dtype,
            )

            # MODIFIED:
            # padding 구간의 gripper action은 open(+1)로 둡니다.
            # 실제 과거 action 구간은 아래 torch.cat의 history 쪽이므로 건드리지 않습니다.
            pad[..., self.skill_vae_action_dim - 1] = 1.0

            history = torch.cat([pad, history], dim=1)

            pad_mask = torch.zeros(batch_size, block - T, device=device, dtype=torch.bool)
            valid_mask = torch.cat([pad_mask, valid_mask], dim=1)

        elif T > block:
            # MODIFIED: 추론 중 32개가 넘으면 최근 32개 action만 History AdaLN 입력으로 사용합니다.
            history = history[:, -block:, :]
            valid_mask = valid_mask[:, -block:]

        history = history.contiguous()
        valid_mask = valid_mask.contiguous()
        if return_valid_mask:
            return history, valid_mask
        return history

    def _normalize_skill_gripper_state_history(
        self,
        gripper_states: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Normalize gripper states to [B, skill_block_size].

        MODIFIED: 이 state는 SkillVAE 입력 action relabeling에만 사용됩니다.
        Flow Transformer의 proprio/action target에는 영향을 주지 않습니다.
        """
        if gripper_states is None:
            if self.skill_vae_relabel_require_state:
                raise KeyError(
                    "skill_vae_relabel_gripper=True이고 skill_vae_relabel_require_state=True이지만 "
                    "batch에 SkillVAE history와 정렬된 gripper state가 없습니다. "
                    "'skill_prev_gripper_states'([B,T]) 또는 'skill_prev_states'([B,T,D])를 넘기세요."
                )
            return None

        if not torch.is_tensor(gripper_states):
            gripper_states = torch.as_tensor(gripper_states)
        if gripper_states.dim() == 4 and gripper_states.size(1) == 1:
            gripper_states = gripper_states.squeeze(1)
        if gripper_states.dim() == 3:
            state_dim = gripper_states.size(-1)
            idx = self.skill_vae_gripper_state_idx
            if idx < 0:
                idx = state_dim + idx
            if idx < 0 or idx >= state_dim:
                raise ValueError(f"gripper_state_idx={self.skill_vae_gripper_state_idx} is out of range for state_dim={state_dim}")
            gripper_states = gripper_states[..., idx]
        elif gripper_states.dim() == 1:
            gripper_states = gripper_states.unsqueeze(0)
        elif gripper_states.dim() != 2:
            raise ValueError(f"gripper state history는 [B,T] 또는 [B,T,D]여야 합니다. got={tuple(gripper_states.shape)}")

        if gripper_states.size(0) == 1 and batch_size > 1:
            gripper_states = gripper_states.expand(batch_size, -1)
        elif gripper_states.size(0) != batch_size:
            raise ValueError(f"gripper state batch mismatch: got={gripper_states.size(0)}, expected={batch_size}")

        gripper_states = gripper_states.to(device=device, dtype=dtype)
        T = gripper_states.size(1)
        block = self.skill_vae_skill_block_size
        if T < block:
            # MODIFIED: padding 위치는 valid_mask=False라 relabel에 쓰이지 않습니다. 값 자체는 0으로 둡니다.
            pad = torch.zeros(batch_size, block - T, device=device, dtype=dtype)
            gripper_states = torch.cat([pad, gripper_states], dim=1)
        elif T > block:
            gripper_states = gripper_states[:, -block:]
        return gripper_states.contiguous()

    def _relabel_skill_vae_gripper_actions(
        self,
        history: torch.Tensor,
        gripper_states: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply gripper-action relabeling only to the SkillVAE input history.

        Equivalent to:
            abs(gripper_action) < 0.9 -> (gripper_state > 0.95) * 2 - 1

        MODIFIED: 이 함수는 clone()을 만든 뒤 gripper 차원만 바꾸므로 batch['actions']나
        Flow Transformer가 denoise하는 action target은 바뀌지 않습니다.
        """
        if not self.skill_vae_relabel_gripper:
            return history
        if gripper_states is None:
            if self.skill_vae_relabel_require_state:
                raise KeyError("SkillVAE gripper relabeling을 위한 gripper state history가 없습니다.")
            return history

        action_dim = history.size(-1)
        action_idx = self.skill_vae_gripper_action_idx
        if action_idx < 0:
            action_idx = action_dim + action_idx
        if action_idx < 0 or action_idx >= action_dim:
            raise ValueError(f"gripper_action_idx={self.skill_vae_gripper_action_idx} is out of range for action_dim={action_dim}")

        if gripper_states.shape != history.shape[:2]:
            raise ValueError(
                "gripper state history shape mismatch: "
                f"states={tuple(gripper_states.shape)}, history={tuple(history.shape)}"
            )

        relabeled = history.clone()
        gripper_action = relabeled[..., action_idx]
        replace_mask = gripper_action.abs() < self.skill_vae_relabel_action_threshold
        if valid_mask is not None:
            replace_mask = replace_mask & valid_mask

        state_label = (gripper_states > self.skill_vae_relabel_state_threshold).to(relabeled.dtype) * 2.0 - 1.0
        relabeled[..., action_idx] = torch.where(replace_mask, state_label, gripper_action)
        return relabeled

    def _get_skill_gripper_states_from_batch(self, batch: Dict) -> Optional[torch.Tensor]:
        """Fetch action-window-aligned gripper state history from dataloader batch."""
        # MODIFIED: [B,T]로 이미 추출된 gripper state를 가장 우선 사용합니다.
        for key in ["skill_prev_gripper_states", "prev_gripper_states", "past_gripper_states", "gripper_state_history"]:
            if key in batch:
                return batch[key]

        # MODIFIED: [B,T,D] state history에서 skill_vae_gripper_state_idx 차원을 추출합니다.
        for key in ["skill_prev_states", "skill_prev_robot_obs", "prev_states", "prev_robot_obs", "past_states", "past_robot_obs"]:
            if key in batch:
                return batch[key]

        # robot_obs는 현재 observation만 [B,D]로 들어오는 경우가 많아 history relabeling에는 쓰지 않습니다.
        # dataloader에서 SkillVAE history와 정렬된 state를 별도 key로 넘기는 것을 권장합니다.
        return None

    def _get_skill_prev_actions_from_batch(self, batch: Dict, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Fetch action history from the dataloader batch and apply SkillVAE-only relabeling."""
        # MODIFIED: dataloader/preprocess에서 이 key 중 하나로 과거 action history를 넘겨야 합니다.
        raw_history = None
        for key in ["skill_prev_actions", "prev_actions", "past_actions", "action_history"]:
            if key in batch:
                raw_history = batch[key]
                break

        history, valid_mask = self._normalize_skill_action_history(raw_history, batch_size, device, dtype, return_valid_mask=True)

        # MODIFIED: raw_action AdaLN baseline은 사용자가 요청한 대로 gripper relabeling을 적용하지 않습니다.
        # 즉, dataloader가 제공한 원본 action history [B,32,7]을 그대로 직접 AdaLN condition으로 사용합니다.
        if self.history_adaln_source == "raw_action":
            return history

        gripper_states = self._normalize_skill_gripper_state_history(
            self._get_skill_gripper_states_from_batch(batch),
            batch_size,
            device,
            dtype,
        )
        # MODIFIED: relabeling은 여기서 만든 SkillVAE 입력 복사본에만 적용됩니다.
        # dataset_batch['actions']는 rf_loss()로 그대로 전달되므로 Flow Transformer target은 원본입니다.
        return self._relabel_skill_vae_gripper_actions(history, gripper_states, valid_mask)

    def _attach_skill_history_from_batch(self, cond: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        """Attach [B,32,7] previous-action history to DiT condition dict during train/val."""
        if not self.use_skill_vae_adaln:
            return cond
        batch_size = cond["features"].size(0)
        device = cond["features"].device
        dtype = next(self.parameters()).dtype
        cond["skill_prev_actions"] = self._get_skill_prev_actions_from_batch(batch, batch_size, device, dtype)
        return cond

    def encode_skill_history(
        self,
        skill_prev_actions: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode previous actions into a DiT AdaLN condition vector.

        MODIFIED:
        - history_adaln_source="skill_vae": frozen SkillVAE encode/quantize code를 사용합니다.
        - history_adaln_source="raw_action": [B,32,7] action history를 relabeling 없이 직접 MLP로 projection합니다.
        - use_skill_vae_adaln_conv1d=True: SkillVAE quantized code sequence를 Conv1d로 projection합니다.
        """
        if not self.use_skill_vae_adaln:
            return torch.zeros(batch_size, self.dit_dim, device=device, dtype=dtype)

        history = self._normalize_skill_action_history(
            skill_prev_actions,
            batch_size,
            device,
            dtype,
        )

        # MODIFIED: raw action history baseline.
        # SkillVAE를 거치지 않고 원본 32-step action history를 flatten하여 AdaLN condition으로 사용합니다.
        if self.history_adaln_source == "raw_action":
            if self.raw_action_history_proj is None:
                raise RuntimeError("history_adaln_source='raw_action'이지만 raw_action_history_proj가 초기화되지 않았습니다.")
            flat_history = history.reshape(batch_size, -1)
            proj_dtype = next(self.raw_action_history_proj.parameters()).dtype
            history_embed = self.raw_action_history_proj(flat_history.to(dtype=proj_dtype))
            return history_embed.to(device=device, dtype=dtype)

        if self.history_adaln_source != "skill_vae":
            raise ValueError(f"Unknown history_adaln_source: {self.history_adaln_source}")
        if self.skill_vae is None:
            raise RuntimeError("history_adaln_source='skill_vae'이지만 SkillVAE가 초기화되지 않았습니다.")
        if self.use_skill_vae_adaln_conv1d:
            if self.skill_vae_conv1d_encoder is None:
                raise RuntimeError(
                    "use_skill_vae_adaln_conv1d=True이지만 skill_vae_conv1d_encoder가 초기화되지 않았습니다."
                )
        elif self.skill_vae_proj is None:
            raise RuntimeError(
                "use_skill_vae_adaln_conv1d=False이지만 skill_vae_proj가 초기화되지 않았습니다."
            )

        # MODIFIED: batch path에서는 _attach_skill_history_from_batch()에서 SkillVAE용으로만 relabel된 tensor가 들어옵니다.
        # inference path에서는 _get_inference_skill_history()가 rolling state를 이용해 SkillVAE용으로만 relabel된 tensor를 반환합니다.
        skill_param = next(self.skill_vae.parameters())
        skill_dtype = skill_param.dtype
        history = history.to(device=skill_param.device, dtype=skill_dtype)

        # MODIFIED: SkillVAE는 frozen이므로 no_grad + eval로 dropout/grad를 차단합니다.
        self.skill_vae.eval()
        with torch.no_grad():
            z = self.skill_vae.encode(history)
            codes, _, _, _, _ = self.skill_vae.quantize(z)

        if self.use_skill_vae_adaln_conv1d:
            if self.skill_vae_conv1d_encoder is None:
                raise RuntimeError(
                    "use_skill_vae_adaln_conv1d=True이지만 skill_vae_conv1d_encoder가 초기화되지 않았습니다."
                )

            enc_param = next(self.skill_vae_conv1d_encoder.parameters())
            skill_embed = self.skill_vae_conv1d_encoder(
                codes.to(device=enc_param.device, dtype=enc_param.dtype)
            )
            return skill_embed.to(device=device, dtype=dtype)

        if self.skill_vae_proj is None:
            raise RuntimeError(
                "use_skill_vae_adaln_conv1d=False이지만 skill_vae_proj가 초기화되지 않았습니다."
            )

        if self.skill_vae_cond_pool == "last":
            pooled = codes[:, -1]
        else:
            pooled = codes.mean(dim=1)

        proj_dtype = next(self.skill_vae_proj.parameters()).dtype
        skill_embed = self.skill_vae_proj(pooled.to(dtype=proj_dtype))
        return skill_embed.to(device=device, dtype=dtype)

    def _get_inference_skill_gripper_history(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return rolling gripper state buffer for inference relabeling."""
        if not self.skill_vae_relabel_gripper:
            return None
        if self.skill_gripper_state_history is None or self.skill_gripper_state_history.size(0) != batch_size:
            self.skill_gripper_state_history = torch.zeros(
                batch_size,
                self.skill_vae_skill_block_size,
                device=device,
                dtype=dtype,
            )
        else:
            self.skill_gripper_state_history = self.skill_gripper_state_history.to(device=device, dtype=dtype)
        return self.skill_gripper_state_history

    def _get_inference_skill_history(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return rolling previous-action buffer for inference after SkillVAE-only relabeling."""
        if self.skill_action_history is None or self.skill_action_history.size(0) != batch_size:
            # MODIFIED: rollout 시작 시 SkillVAE 입력은 [0]*32입니다.
            self.skill_action_history = torch.zeros(
                batch_size,
                self.skill_vae_skill_block_size,
                self.skill_vae_action_dim,
                device=device,
                dtype=dtype,
            )
            # MODIFIED: zero padding이 아니라 gripper-open padding을 사용합니다.
            self.skill_action_history[..., self.skill_vae_action_dim - 1] = 1.0
        else:
            self.skill_action_history = self.skill_action_history.to(device=device, dtype=dtype)

        # MODIFIED: raw_action baseline은 inference에서도 gripper relabeling 없이 원본 rolling action history를 사용합니다.
        if self.history_adaln_source == "raw_action":
            return self.skill_action_history

        valid_mask = torch.ones(batch_size, self.skill_vae_skill_block_size, device=device, dtype=torch.bool)
        gripper_states = self._get_inference_skill_gripper_history(batch_size, device, dtype)
        # MODIFIED: 반환 직전 clone/relabel하므로 rolling action buffer 원본은 그대로 유지됩니다.
        return self._relabel_skill_vae_gripper_actions(self.skill_action_history, gripper_states, valid_mask)

    def _extract_current_gripper_state_from_obs(self, obs: Dict, action: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract current gripper state from inference obs for the rolling relabel buffer."""
        if not self.skill_vae_relabel_gripper:
            return None
        state = None
        if "robot_obs_raw" in obs:
            state = obs["robot_obs_raw"]
        elif "robot_obs" in obs:
            state = obs["robot_obs"]
        if state is None:
            if self.skill_vae_relabel_require_state:
                raise KeyError("Inference obs에 'robot_obs' 또는 'robot_obs_raw'가 없어 SkillVAE gripper relabeling을 할 수 없습니다.")
            return None
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, device=action.device)
        else:
            state = state.to(device=action.device)
        state = state.to(dtype=action.dtype)

        # Supported: [D], [B,D], [B,T,D]. Use the latest state when a time dimension exists.
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state[:, -1, :]
        elif state.dim() != 2:
            raise ValueError(f"Inference robot_obs shape should be [D], [B,D], or [B,T,D], got={tuple(state.shape)}")

        state_dim = state.size(-1)
        idx = self.skill_vae_gripper_state_idx
        if idx < 0:
            idx = state_dim + idx
        if idx < 0 or idx >= state_dim:
            raise ValueError(f"gripper_state_idx={self.skill_vae_gripper_state_idx} is out of range for state_dim={state_dim}")
        gripper_state = state[:, idx]

        if gripper_state.size(0) == 1 and action.size(0) > 1:
            gripper_state = gripper_state.expand(action.size(0))
        elif gripper_state.size(0) != action.size(0):
            raise ValueError(f"gripper state batch mismatch: got={gripper_state.size(0)}, expected={action.size(0)}")
        return gripper_state

    def _append_skill_action_history(self, action: torch.Tensor, gripper_state: Optional[torch.Tensor] = None):
        """Append executed action and optional gripper state to the rolling SkillVAE history buffer."""
        if not self.use_skill_vae_adaln:
            return
        action = action.detach()
        if action.dim() == 1:
            action = action.unsqueeze(0)
        elif action.dim() == 3:
            action = action[:, -1, :]
        if action.dim() != 2:
            raise ValueError(f"history update action은 [B,A] 또는 [B,1,A]여야 합니다. got={tuple(action.shape)}")
        if action.size(-1) != self.skill_vae_action_dim:
            raise ValueError(f"history update action dim mismatch: got={action.size(-1)}")

        B, A = action.shape
        if self.skill_action_history is None or self.skill_action_history.size(0) != B:
            self.skill_action_history = torch.zeros(
                B,
                self.skill_vae_skill_block_size,
                A,
                device=action.device,
                dtype=action.dtype,
            )
        else:
            self.skill_action_history = self.skill_action_history.to(device=action.device, dtype=action.dtype)

        # MODIFIED: 가장 오래된 action을 버리고 새로 실행한 action을 마지막에 추가합니다.
        # 여기서는 원본 action을 저장하고, SkillVAE 입력을 만들 때만 clone/relabel합니다.
        self.skill_action_history = torch.cat(
            [self.skill_action_history[:, 1:, :], action.unsqueeze(1)],
            dim=1,
        ).detach()

        if self.skill_vae_relabel_gripper:
            if gripper_state is None:
                if self.skill_vae_relabel_require_state:
                    raise KeyError("history update에 사용할 gripper_state가 없습니다.")
                return
            if not torch.is_tensor(gripper_state):
                gripper_state = torch.as_tensor(gripper_state, device=action.device)
            gripper_state = gripper_state.to(device=action.device, dtype=action.dtype).detach()
            if gripper_state.dim() == 0:
                gripper_state = gripper_state.view(1)
            if gripper_state.dim() == 2 and gripper_state.size(-1) == 1:
                gripper_state = gripper_state.squeeze(-1)
            if gripper_state.dim() != 1:
                raise ValueError(f"gripper_state는 [B]여야 합니다. got={tuple(gripper_state.shape)}")
            if gripper_state.size(0) == 1 and B > 1:
                gripper_state = gripper_state.expand(B)
            elif gripper_state.size(0) != B:
                raise ValueError(f"gripper_state batch mismatch: got={gripper_state.size(0)}, expected={B}")

            if self.skill_gripper_state_history is None or self.skill_gripper_state_history.size(0) != B:
                self.skill_gripper_state_history = torch.zeros(
                    B,
                    self.skill_vae_skill_block_size,
                    device=action.device,
                    dtype=action.dtype,
                )
            else:
                self.skill_gripper_state_history = self.skill_gripper_state_history.to(device=action.device, dtype=action.dtype)

            # MODIFIED: action history와 같은 timestep에 해당하는 gripper state를 함께 저장합니다.
            self.skill_gripper_state_history = torch.cat(
                [self.skill_gripper_state_history[:, 1:], gripper_state.unsqueeze(1)],
                dim=1,
            ).detach()

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Get parameter groups
        optim_groups = self._get_param_groups()

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.optimizer_config.learning_rate,
                betas=self.optimizer_config.betas
            )

        # Initialize scheduler
        scheduler = TriStageLRScheduler(
            optimizer,
            OmegaConf.create(self.lr_scheduler_config)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _get_param_groups(self):
        """Get parameter groups for optimizer"""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []

        # Collect all parameters, excluding VLM if frozen
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step"""
        # Get optimizer
        opt = self.optimizers()
        
        # Compute loss
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device) 
        bbox_aux_loss = torch.tensor(0.0, device=self.device)
        optimization_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)
            # MODIFIED: 학습 batch의 과거 32개 action을 AdaLN history condition으로 연결합니다.
            # history_adaln_source='skill_vae'이면 SkillVAE 입력에만 relabeling이 적용되고,
            # history_adaln_source='raw_action'이면 relabeling 없이 원본 action history를 직접 사용합니다.
            # dataset_batch["actions"]는 Flow Transformer RF target으로 그대로 유지합니다.
            obs_features = self._attach_skill_history_from_batch(obs_features, dataset_batch)

            act_loss, losses_dict = self.rf_loss(obs_features, dataset_batch["actions"])
            loss = act_loss
            if self.use_bbox_loss:
                bbox_loss = self.compute_bbox_aux_loss(obs_features, dataset_batch[self.bbox_key])
                bbox_aux_loss = bbox_aux_loss + bbox_loss
                loss = loss + self.bbox_loss_weight * bbox_loss

            action_loss = action_loss + act_loss
            total_loss = total_loss + loss
            optimization_loss = optimization_loss + loss
            total_bs = total_bs + len(dataset_batch["actions"])

        total_loss = total_loss / len(batch)
        if self.use_bbox_loss:
            bbox_aux_loss = bbox_aux_loss / len(batch)

        # Log metrics
        self._log_training_metrics(
            total_loss,
            action_loss,
            total_bs,
            bbox_loss=bbox_aux_loss if self.use_bbox_loss else None,
        )

        # Optimization step
        # opt.zero_grad()
        # self.manual_backward(action_loss)
        
        # Clip gradients
         #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Step optimizer
         #opt.step()

        # Update learning rate
         #sch = self.lr_schedulers()
         #if sch is not None:
        #     sch.step()

        return optimization_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step"""
        output = {}
        with torch.no_grad():
            obs_features = self.encode_observations(batch)
            # MODIFIED: validation도 train과 동일한 SkillVAE history condition을 사용합니다.
            obs_features = self._attach_skill_history_from_batch(obs_features, batch)
            target_actions = batch[self.target_modality]
            
            # Generate noise for sampling
            noise_actions = torch.randn_like(target_actions, device=self.device)

            # Sample actions
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            
            # Compute validation loss
            val_loss = F.mse_loss(action_pred, target_actions)
            
            # Log metrics
            self._log_validation_metrics(val_loss, val_loss)
            
            output["validation_loss"] = val_loss / len(batch)
            return output
            
    def rf_loss(self, cond, actions, dataset_idx=None):
        """
        Compute the rectified flow loss.
        """
        default_dtype = next(self.parameters()).dtype
        
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time based on sampling strategy
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)

        # Interpolate
        zt = (1 - texp) * actions + texp * z1

        # Forward pass
        vtheta = self.dit_forward(zt, t, cond)
        # Compute loss on valid dimensions only
        diff = (z1 - actions) - vtheta
        valid_diff = diff
        loss = (valid_diff ** 2).mean()

        # Collect debugging info
        losses_dict = {
            "diff_min": valid_diff.min().item(),
            "diff_max": valid_diff.max().item(),
            "diff_mean": valid_diff.mean().item(),
            "loss": loss.item(),
        }

        return loss, losses_dict

    def compute_bbox_aux_loss(self, obs_features: Dict[str, torch.Tensor], target_bbox: torch.Tensor) -> torch.Tensor:
        """Compute YOLOv1-style cxcywh bbox regression loss on action-conditioning features."""
        if self.bbox_head is None:
            raise RuntimeError("bbox_loss=True is required before computing bbox auxiliary loss.")

        features = obs_features["features"]
        pooled_features = features.mean(dim=1)
        pred_bbox = torch.sigmoid(self.bbox_head(pooled_features))

        target_bbox = target_bbox.to(device=pred_bbox.device, dtype=pred_bbox.dtype)
        if target_bbox.dim() == 3:
            target_bbox = target_bbox[:, -1]
        if target_bbox.shape != pred_bbox.shape:
            raise ValueError(f"bbox target shape should be {tuple(pred_bbox.shape)}, got {tuple(target_bbox.shape)}")

        eps = 1e-6
        xy_loss = (target_bbox[:, :2] - pred_bbox[:, :2]).pow(2).sum(dim=-1)
        wh_loss = (
            torch.sqrt(target_bbox[:, 2:].clamp_min(eps))
            - torch.sqrt(pred_bbox[:, 2:].clamp_min(eps))
        ).pow(2).sum(dim=-1)

        return (xy_loss + wh_loss).mean()

    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool=False):
        """
        Sample actions using Euler method.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        # Integration
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1]*(z.dim()-1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Predict velocity field
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def _normalize_force_history(
        self,
        force_history: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return right_force_history as [B,T,force_history_dim]."""
        if force_history is None:
            raise KeyError(
                "use_force_history_adaln=True이면 batch에 'right_force_history'가 필요합니다. "
                "각 sample shape은 [T,6] 또는 [obs_seq_len,T,6]이어야 합니다."
            )

        if not torch.is_tensor(force_history):
            force_history = torch.as_tensor(force_history)

        if force_history.dim() == 4:
            # Dataset path is usually [B, obs_seq_len, history_len, 6].
            force_history = force_history[:, -1]
        elif force_history.dim() == 2:
            force_history = force_history.unsqueeze(0)
        elif force_history.dim() != 3:
            raise ValueError(f"right_force_history should be [B,T,D] or [B,obs,T,D], got {tuple(force_history.shape)}")

        if force_history.size(0) == 1 and batch_size > 1:
            force_history = force_history.expand(batch_size, -1, -1)
        elif force_history.size(0) != batch_size:
            raise ValueError(f"force history batch mismatch: got={force_history.size(0)}, expected={batch_size}")

        if force_history.size(-1) != self.force_history_dim:
            raise ValueError(
                f"right_force_history last dim should be {self.force_history_dim}, got {force_history.size(-1)}"
            )

        return force_history.to(device=device, dtype=dtype).contiguous()

    def encode_force_history(
        self,
        force_history: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.use_force_history_adaln:
            return torch.zeros(batch_size, self.dit_dim, device=device, dtype=dtype)
        if self.force_history_encoder is None:
            raise RuntimeError("use_force_history_adaln=True이지만 force_history_encoder가 초기화되지 않았습니다.")

        history = self._normalize_force_history(force_history, batch_size, device, dtype)
        enc_dtype = next(self.force_history_encoder.parameters()).dtype
        force_embed = self.force_history_encoder(history.to(dtype=enc_dtype))
        return force_embed.to(device=device, dtype=dtype)

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        """
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape
        
        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)
        
        # Handle proprioception
        if self.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        # MODIFIED: frozen SkillVAE가 과거 32개 action을 quantize한 code를 AdaLN condition으로 제공합니다.
        if self.use_skill_vae_adaln:
            skill_embeds = self.encode_skill_history(
                cond_dict.get('skill_prev_actions'),
                batch_size=B,
                device=z.device,
                dtype=default_dtype,
            )
        else:
            skill_embeds = torch.zeros(B, self.dit_dim, device=z.device, dtype=default_dtype)

        if self.use_force_history_adaln and not self.round_robin_conditioning:
            force_embeds = self.encode_force_history(
                cond_dict.get("right_force_history"),
                batch_size=B,
                device=z.device,
                dtype=default_dtype,
            )
        else:
            force_embeds = torch.zeros(B, self.dit_dim, device=z.device, dtype=default_dtype)
        
        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)
        
        # Add positional encoding if not using ROPE/NOPE
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding
        
        # Process embeddings
        if self.round_robin_conditioning:
            cond = self.cond_linear(self.cond_norm(cond))
            base_t_emb = (
                stateless_norm(self.t_embedder(t))
                + stateless_norm(frequency_embeds).squeeze(1)
            )
            proprio_cond = base_t_emb + stateless_norm(proprio_embeds).squeeze(1)
            skill_cond = base_t_emb + stateless_norm(skill_embeds).squeeze(1)

            if self.use_adaln_cond:
                vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
                proprio_cond = vlm_token + proprio_cond
                skill_cond = vlm_token + skill_cond

            cx = z
            context = cond if self.use_cross_attn else None
            round_robin_conds = (proprio_cond, skill_cond)
            if not self.action_type_adaln:
                round_robin_adalns = tuple(self.adaln(layer_cond) for layer_cond in round_robin_conds)
            else:
                round_robin_adalns = tuple(
                    self.action_specific_adaln(layer_cond, action_type)
                    for layer_cond in round_robin_conds
                )

            for layer_idx, layer in enumerate(self.dit):
                cond_idx = layer_idx % 2
                cx = layer(
                    cx,
                    round_robin_conds[cond_idx],
                    context=context,
                    is_causal=True,
                    global_adaln=round_robin_adalns[cond_idx],
                )

            return self.decode_actions(cx, action_type, valid_dims)

        if self.use_proprio_skill_history_mlp_adaln:
            proprio_skill_embeds = self.encode_proprio_skill_history(
                proprio_embeds,
                skill_embeds,
                action_type,
                frequency_embeds.shape,
                force_embeds=force_embeds,
            )
            t_emb = (
                stateless_norm(self.t_embedder(t))
                + stateless_norm(frequency_embeds).squeeze(1)
                + stateless_norm(proprio_skill_embeds).squeeze(1)
            )
        else:
            t_emb = (
                stateless_norm(self.t_embedder(t))
                + stateless_norm(frequency_embeds).squeeze(1)
                + stateless_norm(proprio_embeds).squeeze(1)
                + stateless_norm(skill_embeds).squeeze(1)
                + stateless_norm(force_embeds).squeeze(1)
            )
        
        cond = self.cond_linear(self.cond_norm(cond))
        
        # Set up conditioning
        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb
        
        # Setup context
        cx = z
        context = cond if self.use_cross_attn else None
        
        # Get adaln signals
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)
        

        # Process through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx, 
                global_cond, 
                context=context, 
                is_causal=True, 
                global_adaln=global_adaln
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims)

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encode proprioception based on action type.
        """
        batch_size = output_shape[0]
        default_dtype = next(self.parameters()).dtype
        
        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)
        
        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)
                encoded_proprio[mask] = encoded.to(encoded_proprio.dtype)
        
        return encoded_proprio

    def encode_proprio_skill_history(
        self,
        proprio_embeds: torch.Tensor,
        skill_embeds: torch.Tensor,
        action_type: torch.Tensor,
        output_shape,
        force_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse proprio, SkillVAE/history, and optional force-history embeddings."""
        batch_size = output_shape[0]
        default_dtype = next(self.parameters()).dtype
        fused_embeds = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
        joint_parts = [proprio_embeds, skill_embeds]
        if self.use_force_history_adaln:
            if force_embeds is None:
                force_embeds = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
            joint_parts.append(force_embeds)
        joint_embeds = torch.cat(joint_parts, dim=-1)
        action_type = action_type.to(self.device)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.proprio_skill_history_fusion[action_name](joint_embeds[mask])
                fused_embeds[mask] = encoded.to(fused_embeds.dtype)

        return fused_embeds

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate action-specific AdaLN signals.
        """
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device
        
        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type) 
            for _ in range(num_chunks)
        ]
        
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal
        
        return mod_signals

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def encode_observations(self, batch: Dict) -> torch.Tensor:
        """Encode observations using Florence-2"""
        device = self.device
        default_type = next(self.parameters()).dtype
        
        
        batch_size = len(batch["rgb_obs"]['rgb_static'])
        embed_tensor = torch.zeros(batch_size, 1, 1)
        # Single-action-space setup: action type is one id per batch element.
        action_type_tensor = torch.ones(batch_size, dtype=torch.long)
        # Process primary image
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape
        
        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)
        
        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)
        
        # Get text embeddings
        # Get text embeddings once to reuse
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)
        
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)
        
        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        
        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 10 #여기서는 하드코딩으로 3으로 고정했었지만 우리 로봇은 10Hz로 10으로 하드 코딩함. 나중에 고쳐야함.
        )
        
        # Get proprioception if enabled
        proprio = None
        if self.use_proprio:
            if "robot_obs" in batch:
                proprio = batch["robot_obs"][:, -1].to(device).to(default_type)
            elif self.obs_modalities and self.obs_modalities in batch and 'proprio' in batch[self.obs_modalities]:
                proprio = batch[self.obs_modalities]['proprio'].to(device).to(default_type)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': action_type_tensor,
            'proprio': proprio,
            'right_force_history': batch.get("right_force_history", None),
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode actions using action-specific encoders."""
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=self.device).to(default_dtype)
        
        # Track valid dimensions per type
        valid_dims = torch.zeros_like(z).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)
        
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """Decode actions using action-specific decoders."""
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(batch_size, z.shape[1], max_action_dim, 
                        device=self.device).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                action_dim = self.action_space_index.get_action_dim(action_idx)
                
                pred = self.action_decoders[action_name](z)
                # Only assign to valid dimensions
                decoded = pred
        return decoded

    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Predicted action sequence
        """
        # batch = {'rgb_obs': obs, '"lang_text"': goal}
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        # Create batch for observation encoding
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }
        if self.use_proprio:
            if "robot_obs" in obs:
                batch["robot_obs"] = obs["robot_obs"]
            elif "robot_obs_raw" in obs:
                robot_obs_raw = obs["robot_obs_raw"]
                if robot_obs_raw.dim() == 1:
                    robot_obs_raw = robot_obs_raw.unsqueeze(0).unsqueeze(0)
                elif robot_obs_raw.dim() == 2:
                    robot_obs_raw = robot_obs_raw.unsqueeze(0)
                batch["robot_obs"] = robot_obs_raw
        if "right_force_history" in obs:
            batch["right_force_history"] = obs["right_force_history"]
        features = self.encode_observations(batch)
        # MODIFIED: 추론에서는 rolling buffer에 저장된 최근 32개 실행 action을 SkillVAE 입력으로 사용합니다.
        if self.use_skill_vae_adaln:
            features['skill_prev_actions'] = self._get_inference_skill_history(
                batch_size=len(features['features']),
                device=features['features'].device,
                dtype=next(self.parameters()).dtype,
            )
        
        # Generate initial noise
        noise = torch.randn(
            len(features['features']),
            self.act_window_size,
            self.action_dim,
            device=features['features'].device
        )
        
        # Sample actions
        return self.sample_actions(noise, features, inference=True)

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Do one step of inference, handling action chunking.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Current action prediction
        """
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        
        if not self.return_act_chunk:
            # Default: return current action
            # MODIFIED: history update를 위해 batch 전체의 현재 action을 보존합니다.
            current_action_for_history = self.pred_action_seq[:, self.rollout_step_counter]
            current_action = current_action_for_history[0]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            # Return whole chunk for ALOHA setups
            current_action_for_history = None
            current_action = self.pred_action_seq

        # MODIFIED: 실제 반환/실행된 action을 rolling SkillVAE history에 추가합니다.
        # gripper state는 SkillVAE 입력 relabeling에만 사용됩니다.
        if current_action_for_history is not None:
            # MODIFIED: raw_action baseline은 gripper relabeling을 쓰지 않으므로 state history를 추출하지 않습니다.
            gripper_state_for_history = None
            if self.history_adaln_source == "skill_vae":
                gripper_state_for_history = self._extract_current_gripper_state_from_obs(obs, current_action_for_history)
            self._append_skill_action_history(current_action_for_history, gripper_state_for_history)
            
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action

    def reset(self):
        """Reset model state for new rollout."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        # MODIFIED: 새 rollout은 zero action history 32개에서 다시 시작합니다.
        self.skill_action_history = None
        self.skill_gripper_state_history = None
        self.eval()

    def on_train_start(self):
        """Convert model to appropriate dtype on training start."""
        # Move core model components to appropriate device/dtype
        self.to(self.device)
        self.vlm.to(self.device)
        # MODIFIED: Lightning이 module mode를 바꿔도 SkillVAE는 계속 frozen/eval 상태로 유지합니다.
        self._freeze_skill_vae()
        
    def on_validation_start(self):
        """Setup before validation starts."""
        self.eval()

    def on_validation_end(self):
        """Cleanup after validation ends."""
        self.train()
        # MODIFIED: self.train() 호출 뒤에도 frozen SkillVAE encoder/quantizer는 eval로 되돌립니다.
        self._freeze_skill_vae()

    def print_model_parameters(self):
        """Print model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        
        for name, submodule in self.named_modules():
            if '.' not in name or name.count('.') <= 1:
                submodule_params = sum(p.numel() for p in submodule.parameters())
                if submodule_params > 0:
                    print(f"{name} - Total Params: {submodule_params}")
                    
    def print_encoded_texts(self, batch, device):
        """Print encoded text inputs for debugging."""
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(self.device)
        ).to(device).squeeze(1)
        
        input_ids = batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'][0].squeeze(0).to(self.device)
        input_ids = input_ids.cpu()
        decoded_text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        print("Original text:", decoded_text)

        decoded_texts = self.processor.tokenizer.batch_decode(text_embeds.cpu(), skip_special_tokens=True)
        print("Encoded texts:")
        for i, text in enumerate(decoded_texts):
            print(f"Sequence {i+1}: {text}")
    
    def construct_prompts(self, dataset_batch):
        """
        Constructs prompts for Florence-2's encoder to extract task-relevant visual features.
        
        Args:
            dataset_batch: Dictionary containing task information including language instructions
            
        Returns:
            text_prompts: List of formatted prompts for encoder conditioning
        """
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                # Original instruction only
                text_prompts.append(self.format_instruction(instruction))
                
            elif self.vlm_prompt_style == "feature_focused":
                # Focus on extracting visual features relevant for manipulation
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
                
            elif self.vlm_prompt_style == "state_oriented":
                # Focus on extracting state-relevant features
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
                
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        
        return text_prompts
    
    def _get_text_embeddings(self, text, device):
        """Get text embeddings to use with VLM"""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])
    
    def _log_training_metrics(self, total_loss, action_loss, total_bs, bbox_loss=None):
        """
        Log training metrics
        Args:
            total_loss: Total loss value
            action_loss: Action-specific loss value
            total_bs: Total batch size
            bbox_loss: Optional auxiliary bbox loss value
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        if bbox_loss is not None:
            self.log("train/bbox_loss", bbox_loss, on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=total_bs)
        
    def _log_validation_metrics(self, pred_loss, val_total_act_loss_pp):
        """
        Log validation metrics
        Args:
            pred_loss: Prediction loss value (scalar)
            val_total_act_loss_pp: Total validation action loss per prediction
        """
        # Log per-modality action loss
        self.log(
            f"val_act/{self.modality_scope}_act_loss_pp", 
            pred_loss, 
            sync_dist=True
        )
        
        # Log average action loss across modalities
        try:
            n_modalities = len(self.trainer.datamodule.modalities)
        except AttributeError:
            n_modalities = 1  # Default if modalities not available
            
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / n_modalities,
            sync_dist=True
        )
