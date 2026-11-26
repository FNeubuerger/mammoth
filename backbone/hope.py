import math

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from backbone import MammothBackbone, register_backbone

# --- Titan Memory ---

class TitanMemory(nn.Module):
    """
    Simplified TITAN-style associative memory.
    Local teaching updates are supervised training operations, not test-time adaptation.
    """
    def __init__(self, dim: int, hidden_multiplier: int = 4, layers: int = 2):
        super().__init__()
        self.dim = dim
        hidden = dim * hidden_multiplier
        
        blocks = []
        for layer_idx in range(layers - 1):
            in_dim = dim if layer_idx == 0 else hidden
            blocks.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU()
            ])
        
        # Final layer
        in_dim = hidden if layers > 1 else dim
        blocks.append(nn.Linear(in_dim, dim))
        
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(dim)
        self.grad_clip = 1.0
        self.local_updates = 0

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # Query: [Batch, Seq, Dim]
        out = self.net(query)
        return self.norm(out)

    def update(self, key, error_signal, lr, momentum=0.0):
        """Apply a local gradient without accumulating into the supervised optimizer's .grad."""
        params = tuple(self.net.parameters())
        with torch.enable_grad():
            pred = self.net(key.detach())
            grads = torch.autograd.grad(pred, params, grad_outputs=(0.1 * error_signal).expand_as(pred))
        with torch.no_grad():
            norm = torch.stack([g.norm() for g in grads]).norm()
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite TitanMemory local gradient norm ({norm.item()}); update aborted")
            scale = (self.grad_clip / (norm + 1e-6)).clamp(max=1)
            for param, grad in zip(params, grads):
                delta = grad * scale
                if momentum:
                    if not hasattr(param, 'momentum_buffer'):
                        param.momentum_buffer = torch.zeros_like(param)
                    delta = param.momentum_buffer.mul_(momentum).add_(delta)
                param.add_(delta, alpha=-lr)
            self.local_updates += 1

# --- CMS (Contextual Memory System) ---

class CMS(nn.Module):
    """
    Contextual Memory System with multiple timescales.
    """
    def __init__(self, dim: int, levels: List[Dict]):
        super().__init__()
        self.dim = dim
        self.levels = levels # List of dicts: {'name': 'fast', 'period': 1}, etc.
        
        self.blocks = nn.ModuleList()
        self.block_map = {}
        
        for level in levels:
            # Each level has a small MLP or memory block
            block = TitanMemory(dim, hidden_multiplier=2, layers=1)
            self.blocks.append(block)
            self.block_map[level['name']] = block
            
        self.step_counter = 0

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        # x: [B, S, D]
        current = x
        inputs = {}
        outputs = {}
        
        for i, level in enumerate(self.levels):
            name = level['name']
            block = self.blocks[i]
            
            inputs[name] = current
            out = block(current)
            outputs[name] = out
            
            # Residual connection
            current = current + out
            
        if return_intermediates:
            return current, inputs, outputs
        return current

    def maybe_update(self, inputs, teach_signal, lr, momentum):
        """Count training minibatches, including those below the surprise gate."""
        self.step_counter += 1
        if teach_signal is None:
            return
        for level, block in zip(self.levels, self.blocks):
            if self.step_counter % level['period'] == 0:
                block.update(inputs[level['name']], teach_signal, lr, momentum)

    def consolidate(self):
        """No task-boundary transfer: this variant only updates during training."""
        pass

# --- Self Modifier ---

class SelfModifier(nn.Module):
    def __init__(self, dim: int, hidden_multiplier: int = 2):
        super().__init__()
        hidden = dim * hidden_multiplier
        # Input: key, value, error_signal. All dim. -> 3*dim
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim) # Output delta
        )

    def forward(self, key: torch.Tensor, value: torch.Tensor, error_signal: torch.Tensor) -> torch.Tensor:
        # Concatenate along last dim
        inp = torch.cat([key, value, error_signal], dim=-1)
        return self.net(inp)

# --- HOPE Block ---

class HOPEBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, surprise_threshold: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.surprise_threshold = surprise_threshold
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.titan = TitanMemory(dim)
        
        # Define CMS levels
        cms_levels = [
            {'name': 'fast', 'period': 1},
            {'name': 'mid', 'period': 5},
            {'name': 'slow', 'period': 100} 
        ]
        self.cms = CMS(dim, cms_levels)
        
        self.self_mod = SelfModifier(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, teach_signal: Optional[torch.Tensor] = None,
                local_lr: float = 1e-3, local_momentum: float = 0.9) -> torch.Tensor:
        # x: [B, S, D]
        
        # 1. Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 2. Titan Memory
        mem_out = self.titan(x)
        combined = x + mem_out
        
        # 3. CMS
        cms_out, cms_inputs, _ = self.cms(combined, return_intermediates=True)
        out = self.norm2(cms_out)
        
        # 4. Updates (if teach_signal is present)
        if teach_signal is not None and self.training:
            # STABILITY FIX: Pre-normalize teach signal similar to Titan implementation
            # This prevents exploding updates to the memory modules
            with torch.no_grad():
                teach_norm = teach_signal.norm(dim=-1, keepdim=True)
                teach_clip = 1.0
                scale = torch.clamp(teach_norm / teach_clip, min=1.0)
                teach_signal = teach_signal / scale
            
            magnitude = teach_signal.norm()
            if not torch.isfinite(magnitude):
                raise FloatingPointError("Nonfinite HOPE teaching signal; update aborted")
            if magnitude >= self.surprise_threshold:
                self.titan.update(x, teach_signal, local_lr)
            else:
                teach_signal = None
            self.cms.maybe_update(cms_inputs, teach_signal, local_lr, local_momentum)
            
        return out

    def consolidate(self):
        self.cms.consolidate()

# --- HOPE Backbone ---

class HOPEBackbone(MammothBackbone):
    """Patch-based image classifier or window classifier with one token per timestep."""

    def __init__(self, input_size: int, input_channels: int, num_classes: int, dim: int = 128,
                 depth: int = 2, heads: int = 4, surprise_threshold: float = 1e-6, sequence_features: int = 0):
        super(HOPEBackbone, self).__init__()
        
        self.input_size = input_size # e.g. 32 for CIFAR
        self.patch_size = 4
        self.dim = dim
        self.num_classes = num_classes
        
        self.sequence_features = sequence_features
        if sequence_features:
            self.sensor_embed = nn.Linear(sequence_features, dim)
            frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2) / dim)
            self.register_buffer("position_frequencies", frequencies, persistent=False)
        else:
            num_patches = (input_size // self.patch_size) ** 2
            self.patch_embed = nn.Conv2d(input_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        
        # HOPE Blocks
        self.blocks = nn.ModuleList([
            HOPEBlock(dim, heads, surprise_threshold=surprise_threshold) for _ in range(depth)
        ])
        
        self.out_dim = dim # For Mammoth compatibility
        self.classifier = nn.Linear(dim, num_classes)

    def consolidate(self):
        """
        Trigger consolidation in all HOPE blocks.
        """
        for block in self.blocks:
             block.consolidate()

    def forward(self, x: torch.Tensor, returnt: str = 'out', teach_signal: Optional[torch.Tensor] = None,
                local_lr: float = 1e-3, local_momentum: float = 0.9) -> torch.Tensor:
        """Classify independent complete windows [B,T,C] or square images [B,C,H,W]."""
        if self.sequence_features:
            if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.sequence_features:
                raise ValueError(f"Expected [B,T,{self.sequence_features}] with T>0, got {tuple(x.shape)}")
            x = self.sensor_embed(x)
            angles = torch.arange(x.shape[1], device=x.device)[:, None] * self.position_frequencies
            positions = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :self.dim]
            x = x + positions.to(x.dtype)
            if teach_signal is not None:
                # The classifier's feature is a *mean* of T tokens.
                teach_signal = teach_signal / x.shape[1]
        else:
            x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        # Blocks
        for block in self.blocks:
            x = block(x, teach_signal=teach_signal, local_lr=local_lr, local_momentum=local_momentum)
            
        # Pooling (Mean)
        features = x.mean(dim=1) # [B, D]
        
        if returnt == 'features':
            return features
            
        logits = self.classifier(features)
        
        if returnt == 'out':
            return logits
        elif returnt == 'all':
            return logits, features
        return logits

@register_backbone("hope")
def hope(input_size: int = None, input_channels: int = None, num_classes: int = None, dim: int = 128,
         depth: int = 2, heads: int = 4, surprise_threshold: float = 1e-6, sequence_features: int = 0) -> MammothBackbone:
    """
    Instantiates the HOPE backbone.
    """
    if num_classes is None:
        raise ValueError("num_classes must be provided")
    if not sequence_features and (input_size is None or input_channels is None):
        raise ValueError("Provide input_size and input_channels for images, or sequence_features for windows")
    return HOPEBackbone(input_size, input_channels, num_classes, dim=dim, depth=depth, heads=heads,
                        surprise_threshold=surprise_threshold, sequence_features=sequence_features)
