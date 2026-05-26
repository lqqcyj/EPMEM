"""
Qwen-2.5-VL adaptation of EPMEM (Emotion Perception Memory Enhancement).

Implements:
  - Qwen2efvrMLP     : drop-in Qwen2MLP with EFVR injection state
  - Qwen2_5_VLForEPMEM : wraps Qwen2_5_VLForConditionalGeneration with
      * activate_efvr() / deactivate_efvr() control API
      * entropy-triggered EFVR via post-forward hooks (no model.forward override)
      * generate() that returns (output_ids, token_logits, attn_map)

Requires: transformers >= 4.49.0
"""

import warnings
import numpy as np
from typing import List, Optional, Set
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Qwen2.5-VL support requires transformers >= 4.49.0. "
        "Install with: pip install transformers>=4.49.0"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# EFVR-capable MLP  (replaces Qwen2MLP in every decoder layer)
# ─────────────────────────────────────────────────────────────────────────────

class Qwen2efvrMLP(nn.Module):
    """
    Qwen2 FFN block augmented with Emotion-Focused Visual Retracing (EFVR).
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size       = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]

        # ── 仅保留层级别的注入状态 (全局配置移至主模型) ──
        self.retracing_ratio   = 0.3
        self.adpt_sign         = 0
        self.register_parameter('adpt_w1', None)   # [kept, H]
        self.register_parameter('adpt_w2', None)   # [H, kept]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adpt_sign == 0:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        # ── adpt_sign == 1: 注入降噪视觉特征 ──
        ffn_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        if self.adpt_w1 is None or self.adpt_w2 is None:
            warnings.warn("[EFVR] adpt_sign=1 but adapter weights not initialised – skipping.")
            return ffn_out

        adapter_out  = torch.matmul(torch.matmul(x, self.adpt_w1.T), self.adpt_w2.T)
        ffn_norm     = torch.mean(torch.abs(ffn_out))
        adapter_norm = torch.mean(torch.abs(adapter_out)) + 1e-9
        norm_adapter = (ffn_norm / adapter_norm) * adapter_out

        return ffn_out * (1.0 - self.retracing_ratio) + norm_adapter * self.retracing_ratio


# ─────────────────────────────────────────────────────────────────────────────
# EPMEM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Qwen2_5_VLForEPMEM(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self._mlps_replaced  = False
        self._efvr_hooks: List = []
        self._cached_text_model = None
        
        # 【核心修复】：统一、可靠的全局状态管理
        self.efvr_state = {
            'active': False,
            'visual_token': None,
            'entropy_threshold': 0.7,
            'retracing_ratio': 0.3,
            'starting_layer': 0,
            'ending_layer': 20,
            'trigger_layers': None,
            'triggered_this_step': False,  # Step 级别防抖
        }

    @property
    def _text_model(self):
        if self._cached_text_model is not None:
            return self._cached_text_model

        if hasattr(self.model, 'layers') and isinstance(self.model.layers, torch.nn.ModuleList):
            self._cached_text_model = self.model
            return self.model

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') \
                and isinstance(self.model.model.layers, torch.nn.ModuleList):
            self._cached_text_model = self.model.model
            return self.model.model

        for _, child in self.model.named_children():
            if hasattr(child, 'layers') and isinstance(child.layers, torch.nn.ModuleList):
                self._cached_text_model = child
                return child

        raise AttributeError("Cannot find decoder 'layers' inside the model.")

    def _replace_mlps_with_efvr(self):
        for layer in self._text_model.layers:
            old = layer.mlp
            new = Qwen2efvrMLP(self.config.text_config)
            new.gate_proj.weight = old.gate_proj.weight
            new.up_proj.weight   = old.up_proj.weight
            new.down_proj.weight = old.down_proj.weight
            if old.gate_proj.bias is not None: new.gate_proj.bias = old.gate_proj.bias
            if old.up_proj.bias is not None:   new.up_proj.bias = old.up_proj.bias
            if old.down_proj.bias is not None: new.down_proj.bias = old.down_proj.bias
            layer.mlp = new
        self._mlps_replaced = True

    def activate_efvr(
        self,
        entropy_threshold: float = 0.7,
        retracing_ratio:   float = 0.3,
        starting_layer:    int   = 0,
        ending_layer:      int   = 20,
        trigger_layers: Optional[List[int]] = None,
    ):
        if not self._mlps_replaced:
            self._replace_mlps_with_efvr()
            self._register_efvr_hooks()

        # 更新全局状态
        self.efvr_state.update({
            'active': True,
            'entropy_threshold': entropy_threshold,
            'retracing_ratio': retracing_ratio,
            'starting_layer': starting_layer,
            'ending_layer': ending_layer,
            'trigger_layers': set(trigger_layers) if trigger_layers is not None else None,
            'triggered_this_step': False
        })

        for layer in self._text_model.layers:
            layer.mlp.retracing_ratio = retracing_ratio

        print(f"[EFVR] Activated  entropy_threshold={entropy_threshold}  retracing_ratio={retracing_ratio}  trigger_layers={trigger_layers}", flush=True)

    def deactivate_efvr(self):
        if self._mlps_replaced:
            for layer in self._text_model.layers:
                m = layer.mlp
                m.adpt_sign = 0
                m.adpt_w1   = None
                m.adpt_w2   = None
        self.efvr_state['active'] = False
        self.efvr_state['visual_token'] = None
        self.efvr_state['triggered_this_step'] = False

    def _register_efvr_hooks(self):
        for i, layer in enumerate(self._text_model.layers):
            h = layer.register_forward_hook(self._make_layer_hook(i))
            self._efvr_hooks.append(h)

    def _make_layer_hook(self, layer_idx: int):
        model_ref = self

        def hook(module, inp, output):
            # 用 Try-Except 包裹整个 Hook，防止底层报错被静默吞噬
            try:
                # ── 0. Step 级别防抖重置 ──
                if layer_idx == 0:
                    model_ref.efvr_state['triggered_this_step'] = False 

                # ── 1. Cleanup ──
                mlp = module.mlp
                if mlp.adpt_sign == 1:
                    mlp.adpt_sign = 0
                    mlp.adpt_w1   = None
                    mlp.adpt_w2   = None

                # ── 2. 判断是否激活 ──
                if not model_ref.efvr_state['active']:
                    return output

                hidden_states = output[0] if isinstance(output, tuple) else output
                
                # 安全校验，防止异常的空 Tensor
                if len(hidden_states.shape) < 3 or hidden_states.shape[1] == 0:
                    return output

                # ── 3. 提取最后一位 Token 计算熵值并【强制刷新打印】 ──
                with torch.no_grad():
                    last_token_hidden = hidden_states[:, -1:, :] 
                    norm_h     = model_ref._text_model.norm(last_token_hidden)    
                    logits     = model_ref.lm_head(norm_h[:, 0, :]).float()  
                    top_scores, _ = torch.topk(logits, 6, dim=-1)
                    probs      = F.softmax(top_scores, dim=-1) + 1e-9
                    entropy    = torch.sum(-probs * torch.log(probs) / np.log(10)).item()

                print(f"[EFVR] layer={layer_idx:2d}  entropy={entropy:.3f}  threshold={model_ref.efvr_state['entropy_threshold']:.3f}", flush=True)

                # ── 4. 拦截已触发状态 & 判断监控层 ──
                if model_ref.efvr_state['triggered_this_step']:
                    return output
                
                t_layers = model_ref.efvr_state['trigger_layers']
                if t_layers is not None:
                    if layer_idx not in t_layers:
                        return output
                else:
                    if not (model_ref.efvr_state['starting_layer'] < layer_idx < model_ref.efvr_state['ending_layer']):
                        return output

                # ── 5. 触发 EFVR 注入 ──
                visual_token = model_ref.efvr_state['visual_token']
                if entropy > model_ref.efvr_state['entropy_threshold'] and visual_token is not None:
                    next_idx = layer_idx + 1
                    if next_idx < len(model_ref._text_model.layers):
                        model_ref.efvr_state['triggered_this_step'] = True
                        next_mlp  = model_ref._text_model.layers[next_idx].mlp
                        next_mlp.adpt_sign = 1

                        dev, dt = hidden_states.device, hidden_states.dtype

                        if next_mlp.adpt_w1 is None:
                            next_mlp.adpt_w1 = nn.Parameter(torch.zeros_like(visual_token), requires_grad=False).to(dev, dtype=dt)
                        if next_mlp.adpt_w2 is None:
                            next_mlp.adpt_w2 = nn.Parameter(torch.zeros_like(visual_token.T), requires_grad=False).to(dev, dtype=dt)

                        w1_denom = torch.mean(torch.abs(visual_token)) + 1e-9
                        w2_denom = torch.mean(torch.abs(visual_token.T)) + 1e-9
                        
                        next_mlp.adpt_w1.data.copy_(
                            (torch.mean(torch.abs(next_mlp.up_proj.weight)) / w1_denom) * visual_token.to(dev, dtype=dt)
                        )
                        next_mlp.adpt_w2.data.copy_(
                            (torch.mean(torch.abs(next_mlp.down_proj.weight)) / w2_denom) * visual_token.T.to(dev, dtype=dt)
                        )
                        print(f"[EFVR INJECT] Triggered at layer {layer_idx} → injecting at layer {next_idx}", flush=True)

            except Exception as e:
                # 捕获并强制打印任何隐蔽的 Hook 错误
                import traceback
                print(f"\n[EFVR HOOK ERROR] Layer {layer_idx}: {e}", flush=True)
                traceback.print_exc()

            return output

        return hook

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mask_index: Optional[List[int]] = None,
        mask_flag:  bool = False,       
        mask_input: bool = False,       
        **kwargs,
    ):
        prefill_hidden_states = None
        if inputs is not None and pixel_values is not None:
            attention_mask = kwargs.get('attention_mask', None)
            try:
                fwd_out = self.forward(
                    input_ids=inputs,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                prefill_hidden_states = fwd_out.hidden_states

                if self.efvr_state['active']:
                    img_tok_id = getattr(
                        self.config, 'image_token_id',
                        getattr(getattr(self.config, 'text_config', None), 'image_token_id', None)
                    )
                    if img_tok_id is not None:
                        img_pos = (inputs[0] == img_tok_id).nonzero(as_tuple=True)[0]
                        if img_pos.numel() > 0:
                            img_embeds = prefill_hidden_states[0][0, img_pos, :]
                            if mask_index is not None and len(mask_index) > 0:
                                n    = img_embeds.shape[0]
                                keep = torch.ones(n, dtype=torch.bool, device=img_embeds.device)
                                for idx in mask_index:
                                    if 0 <= idx < n:
                                        keep[idx] = False
                                img_embeds = img_embeds[keep]
                            self.efvr_state['visual_token'] = img_embeds
                        else:
                            warnings.warn("[EFVR] No image tokens found in input_ids; visual_token not set.")
                    else:
                        warnings.warn("[EFVR] image_token_id not found in config; visual_token not set.")
                    self.efvr_state['triggered_this_step'] = False   

            except Exception as exc:
                warnings.warn(f"[EPMEM] Prefill forward pass failed: {exc}")
                prefill_hidden_states = None

        gen_kwargs = dict(kwargs)
        gen_kwargs.setdefault('output_scores', True)
        gen_kwargs.setdefault('return_dict_in_generate', True)

        used_embeds = False   
        if (mask_input
                and mask_index is not None
                and len(mask_index) > 0
                and prefill_hidden_states is not None
                and inputs is not None):
            img_tok_id = getattr(
                self.config, 'image_token_id',
                getattr(getattr(self.config, 'text_config', None), 'image_token_id', None)
            )
            if img_tok_id is not None:
                img_pos = (inputs[0] == img_tok_id).nonzero(as_tuple=True)[0]
                if img_pos.numel() > 0:
                    masked_embeds = prefill_hidden_states[0].clone()
                    for local_idx in mask_index:
                        if 0 <= local_idx < img_pos.numel():
                            global_pos = img_pos[local_idx].item()
                            masked_embeds[0, global_pos, :] = 0.0
                    gen_out     = super().generate(
                        inputs_embeds=masked_embeds,
                        **gen_kwargs,
                    )
                    used_embeds = True   
                else:
                    gen_out = super().generate(input_ids=inputs, pixel_values=pixel_values, image_grid_thw=image_grid_thw, **gen_kwargs)
            else:
                gen_out = super().generate(input_ids=inputs, pixel_values=pixel_values, image_grid_thw=image_grid_thw, **gen_kwargs)
        else:
            gen_out = super().generate(input_ids=inputs, pixel_values=pixel_values, image_grid_thw=image_grid_thw, **gen_kwargs)

        prompt_len = 0 if used_embeds else (inputs.shape[1] if inputs is not None else 0)
        new_token_ids = gen_out.sequences[:, prompt_len:]

        token_logits = list(gen_out.scores) if gen_out.scores is not None else []

        return new_token_ids, token_logits, prefill_hidden_states