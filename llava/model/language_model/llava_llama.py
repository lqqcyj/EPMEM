#    Copyright 2023 Haotian Liu
# ( ... Apache License ... )


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modelling_logits_llama import LlamaLogitsModel, LlamaLogitsForCausalLM
from transformers.generation.logits_process import LogitsProcessorList


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaLogitsModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaLogitsForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaLogitsForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # efvr MERGE: 添加一个新的方法来激活和配置 efvr
    def activate_efvr(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float,
        trigger_layers: Optional[List[int]] = None # <-- 添加此行
    ):
        """
        激活内联的 efvr 功能。
        这取代了 efvr.py 中的 apply_efvr_llama 猴子补丁。
        """
        print(f"efvr MERGE: Activating efvr logic with params:")
        print(f"  starting_layer: {starting_layer}")
        print(f"  ending_layer: {ending_layer}")
        print(f"  entropy_threshold: {entropy_threshold}")
        print(f"  retracing_ratio: {retracing_ratio}")

        # 检查 self.model.layers[0].mlp 是否是我们预期的 LlamaefvrMLP
        # (我们在 modelling_logits_llama.py 中设置了它)
        if not hasattr(self.model.layers[0].mlp, 'apply_efvr'):
             raise TypeError(
                 "Model's MLP layer does not seem to be LlamaefvrMLP."
                 "Merge failed in 'modelling_logits_llama.py'."
             )

        # 传递配置到 LlamaefvrMLP 实例 (存储在第0层)
        self.model.layers[0].mlp.apply_efvr = True
        self.model.layers[0].mlp.starting_layer = starting_layer
        self.model.layers[0].mlp.ending_layer = ending_layer
        self.model.layers[0].mlp.entropy_threshold = entropy_threshold
        self.model.layers[0].mlp.trigger_layers = trigger_layers # <-- 添加此行
        
        # 将 retracing_ratio 设置到 *所有* 层的 MLP
        for layer in range(self.config.num_hidden_layers):
            self.model.layers[layer].mlp.retracing_ratio = retracing_ratio


    def deactivate_efvr(self):
        """
        显式关闭 efvr 功能，用于 Stage 1 或重置状态。
        """
        print("efvr CONTROL: Deactivating efvr logic.")
        if hasattr(self.model.layers[0].mlp, 'apply_efvr'):
            self.model.layers[0].mlp.apply_efvr = False
            self.model.layers[0].mlp.visual_token = None
            
            # 清理所有层可能的残留状态
            for layer in self.model.layers:
                if hasattr(layer.mlp, 'adpt_sign'):
                    layer.mlp.adpt_sign = 0
                if hasattr(layer.mlp, 'adpt_w1'):
                    layer.mlp.adpt_w1 = None
                if hasattr(layer.mlp, 'adpt_w2'):
                    layer.mlp.adpt_w2 = None

    def forward(
        self,
        # ( ... 此处内容与原文件一致 ... )
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # efvr MERGE: 在这里添加 logits_processor，以接收来自 greedy_search 的调用
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                # efvr MERGE: 确保我们解包 llava_arch.py 返回的额外值
                # 即使它们在这里未被使用 (它们在 generate() 中被使用)
                _image_shape, 
                _pre_prompt_length_list,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        # efvr MERGE: LlamaLogitsForCausalLM.forward 现在需要 'logits_processor'
        # 在标准的 .forward() 调用中 (非 .generate())，我们传递 None
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
            logits_processor=None, # efvr MERGE: 添加
        )

    @torch.no_grad()
    def generate(
        self,
        # ( ... 此处内容与原文件一致 ... )
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        # ( ... 此处内容与原文件一致 ... )
        attention_mask = kwargs.pop("attention_mask", None)
        mask_flag = kwargs.pop("mask_flag", False)
        mask_index = kwargs.pop("mask_index", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_shape, # epmem:
                pre_prompt_length_list # epmem:
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                mask_flag=mask_flag, 
                mask_token=mask_index
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # efvr MERGE: 确保 epmem 的值被定义，即使没有图像
            image_shape = 0
            pre_prompt_length_list = []

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_shape = image_shape, # epmem:
            pre_prompt_length_list = pre_prompt_length_list, # epmem:
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)