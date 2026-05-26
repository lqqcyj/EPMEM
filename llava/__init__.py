try:
    from .model.language_model.llava_llama import LlavaLlamaForCausalLM
except Exception:
    # Older LLaVA dependencies (e.g. transformers.generation.beam_constraints)
    # may not be present in the Qwen conda environment.  The Qwen-2.5-VL
    # pipeline does not require LlavaLlamaForCausalLM.
    pass
