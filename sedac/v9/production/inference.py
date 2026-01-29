"""
SEDAC V9.0 Production Inference Pipeline

完整的生产级推理管线，支持 Qwen/LLaMA 等主流模型
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Generator
from dataclasses import dataclass
import logging
import time
from pathlib import Path

from .config import ProductionConfig, ModelConfig, PrecisionMode
from .engine import ProductionSEDACEngine, ForwardOutput
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_strings: Optional[List[str]] = None


@dataclass
class InferenceResult:
    """推理结果"""
    generated_text: str
    generated_tokens: int
    input_tokens: int
    total_latency_ms: float
    tokens_per_second: float
    avg_exit_layer: float
    skip_ratio: float
    used_o1: bool
    thinking_steps: int


class SEDACInferencePipeline:
    """
    SEDAC 生产级推理管线
    
    特性:
    - 自动模型加载与配置
    - 流式生成支持
    - 批量推理
    - 完整的错误处理
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[ProductionConfig] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        local_files_only: bool = False,  # 支持离线模式
    ):
        self.model_name = model_name_or_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if config is None:
            config = ProductionConfig()
            if not local_files_only:
                try:
                    config.model = ModelConfig.from_pretrained(model_name_or_path)
                except Exception as e:
                    logger.warning(f"Could not load model config: {e}")
        
        config.device = str(self.device)
        self.config = config
        
        if dtype is not None:
            self.dtype = dtype
        elif load_in_8bit:
            self.dtype = torch.int8
            config.precision = PrecisionMode.INT8
        elif load_in_4bit:
            self.dtype = torch.int8
            config.precision = PrecisionMode.INT4
        else:
            self.dtype = config.get_dtype()
        
        self.model = None
        self.tokenizer = None
        self.sedac_engine = None
        
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._local_files_only = local_files_only
        self._is_loaded = False
    
    def load(self) -> "SEDACInferencePipeline":
        """加载模型和 tokenizer"""
        if self._is_loaded:
            return self
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=self._local_files_only,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device.type == "cuda" else None,
            "local_files_only": self._local_files_only,
        }
        
        if self._load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self._load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = self.dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        if not (self._load_in_8bit or self._load_in_4bit) and self.device.type == "cuda":
            if not hasattr(self.model, "hf_device_map"):
                self.model = self.model.to(self.device)
        
        self.model.eval()
        
        self.sedac_engine = ProductionSEDACEngine(self.config, self.model)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        self._is_loaded = True
        return self
    
    def _ensure_loaded(self) -> None:
        """确保模型已加载"""
        if not self._is_loaded:
            self.load()
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """
        推理接口
        
        Args:
            prompt: 输入文本或文本列表
            generation_config: 生成配置
            **kwargs: 额外参数传递给 generation_config
        
        Returns:
            单个或多个 InferenceResult
        """
        self._ensure_loaded()
        
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        
        results = []
        for p in prompts:
            result = self._generate_single(p, generation_config)
            results.append(result)
        
        return results if is_batch else results[0]
    
    def _generate_single(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> InferenceResult:
        """单条生成"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_position_embeddings - config.max_new_tokens,
        ).to(self.device)
        
        input_tokens = inputs.input_ids.shape[1]
        
        start_time = time.perf_counter()
        
        exit_layers = []
        o1_used = False
        thinking_steps = 0
        
        generated = inputs.input_ids.clone()
        past_key_values = None
        attention_mask = inputs.attention_mask
        
        for step in range(config.max_new_tokens):
            if past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            output = self.sedac_engine.forward_with_sedac(
                current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            exit_layers.append(output.exit_layer)
            if output.used_o1:
                o1_used = True
                thinking_steps += output.thinking_steps
            
            logits = output.logits[:, -1, :]
            
            next_token = self._sample_token(logits, config)
            
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = output.past_key_values
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            eos_id = config.eos_token_id or self.tokenizer.eos_token_id
            if eos_id is not None and next_token.item() == eos_id:
                break
        
        total_latency = (time.perf_counter() - start_time) * 1000
        generated_tokens = generated.shape[1] - input_tokens
        
        generated_text = self.tokenizer.decode(
            generated[0, input_tokens:],
            skip_special_tokens=True
        )
        
        avg_exit = sum(exit_layers) / len(exit_layers) if exit_layers else self.config.model.num_hidden_layers
        total_layers = self.config.model.num_hidden_layers
        skip_ratio = 1.0 - avg_exit / total_layers
        
        return InferenceResult(
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            input_tokens=input_tokens,
            total_latency_ms=total_latency,
            tokens_per_second=generated_tokens / (total_latency / 1000) if total_latency > 0 else 0,
            avg_exit_layer=avg_exit,
            skip_ratio=skip_ratio,
            used_o1=o1_used,
            thinking_steps=thinking_steps,
        )
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """采样下一个 token"""
        if config.repetition_penalty != 1.0:
            pass
        
        if config.temperature > 0 and config.do_sample:
            logits = logits / config.temperature
            
            if config.top_k > 0:
                top_k = min(config.top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return logits.argmax(dim=-1, keepdim=True)
    
    def stream_generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成
        
        Yields:
            生成的文本片段
        """
        self._ensure_loaded()
        
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated = inputs.input_ids.clone()
        past_key_values = None
        attention_mask = inputs.attention_mask
        
        prev_text_len = 0
        
        for step in range(generation_config.max_new_tokens):
            if past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            output = self.sedac_engine.forward_with_sedac(
                current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = output.logits[:, -1, :]
            next_token = self._sample_token(logits, generation_config)
            
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = output.past_key_values
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            current_text = self.tokenizer.decode(
                generated[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            if len(current_text) > prev_text_len:
                yield current_text[prev_text_len:]
                prev_text_len = len(current_text)
            
            eos_id = generation_config.eos_token_id or self.tokenizer.eos_token_id
            if eos_id is not None and next_token.item() == eos_id:
                break
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 8,
        **kwargs
    ) -> List[InferenceResult]:
        """批量生成"""
        self._ensure_loaded()
        
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = [self._generate_single(p, generation_config) for p in batch]
            results.extend(batch_results)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.sedac_engine is None:
            return {}
        return self.sedac_engine.get_metrics()
    
    def reset_metrics(self) -> None:
        """重置指标"""
        if self.sedac_engine is not None:
            self.sedac_engine.metrics.reset()


def create_pipeline(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    **kwargs
) -> SEDACInferencePipeline:
    """
    工厂函数：创建推理管线
    
    Args:
        model_name: 模型名称或路径
        device: 设备
        **kwargs: 传递给 SEDACInferencePipeline
    
    Returns:
        已加载的推理管线
    """
    pipeline = SEDACInferencePipeline(model_name, device=device, **kwargs)
    pipeline.load()
    return pipeline
