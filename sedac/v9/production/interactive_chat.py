"""
SEDAC V9.0 Interactive Chat with SEDAC Visualization

交互式对话测试，实时显示 SEDAC 介入过程
"""
from __future__ import annotations
import torch
import time
import sys
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class SEDACStepInfo:
    """单步 SEDAC 信息"""
    layer_idx: int
    entropy: float
    confidence: float
    threshold: float
    decision: str  # "continue", "exit", "thinking"
    ghost_kv_used: bool = False


@dataclass
class SEDACTokenTrace:
    """单 Token 生成的 SEDAC 追踪"""
    token_id: int
    token_text: str
    steps: List[SEDACStepInfo] = field(default_factory=list)
    exit_layer: int = 0
    total_layers: int = 28
    generation_time_ms: float = 0.0
    
    @property
    def skip_ratio(self) -> float:
        if self.total_layers == 0:
            return 0.0
        return 1.0 - self.exit_layer / self.total_layers


def colorize(text: str, color: str) -> str:
    """简单终端着色"""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


class InteractiveSEDACChat:
    """
    交互式 SEDAC 对话
    
    实时显示每个 Token 生成时 SEDAC 的决策过程
    
    支持:
    - 本地模型路径 (避免网络问题)
    - 离线模式
    - 自动校准
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        verbose: bool = True,
        local_files_only: bool = False,  # 支持离线模式
    ):
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.local_files_only = local_files_only
        
        self.model = None
        self.tokenizer = None
        self.sedac_engine = None
        self.calibrator = None
        
        self.token_traces: List[SEDACTokenTrace] = []
    
    def setup(self) -> bool:
        """初始化模型和 SEDAC"""
        print(colorize("Loading model and SEDAC engine...", "bold"))
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .config import ProductionConfig
            from .engine import ProductionSEDACEngine
            from .auto_calibration import AutoCalibrator
            
            print(f"  Loading tokenizer from {colorize(self.model_name, 'cyan')}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                )
            except Exception as e:
                print(colorize(f"  Failed to load tokenizer: {e}", "yellow"))
                print(colorize("  Tip: Use --local if model is cached, or specify local path", "dim"))
                raise
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"  Loading model {colorize(self.model_name, 'cyan')}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                )
            except Exception as e:
                print(colorize(f"  Failed to load model: {e}", "yellow"))
                print(colorize("  Tip: Ensure model is downloaded or use local path", "dim"))
                raise
            
            config = ProductionConfig()
            config.device = self.device
            
            try:
                num_layers = self.model.config.num_hidden_layers
                hidden_size = self.model.config.hidden_size
                config.model.num_hidden_layers = num_layers
                config.model.hidden_size = hidden_size
                config.model.vocab_size = self.model.config.vocab_size
            except:
                pass
            
            self.calibrator = AutoCalibrator(model_layers=config.model.num_hidden_layers)
            
            self.sedac_engine = ProductionSEDACEngine(config)
            
            print(colorize("✓ Setup complete!", "green"))
            print(colorize("✓ Auto-calibration enabled - thresholds will adapt automatically", "cyan"))
            return True
            
        except Exception as e:
            print(colorize(f"Setup failed: {e}", "red"))
            import traceback
            traceback.print_exc()
            return False
    
    @torch.no_grad()
    def generate_with_trace(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Tuple[str, List[SEDACTokenTrace]]:
        """生成文本并追踪 SEDAC 决策"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        traces = []
        generated_ids = input_ids.clone()
        
        total_layers = self.sedac_engine.config.model.num_hidden_layers
        
        print(f"\n{colorize('═══ SEDAC Generation Trace ═══', 'cyan')}\n")
        print(f"{'#':>4} {'Token':<12} {'Entropy':>8} {'Conf':>6} {'Layer':>10} {'Bar':<12} {'Decision':<10} {'Time':>8}")
        print("-" * 80)
        
        for step in range(max_new_tokens):
            start_time = time.perf_counter()
            
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1:, :]
            
            exit_mask, entropy, confidence = self.sedac_engine.should_exit(
                hidden, logits.unsqueeze(1), layer_idx=total_layers - 1, total_layers=total_layers
            )
            
            entropy_val = entropy.mean().item()
            conf_val = confidence.mean().item()
            threshold = self.sedac_engine.threshold_controller.get_threshold(0.5)
            
            if entropy_val < 3.0:
                decision = "exit"
                exit_layer = max(4, int(total_layers * (entropy_val / 5.0)))
            elif entropy_val > self.sedac_engine.config.sedac.o1_high_entropy_threshold:
                decision = "thinking"
                exit_layer = total_layers
            else:
                decision = "continue"
                exit_layer = total_layers
            
            self.calibrator.record_sample(
                entropy=entropy_val,
                confidence=conf_val,
                exit_layer=exit_layer,
                quality=1.0,
            )
            
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_text = self.tokenizer.decode(next_token[0])
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            trace = SEDACTokenTrace(
                token_id=step,
                token_text=token_text,
                exit_layer=exit_layer,
                total_layers=total_layers,
                generation_time_ms=elapsed_ms,
            )
            trace.steps.append(SEDACStepInfo(
                layer_idx=exit_layer,
                entropy=entropy_val,
                confidence=conf_val,
                threshold=threshold,
                decision=decision,
                ghost_kv_used=(exit_layer < total_layers - 4),
            ))
            traces.append(trace)
            
            if self.verbose:
                self._print_step(trace)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        return response, traces
    
    def _print_step(self, trace: SEDACTokenTrace) -> None:
        """打印单步信息"""
        step = trace.steps[-1] if trace.steps else None
        if not step:
            return
        
        token_display = trace.token_text.replace("\n", "↵").replace("\t", "→")
        if len(token_display) > 10:
            token_display = token_display[:10] + ".."
        
        if step.entropy < 3.0:
            entropy_color = "green"
        elif step.entropy < 5.0:
            entropy_color = "yellow"
        else:
            entropy_color = "red"
        
        decision_icons = {"exit": "⚡EXIT", "thinking": "🤔THINK", "continue": "→CONT"}
        decision_display = decision_icons.get(step.decision, step.decision)
        
        bar_len = trace.exit_layer * 10 // trace.total_layers
        layer_bar = "▓" * bar_len + "░" * (10 - bar_len)
        
        print(
            f"{trace.token_id:>4} "
            f"{token_display:<12} "
            f"{colorize(f'{step.entropy:>7.2f}', entropy_color)} "
            f"{step.confidence:>5.0%} "
            f"{trace.exit_layer:>3}/{trace.total_layers:<3} "
            f"{colorize(layer_bar, 'cyan')} "
            f"{decision_display:<10} "
            f"{trace.generation_time_ms:>6.1f}ms"
        )
    
    def print_summary(self, traces: List[SEDACTokenTrace]) -> None:
        """打印汇总统计"""
        if not traces:
            return
        
        avg_exit = sum(t.exit_layer for t in traces) / len(traces)
        avg_skip = sum(t.skip_ratio for t in traces) / len(traces)
        avg_time = sum(t.generation_time_ms for t in traces) / len(traces)
        total_time = sum(t.generation_time_ms for t in traces)
        
        exit_decisions = sum(1 for t in traces if t.steps and t.steps[-1].decision == "exit")
        thinking_decisions = sum(1 for t in traces if t.steps and t.steps[-1].decision == "thinking")
        
        print(f"\n{colorize('═══ SEDAC Summary ═══', 'cyan')}")
        print(f"  Total Tokens:    {len(traces)}")
        print(f"  Avg Exit Layer:  {avg_exit:.1f} / {traces[0].total_layers}")
        print(f"  Avg Skip Ratio:  {colorize(f'{avg_skip*100:.1f}%', 'green')}")
        print(f"  Early Exits:     {exit_decisions} ({exit_decisions/len(traces)*100:.1f}%)")
        print(f"  Deep Thinking:   {thinking_decisions} ({thinking_decisions/len(traces)*100:.1f}%)")
        print(f"  Avg Time/Token:  {avg_time:.1f}ms")
        print(f"  Total Time:      {total_time:.1f}ms")
        print(f"  TPS:             {len(traces) / (total_time/1000):.1f}")
        
        if self.calibrator.is_calibrated:
            params = self.calibrator.get_calibrated_params()
            print(f"\n{colorize('═══ Calibrated Parameters ═══', 'magenta')}")
            print(f"  Entropy Base:    {params.entropy_threshold_base:.3f} (auto-learned)")
            print(f"  O1 Threshold:    {params.o1_high_entropy_threshold:.2f} (auto-learned)")
            print(f"  Min Exit Layer:  {params.min_exit_layer} (auto-learned)")
    
    def chat_loop(self) -> None:
        """交互式对话循环"""
        print(f"\n{colorize('╔══════════════════════════════════════════════╗', 'cyan')}")
        print(f"{colorize('║', 'cyan')}   SEDAC V9.0 Interactive Chat                {colorize('║', 'cyan')}")
        print(f"{colorize('║', 'cyan')}   Commands: /quit, /stats, /clear, /params   {colorize('║', 'cyan')}")
        print(f"{colorize('╚══════════════════════════════════════════════╝', 'cyan')}\n")
        
        session_traces = []
        
        while True:
            try:
                user_input = input(f"\n{colorize('You:', 'green')} ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/stats":
                self.print_summary(session_traces)
                continue
            elif user_input.lower() == "/clear":
                session_traces.clear()
                print(colorize("Session cleared.", "dim"))
                continue
            elif user_input.lower() == "/params":
                if self.calibrator.is_calibrated:
                    params = self.calibrator.get_calibrated_params()
                    print(f"\n{colorize('Current Calibrated Parameters:', 'magenta')}")
                    import json
                    print(json.dumps(params.to_dict(), indent=2))
                else:
                    print(colorize("Not yet calibrated. Generate more tokens.", "yellow"))
                continue
            
            response, traces = self.generate_with_trace(user_input)
            session_traces.extend(traces)
            
            print(f"\n{colorize('Assistant:', 'blue')} {response}")
            self.print_summary(traces)
        
        print(colorize("\nGoodbye!", "dim"))


def run_interactive_chat(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cuda",
    local_files_only: bool = False,
) -> None:
    """运行交互式对话"""
    
    if not torch.cuda.is_available() and device == "cuda":
        print(colorize("CUDA not available, using CPU", "yellow"))
        device = "cpu"
    
    chat = InteractiveSEDACChat(model_name, device, local_files_only=local_files_only)
    
    if not chat.setup():
        return
    
    chat.chat_loop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC Interactive Chat")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name or local path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")
    parser.add_argument("--local", action="store_true",
                       help="Use local files only (offline mode)")
    
    args = parser.parse_args()
    run_interactive_chat(args.model, args.device, args.local)
