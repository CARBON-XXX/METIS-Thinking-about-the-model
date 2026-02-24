"""
SEDAC V10 元认知核心测试
验证元认知系统的自我意识能力
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from sedac.v10 import (
    MetacognitiveCore,
    MetaJudgment,
    EpistemicState,
    CognitiveLoad,
    CognitiveBoundary,
    MetaAction,
    create_metacognitive_core,
)


def print_judgment(judgment: MetaJudgment, prefix: str = ""):
    """打印元认知判断"""
    print(f"{prefix}┌─────────────────────────────────────────")
    print(f"{prefix}│ 认知状态: {judgment.epistemic_state.value}")
    print(f"{prefix}│ 认知负荷: {judgment.cognitive_load.value}")
    print(f"{prefix}│ 边界状态: {judgment.boundary_status.value}")
    print(f"{prefix}│ 建议行为: {judgment.suggested_action.value}")
    print(f"{prefix}├─────────────────────────────────────────")
    print(f"{prefix}│ 确信度: {judgment.certainty_score:.2f}")
    print(f"{prefix}│ 负荷度: {judgment.load_score:.2f}")
    print(f"{prefix}│ 边界分: {judgment.boundary_score:.2f}")
    print(f"{prefix}│ 熵趋势: {judgment.entropy_trend}")
    print(f"{prefix}├─────────────────────────────────────────")
    print(f"{prefix}│ 内省: {judgment.introspection}")
    print(f"{prefix}└─────────────────────────────────────────")


class MetacognitiveChat:
    """带元认知的对话系统"""
    
    def __init__(self, model_path: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        
        # 4-bit 量化
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.model.eval()
        
        # 创建元认知核心
        self.metacog = create_metacognitive_core(self.model)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 元认知核心已初始化")
    
    @torch.no_grad()
    def chat(self, query: str, max_tokens: int = 100, verbose: bool = True):
        """带元认知的对话"""
        
        # 重置认知轨迹
        self.metacog.reset()
        
        # 构建输入
        messages = [{"role": "user", "content": query}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
        
        input_ids = inputs.input_ids
        past_key_values = None
        generated_tokens = []
        judgments = []
        
        for step in range(max_tokens):
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # 元认知内省
            judgment = self.metacog.introspect(logits)
            judgments.append(judgment)
            
            # 采样
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            
            if verbose and step < 10:
                token_str = self.tokenizer.decode([next_token.item()])
                print(f"[{step:2d}] '{token_str}' | "
                      f"状态:{judgment.epistemic_state.value[:4]} "
                      f"确信:{judgment.certainty_score:.2f} "
                      f"行为:{judgment.suggested_action.value}")
            
            # 元认知行为响应
            if judgment.suggested_action == MetaAction.ACKNOWLEDGE:
                if verbose:
                    print(f"\n[元认知] {judgment.introspection}")
            elif judgment.suggested_action == MetaAction.ABORT:
                if verbose:
                    print(f"\n[元认知] 停止：{judgment.introspection}")
                break
            elif judgment.suggested_action == MetaAction.SEEK_INFO:
                if verbose:
                    print(f"\n[元认知] 触发信息检索：{judgment.introspection}")
            
            # 检查结束
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            input_ids = next_token
        
        # 解码响应
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Response: {response}")
            print(f"{'─'*60}")
            
            # 最终元认知总结
            if judgments:
                final = judgments[-1]
                print("\n[最终元认知状态]")
                print_judgment(final, "  ")
        
        return response, judgments


def main():
    print("="*60)
    print("SEDAC V10 元认知系统测试")
    print("="*60)
    
    # 测试用例
    test_cases = [
        ("简单问答", "中国的首都是哪里？"),
        ("数学推理", "一个水池有两个进水管，A管6小时注满，B管4小时注满，同时开多久注满？"),
        ("未知问题", "2030年世界杯冠军是谁？"),
        ("代码生成", "用Python写一个快速排序"),
        ("哲学问题", "意识的本质是什么？"),
    ]
    
    model_path = "G:/models/qwen2.5-7b"
    
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return
    
    chat = MetacognitiveChat(model_path)
    
    for name, query in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"# 测试: {name}")
        print(f"{'#'*60}")
        
        response, judgments = chat.chat(query, max_tokens=50, verbose=True)
        
        # 统计
        if judgments:
            states = [j.epistemic_state.value for j in judgments]
            actions = [j.suggested_action.value for j in judgments]
            avg_certainty = sum(j.certainty_score for j in judgments) / len(judgments)
            
            print(f"\n[统计]")
            print(f"  平均确信度: {avg_certainty:.2f}")
            print(f"  认知状态分布: {dict((s, states.count(s)) for s in set(states))}")
            print(f"  行为触发: {dict((a, actions.count(a)) for a in set(actions))}")
        
        input("\n按 Enter 继续下一个测试...")
    
    print("\n\n测试完成！")


if __name__ == "__main__":
    main()
