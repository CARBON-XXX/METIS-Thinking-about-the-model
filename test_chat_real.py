"""SEDAC V9.0 真实多轮对话测试 - 逐Token显示SEDAC决策"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sedac.v9.production.config import ProductionConfig
from sedac.v9.production.engine import ProductionSEDACEngine, EntropyComputer
from sedac.v9.production.auto_calibration import AutoCalibrator

class SEDACChatTester:
    def __init__(self):
        print('=' * 70)
        print('SEDAC V9.0 - Real Chat Test with Token-by-Token SEDAC Decisions')
        print('=' * 70)
        
        print('\n[Loading Model...]')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True,
            torch_dtype=torch.float16, device_map='auto'
        )
        self.model.eval()
        
        # 对话历史
        self.messages = []
        
        print(f'Model: Qwen2.5-0.5B ({self.model.config.num_hidden_layers} layers)')
        
        # SEDAC Engine
        config = ProductionConfig()
        config.device = 'cuda'
        config.model.num_hidden_layers = self.model.config.num_hidden_layers
        config.model.hidden_size = self.model.config.hidden_size
        config.model.vocab_size = self.model.config.vocab_size
        
        self.config = config
        self.engine = ProductionSEDACEngine(config)
        self.entropy_computer = EntropyComputer(config)
        self.calibrator = AutoCalibrator(model_layers=self.model.config.num_hidden_layers)
        self.total_layers = self.model.config.num_hidden_layers
        
        # Stats
        self.total_tokens = 0
        self.early_exits = 0
        self.o1_triggers = 0
        self.normal_passes = 0
        
        print('[SEDAC Engine Ready]\n')
    
    def get_decision(self, entropy_val, conf_val):
        """根据熵值决定策略"""
        if entropy_val < 2.5:  # 低熵 - 确定性高
            return 'EXIT', max(4, int(self.total_layers * 0.3))
        elif entropy_val > 5.0:  # 高熵 - 需要深度思考
            return 'O1', self.total_layers
        else:  # 中熵 - 正常
            return 'NORM', self.total_layers
    
    def format_prompt(self, user_input, use_history=True):
        """使用Qwen chat模板格式化prompt，支持多轮对话"""
        if use_history:
            # 添加新用户消息到历史
            self.messages.append({"role": "user", "content": user_input})
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Remember the conversation context."}
            ] + self.messages
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        
        # 使用tokenizer的chat模板
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    
    def add_assistant_response(self, response):
        """将助手回复添加到历史"""
        self.messages.append({"role": "assistant", "content": response})
    
    def clear_history(self):
        """清除对话历史"""
        self.messages = []
        print("[对话历史已清除]")
    
    @torch.no_grad()
    def generate_with_sedac(self, prompt, max_tokens=50):
        """逐Token生成并显示SEDAC决策"""
        # 使用chat模板
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors='pt').to('cuda')
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]
        
        print(f'\n{"#":>3} {"Token":<15} {"Entropy":>8} {"Conf":>7} {"Decision":>8} {"Layer":>8}')
        print('-' * 60)
        
        generated_tokens = []
        token_stats = []
        
        for step in range(max_tokens):
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :]
            
            # 计算熵和置信度
            entropy, confidence = self.entropy_computer.compute(logits)
            entropy_val = entropy.mean().item() if hasattr(entropy, 'mean') else float(entropy)
            conf_val = confidence.mean().item() if hasattr(confidence, 'mean') else float(confidence)
            
            # SEDAC 决策
            decision, exit_layer = self.get_decision(entropy_val, conf_val)
            
            # 更新统计
            self.total_tokens += 1
            if decision == 'EXIT':
                self.early_exits += 1
            elif decision == 'O1':
                self.o1_triggers += 1
            else:
                self.normal_passes += 1
            
            # 记录校准数据
            self.calibrator.record_sample(entropy_val, conf_val, exit_layer, 1.0)
            
            # 采样下一个token (greedy)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            token_text = self.tokenizer.decode([token_id])
            
            # 显示
            token_display = repr(token_text)[1:-1][:12]
            skip_pct = (1 - exit_layer/self.total_layers) * 100
            
            color_code = '\033[92m' if decision == 'EXIT' else ('\033[91m' if decision == 'O1' else '\033[93m')
            reset = '\033[0m'
            
            print(f'{step:>3} {token_display:<15} {entropy_val:>8.2f} {conf_val:>6.1%} {color_code}{decision:>8}{reset} {exit_layer:>3}/{self.total_layers} ({skip_pct:>4.0f}%)')
            
            token_stats.append({
                'token': token_text,
                'entropy': entropy_val,
                'confidence': conf_val,
                'decision': decision,
                'exit_layer': exit_layer
            })
            
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 终止条件
            if token_id == self.tokenizer.eos_token_id:
                break
            # 检测重复或无意义输出
            if len(generated_tokens) > 10:
                last_tokens = generated_tokens[-5:]
                if len(set(last_tokens)) <= 2:  # 重复检测
                    break
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip(), token_stats
    
    def print_stats(self):
        """打印统计信息"""
        print('\n' + '=' * 60)
        print('SEDAC Statistics')
        print('=' * 60)
        print(f'Total Tokens:    {self.total_tokens}')
        print(f'Early Exits:     {self.early_exits} ({self.early_exits/max(1,self.total_tokens)*100:.1f}%)')
        print(f'O1 Triggers:     {self.o1_triggers} ({self.o1_triggers/max(1,self.total_tokens)*100:.1f}%)')
        print(f'Normal Passes:   {self.normal_passes} ({self.normal_passes/max(1,self.total_tokens)*100:.1f}%)')
        
        if self.calibrator.is_calibrated:
            params = self.calibrator.get_calibrated_params()
            print(f'\nCalibrated Thresholds:')
            print(f'  Entropy Base: {params.entropy_threshold_base:.3f}')
            print(f'  O1 Threshold: {params.o1_high_entropy_threshold:.2f}')
    
    def chat(self, prompt):
        """多轮对话 - 维护上下文历史"""
        print(f'\n{"="*60}')
        print(f'User: {prompt}')
        print(f'[History: {len(self.messages)} messages]')
        print('='*60)
        
        response, stats = self.generate_with_sedac(prompt)
        
        # 保存助手回复到历史
        self.add_assistant_response(response)
        
        print('-' * 60)
        print(f'Response: {response}')
        
        # Token级统计
        avg_entropy = sum(s['entropy'] for s in stats) / len(stats) if stats else 0
        avg_conf = sum(s['confidence'] for s in stats) / len(stats) if stats else 0
        exits = sum(1 for s in stats if s['decision'] == 'EXIT')
        
        print(f'\n[Stats] Tokens: {len(stats)}, Avg Entropy: {avg_entropy:.2f}, Avg Conf: {avg_conf:.1%}, Early Exits: {exits}')
        
        return response

def main():
    tester = SEDACChatTester()
    
    # 测试对话
    test_prompts = [
        "Hi",
        "What is 2+2?",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about AI.",
    ]
    
    for prompt in test_prompts:
        tester.chat(prompt)
        print()
    
    tester.print_stats()
    
    print('\n' + '=' * 60)
    print('Interactive Mode (Multi-turn with Context Memory)')
    print('Commands: quit, stats, clear (clear history)')
    print('=' * 60)
    
    while True:
        try:
            user_input = input('\nYou: ').strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.lower() == 'stats':
                tester.print_stats()
                continue
            if user_input.lower() == 'clear':
                tester.clear_history()
                continue
            if not user_input:
                continue
            tester.chat(user_input)
        except KeyboardInterrupt:
            break
    
    print('\nFinal Statistics:')
    tester.print_stats()

if __name__ == '__main__':
    main()
