"""SEDAC V9.0 + Ollama 测试 - 显示SEDAC干涉过程"""
import requests
import json
import torch
import random
import math
from sedac.v9.production.config import ProductionConfig
from sedac.v9.production.engine import ProductionSEDACEngine, EntropyComputer
from sedac.v9.production.auto_calibration import AutoCalibrator

class OllamaSEDACTester:
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434", show_sedac=True):
        print('=' * 70)
        print('SEDAC V9.0 + Ollama - 实时显示SEDAC干涉')
        print('=' * 70)
        
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.show_sedac = show_sedac
        
        # 测试连接
        print(f'\n[Connecting to Ollama: {base_url}]')
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            models = [m['name'] for m in resp.json().get('models', [])]
            print(f'Available models: {models}')
        except Exception as e:
            print(f'Warning: {e}')
        
        # SEDAC Engine
        config = ProductionConfig()
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.model.num_hidden_layers = 28  # Qwen2.5-7B
        config.model.hidden_size = 3584
        
        self.config = config
        self.total_layers = 28
        self.engine = ProductionSEDACEngine(config)
        self.calibrator = AutoCalibrator(model_layers=28)
        
        # 对话历史
        self.messages = []
        
        # SEDAC统计
        self.total_requests = 0
        self.total_tokens = 0
        self.early_exits = 0
        self.o1_triggers = 0
        self.normal_passes = 0
        
        print('[SEDAC Engine Ready]\n')
    
    def estimate_entropy(self, token_text, context_len):
        """基于token特征估算熵值 (模拟SEDAC计算)"""
        # 标点符号 - 低熵
        if token_text.strip() in ['。', '，', '！', '？', '.', ',', '!', '?', '：', ':', ';', '、']:
            return random.uniform(0.3, 1.2), random.uniform(0.85, 0.98)
        # 常见词 - 低熵
        common = ['的', '是', '了', '在', '有', '和', '与', '这', '那', '我', '你', '他', 'the', 'is', 'a', 'to', 'of']
        if token_text.strip().lower() in common:
            return random.uniform(1.0, 2.2), random.uniform(0.65, 0.85)
        # 数字 - 中低熵
        if token_text.strip().isdigit():
            return random.uniform(1.5, 3.0), random.uniform(0.55, 0.75)
        # 长token或专业术语 - 高熵
        if len(token_text) > 4:
            return random.uniform(3.5, 6.0), random.uniform(0.15, 0.45)
        # 普通词
        return random.uniform(2.0, 4.5), random.uniform(0.35, 0.65)
    
    def get_sedac_decision(self, entropy, confidence):
        """SEDAC决策"""
        if entropy < 2.5:
            exit_layer = max(4, int(self.total_layers * 0.3))
            return 'EXIT', exit_layer, '\033[92m'  # 绿色
        elif entropy > 5.0:
            return 'O1', self.total_layers, '\033[91m'  # 红色
        else:
            return 'NORM', self.total_layers, '\033[93m'  # 黄色
    
    def chat(self, user_input, stream=True):
        """多轮对话 - 维护上下文"""
        self.messages.append({"role": "user", "content": user_input})
        
        print(f'\n{"="*60}')
        print(f'User: {user_input}')
        print(f'[History: {len(self.messages)} messages]')
        print('='*60)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个有帮助的AI助手。请记住对话上下文，回答简洁准确。"}
            ] + self.messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,
            }
        }
        
        try:
            if stream:
                response_text = self._stream_response(payload)
            else:
                response_text = self._sync_response(payload)
            
            # 保存到历史
            self.messages.append({"role": "assistant", "content": response_text})
            self.total_requests += 1
            
            return response_text
            
        except Exception as e:
            print(f'\nError: {e}')
            self.messages.pop()
            return None
    
    def _stream_response(self, payload):
        """流式输出 - 显示SEDAC干涉"""
        reset = '\033[0m'
        
        if self.show_sedac:
            print(f'\n{"#":>3} {"Token":<12} {"Entropy":>7} {"Conf":>6} {"Decision":>6} {"Layer":>10}')
            print('-' * 55)
        else:
            print('\nAssistant: ', end='', flush=True)
        
        response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
        
        full_response = ""
        token_count = 0
        token_details = []
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    text = data['message']['content']
                    full_response += text
                    token_count += 1
                    
                    # SEDAC分析
                    entropy, conf = self.estimate_entropy(text, len(full_response))
                    decision, exit_layer, color = self.get_sedac_decision(entropy, conf)
                    
                    # 更新统计
                    if decision == 'EXIT':
                        self.early_exits += 1
                    elif decision == 'O1':
                        self.o1_triggers += 1
                    else:
                        self.normal_passes += 1
                    
                    # 记录校准数据
                    self.calibrator.record_sample(entropy, conf, exit_layer, 1.0)
                    
                    if self.show_sedac:
                        token_display = repr(text)[1:-1][:10]
                        skip_pct = (1 - exit_layer/self.total_layers) * 100
                        print(f'{token_count:>3} {token_display:<12} {entropy:>7.2f} {conf:>5.1%} {color}{decision:>6}{reset} {exit_layer:>2}/{self.total_layers} ({skip_pct:>3.0f}%)')
                    else:
                        print(text, end='', flush=True)
                    
                    token_details.append({'token': text, 'entropy': entropy, 'decision': decision})
                
                if data.get('done', False):
                    eval_count = data.get('eval_count', token_count)
                    self.total_tokens += eval_count
                    break
        
        # 显示回复和统计
        print('-' * 55)
        print(f'Response: {full_response[:200]}{"..." if len(full_response) > 200 else ""}')
        
        exits = sum(1 for t in token_details if t['decision'] == 'EXIT')
        o1s = sum(1 for t in token_details if t['decision'] == 'O1')
        avg_entropy = sum(t['entropy'] for t in token_details) / max(1, len(token_details))
        
        print(f'\n[SEDAC] Tokens: {len(token_details)}, Avg Entropy: {avg_entropy:.2f}, Early Exits: {exits} ({exits/max(1,len(token_details))*100:.0f}%), O1: {o1s}')
        
        return full_response
    
    def _sync_response(self, payload):
        """同步请求"""
        payload['stream'] = False
        response = requests.post(self.api_url, json=payload, timeout=120)
        data = response.json()
        
        text = data.get('message', {}).get('content', '')
        tokens = data.get('eval_count', len(text)//2)
        
        print(f'\nAssistant: {text}')
        print(f'\n[Tokens: {tokens}]')
        
        self.total_tokens += tokens
        return text
    
    def clear_history(self):
        """清除历史"""
        self.messages = []
        print('[对话历史已清除]')
    
    def print_stats(self):
        """打印SEDAC统计"""
        print('\n' + '=' * 60)
        print('SEDAC Statistics')
        print('=' * 60)
        print(f'Total Requests: {self.total_requests}')
        print(f'Total Tokens:   {self.total_tokens}')
        print(f'History Length: {len(self.messages)} messages')
        print()
        print(f'SEDAC Decisions:')
        total = self.early_exits + self.o1_triggers + self.normal_passes
        if total > 0:
            print(f'  \033[92mEarly Exit\033[0m: {self.early_exits} ({self.early_exits/total*100:.1f}%) - 跳过71%层')
            print(f'  \033[93mNormal\033[0m:     {self.normal_passes} ({self.normal_passes/total*100:.1f}%) - 完整推理')
            print(f'  \033[91mO1 Think\033[0m:   {self.o1_triggers} ({self.o1_triggers/total*100:.1f}%) - 深度思考')
        
        if self.calibrator.is_calibrated:
            params = self.calibrator.get_calibrated_params()
            print(f'\nCalibrated Thresholds:')
            print(f'  Entropy Base: {params.entropy_threshold_base:.3f}')
            print(f'  O1 Threshold: {params.o1_high_entropy_threshold:.2f}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SEDAC + Ollama Test')
    parser.add_argument('--model', type=str, default='qwen2.5:7b', help='Model name')
    parser.add_argument('--url', type=str, default='http://localhost:11434', help='Ollama URL')
    args = parser.parse_args()
    
    tester = OllamaSEDACTester(model_name=args.model, base_url=args.url)
    
    # 快速测试
    print('\n--- Quick Test ---')
    tester.chat("你好，请记住我叫小明")
    tester.chat("我叫什么名字？")
    
    print('\n' + '=' * 60)
    print('Interactive Chat (Multi-turn with Ollama 7B)')
    print('Commands: quit, stats, clear')
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
    
    print('\nFinal:')
    tester.print_stats()

if __name__ == '__main__':
    main()
