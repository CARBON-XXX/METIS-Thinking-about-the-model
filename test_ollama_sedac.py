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
        
        # 动态复杂度 - 根据实时熵分布自适应
        self.current_complexity = 'normal'
        self.entropy_window = []  # 滑动窗口
        self.adaptive_o1_threshold = 5.0  # 动态O1阈值
        
        print('[SEDAC Engine Ready]\n')
    
    def detect_complexity(self, user_input):
        """
        语义级复杂度检测 - 基于语言结构特征而非关键词
        
        特征维度:
        1. 句法复杂度: 句子长度、从句深度、标点密度
        2. 词汇复杂度: 平均词长、罕见字符比例、数学符号
        3. 语义深度: 疑问类型、抽象层次、推理需求
        4. 信息密度: 实词/虚词比、概念密度
        """
        # 基础统计
        text = user_input
        length = len(text)
        
        # === 特征提取 ===
        score = 0.0
        
        # 1. 句法复杂度 (0-3分)
        # 长问题通常更复杂
        if length > 50: score += 0.5
        if length > 100: score += 0.5
        if length > 200: score += 0.5
        
        # 从句标记 (逗号、分号密度)
        clause_markers = text.count('，') + text.count(',') + text.count('；') + text.count(';')
        if clause_markers > 2: score += 0.3
        if clause_markers > 5: score += 0.3
        
        # 嵌套结构 (括号深度)
        bracket_depth = text.count('(') + text.count('（') + text.count('[') + text.count('【')
        if bracket_depth > 0: score += 0.4
        if bracket_depth > 2: score += 0.4
        
        # 2. 词汇复杂度 (0-3分)
        # 数学/逻辑符号密度
        math_chars = sum(1 for c in text if c in '∀∃∈⊂∪∩→⇒≡≅∫∑∏√≤≥≠∞αβγδεθλμπσφψω')
        latex_patterns = text.count('\\') + text.count('$') + text.count('^') + text.count('_')
        if math_chars > 0 or latex_patterns > 0:
            score += min(1.5, (math_chars + latex_patterns) * 0.3)
        
        # 罕见Unicode字符 (专业术语指标)
        rare_chars = sum(1 for c in text if ord(c) > 0x4E00 and ord(c) < 0x9FFF)  # CJK
        tech_density = rare_chars / max(1, length)
        if tech_density > 0.3: score += 0.5
        
        # 平均"词"长度 (中文按字，英文按空格分词)
        words = text.replace('，', ' ').replace('。', ' ').split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len > 4: score += 0.4
        
        # 3. 语义深度 (0-3分)
        # 疑问类型分析
        wh_questions = any(q in text for q in ['为什么', '如何', '怎样', 'why', 'how', '本质', '原理'])
        if wh_questions: score += 0.6
        
        # 抽象层次 (元认知词汇)
        meta_cognitive = any(m in text for m in ['解释', '分析', '比较', '评价', '综合', '证', '推', '论'])
        if meta_cognitive: score += 0.5
        
        # 多步推理标记
        multi_step = any(s in text for s in ['首先', '然后', '因此', '所以', '由此', '步骤', 'step'])
        if multi_step: score += 0.4
        
        # 4. 信息密度 (0-2分)
        # 数字密度 (具体数值问题)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, length)
        if digit_ratio > 0.05: score += 0.3
        
        # 专有名词密度 (大写字母在英文部分)
        english_part = ''.join(c for c in text if ord(c) < 128)
        if english_part:
            upper_ratio = sum(1 for c in english_part if c.isupper()) / max(1, len(english_part))
            if upper_ratio > 0.15: score += 0.4
        
        # === 复杂度判定 ===
        # score范围约0-11，映射到三级
        if score >= 4.0:
            return 'proof'
        elif score >= 2.0:
            return 'complex'
        else:
            return 'normal'
    
    def estimate_entropy(self, token_text, context_len):
        """
        确定性熵估算 - 基于token语言特征，无随机成分
        
        熵值 = 基础熵 + 稀有度加成 + 上下文加成
        置信度 = 1 - 归一化熵
        """
        text = token_text.strip()
        
        # === 1. 基础熵 (token类型) ===
        # 标点符号 - 最低熵 (高度可预测)
        if text in '。，！？.?!,:;：；、':
            base_entropy = 0.8
            base_conf = 0.92
        # 常见虚词
        elif text in ['的', '是', '了', '在', '有', '和', '与', '这', '那', '我', '你', '他', 
                      'the', 'is', 'a', 'an', 'to', 'of', 'and', 'or', 'but']:
            base_entropy = 1.5
            base_conf = 0.78
        # 数字
        elif text.isdigit():
            base_entropy = 2.0
            base_conf = 0.65
        # 单个汉字
        elif len(text) == 1 and '\u4e00' <= text <= '\u9fff':
            base_entropy = 3.0
            base_conf = 0.50
        # 短英文
        elif len(text) <= 3 and text.isalpha():
            base_entropy = 2.5
            base_conf = 0.55
        # 长token
        else:
            base_entropy = 3.5 + min(2.0, len(text) * 0.3)
            base_conf = max(0.2, 0.6 - len(text) * 0.05)
        
        # === 2. 稀有度加成 (专业术语检测) ===
        rare_bonus = 0.0
        
        # 数学/逻辑符号
        math_chars = set('∀∃∈⊂⊃∪∩→⇒⇔≡≅≈∫∑∏√≤≥≠∞∂∇αβγδεζηθλμνξπρσφψω')
        if any(c in text for c in math_chars):
            rare_bonus += 3.0
        
        # LaTeX标记
        if any(m in text for m in ['\\', '$', '^', '_', '{', '}']):
            rare_bonus += 2.0
        
        # 核心数学概念 (高认知负荷词汇)
        math_concepts = {'群', '环', '域', '模', '拓扑', '流形', '同构', '同态', '映射', 
                        '核', '像', '商', '范畴', '函子', '态射', '极限', '余极限',
                        '积分', '微分', '导数', '变换', '空间', '维', '算子', '谱',
                        '特征', '本征', '矩阵', '向量', '张量', '李代数', '伽罗瓦'}
        if any(c in text for c in math_concepts):
            rare_bonus += 3.5
        
        # 英文专业术语 (大写开头或全大写)
        if text and text[0].isupper() and len(text) > 2:
            rare_bonus += 1.0
        if text.isupper() and len(text) > 1:
            rare_bonus += 0.5
        
        # === 3. 上下文位置加成 ===
        # 长上下文中的token更难预测
        context_bonus = min(1.0, context_len / 500)
        
        # === 最终计算 ===
        entropy = base_entropy + rare_bonus + context_bonus
        confidence = max(0.05, base_conf - rare_bonus * 0.1 - context_bonus * 0.05)
        
        return entropy, confidence
    
    def update_adaptive_threshold(self, entropy, confidence):
        """
        自适应阈值更新 - 基于实时熵分布动态调整O1触发条件
        
        核心思想: 检测到高熵token就立即响应，而非等待统计积累
        """
        # 更新滑动窗口
        self.entropy_window.append(entropy)
        if len(self.entropy_window) > 15:
            self.entropy_window.pop(0)
        
        if len(self.entropy_window) >= 3:
            window_mean = sum(self.entropy_window) / len(self.entropy_window)
            window_max = max(self.entropy_window[-5:]) if len(self.entropy_window) >= 5 else max(self.entropy_window)
            
            # 检测专业术语峰值 (单个高熵token就足以说明问题复杂)
            has_peak = window_max > 5.5
            
            # 计算高熵比例
            high_ratio = sum(1 for e in self.entropy_window if e > 3.5) / len(self.entropy_window)
            
            # 动态阈值调整 - 更激进的策略
            if has_peak or high_ratio > 0.4:
                # 发现专业术语 → 立即降低阈值
                self.adaptive_o1_threshold = max(3.0, min(self.adaptive_o1_threshold, window_mean + 0.5))
                self.current_complexity = 'proof'
            elif high_ratio > 0.2:
                self.adaptive_o1_threshold = max(3.5, window_mean + 0.8)
                self.current_complexity = 'complex'
            else:
                self.adaptive_o1_threshold = min(5.0, self.adaptive_o1_threshold + 0.02)
                self.current_complexity = 'normal'
    
    def get_sedac_decision(self, entropy, confidence):
        """
        SEDAC决策 - 自适应动态阈值
        
        不依赖关键词，而是根据模型生成过程中的实际不确定性来判断
        """
        # 先更新自适应阈值
        self.update_adaptive_threshold(entropy, confidence)
        
        # 动态阈值
        o1_threshold = self.adaptive_o1_threshold
        exit_threshold = 2.5 if self.current_complexity == 'normal' else 1.8
        
        # 低置信度也触发O1 (模型自己不确定)
        low_confidence_trigger = confidence < 0.25
        
        if entropy < exit_threshold and confidence > 0.7:
            exit_layer = max(4, int(self.total_layers * 0.3))
            return 'EXIT', exit_layer, '\033[92m'  # 绿色 - 快速退出
        elif entropy > o1_threshold or low_confidence_trigger:
            return 'O1', self.total_layers, '\033[91m'  # 红色 - 深度思考
        else:
            return 'NORM', self.total_layers, '\033[93m'  # 黄色 - 正常推理
    
    def chat(self, user_input, stream=True):
        """多轮对话 - 维护上下文"""
        self.messages.append({"role": "user", "content": user_input})
        
        # 重置滑动窗口 - 每个新问题从头学习复杂度
        self.entropy_window = []
        self.adaptive_o1_threshold = 5.0
        
        # 初始复杂度估计 (仅作为起点，会被自适应覆盖)
        self.current_complexity = self.detect_complexity(user_input)
        complexity_labels = {'normal': '→自适应', 'complex': '→自适应', 'proof': '→自适应'}
        
        print(f'\n{"="*60}')
        print(f'User: {user_input}')
        print(f'[History: {len(self.messages)} msgs | Complexity: {complexity_labels[self.current_complexity]}]')
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
