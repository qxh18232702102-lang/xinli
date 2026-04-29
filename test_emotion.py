# test_emotion.py
import os
import sys
# 确保导入的是最新版本的类
from nlp_bert_model import TextEmotionAnalyzer

def test_model_loading():
    print("=========================================")
    print("=== 正在运行模型加载测试 (绕过 Streamlit) ===")
    print("=========================================")
    
    analyzer = TextEmotionAnalyzer()
    
    if analyzer.model1 is not None and analyzer.model2 is not None:
        try:
            # 使用一个明确的积极和消极文本进行测试
            test_text_pos = "今天天气很好，我感到非常开心，心情棒极了。"
            test_text_neg = "我的毕设一直报错，让我感到非常沮丧和愤怒！"
            
            # --- 测试积极文本 ---
            result_pos = analyzer.predict(test_text_pos)
            print("\n==============================================")
            print("✅ 测试 1 (积极文本):")
            print(f"输入文本: {test_text_pos}")
            print(f"情感极性: {result_pos.get('情感极性')}, 置信度: {result_pos.get('极性置信度')}")
            print(f"具体情绪: {result_pos.get('具体情绪')}, 置信度: {result_pos.get('情绪置信度')}")
            print(f"最终情绪 (融合): **{result_pos.get('最终情绪')}**")
            print(f"各情绪概率 Top 3: {sorted(result_pos.get('各情绪概率').items(), key=lambda item: item[1], reverse=True)[:3]}")

            # --- 测试消极文本 ---
            result_neg = analyzer.predict(test_text_neg)
            print("\n==============================================")
            print("✅ 测试 2 (消极文本):")
            print(f"输入文本: {test_text_neg}")
            print(f"情感极性: {result_neg.get('情感极性')}, 置信度: {result_neg.get('极性置信度')}")
            print(f"具体情绪: {result_neg.get('具体情绪')}, 置信度: {result_neg.get('情绪置信度')}")
            print(f"最终情绪 (融合): **{result_neg.get('最终情绪')}**")
            print(f"各情绪概率 Top 3: {sorted(result_neg.get('各情绪概率').items(), key=lambda item: item[1], reverse=True)[:3]}")
            print("==============================================")

        except Exception as e:
            print(f"\n❌ 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ 模型加载失败 (TextEmotionAnalyzer.__init__ 中已捕获错误)")
        
if __name__ == "__main__":
    test_model_loading()