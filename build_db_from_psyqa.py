import json
from openai import OpenAI
import time

# 替换为你自己的 DeepSeek API Key
DEEPSEEK_KEY = "sk-4fa691e4f66940d3b797f02b404c478b"
BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_KEY, base_url=BASE_URL)

def extract_intervention_with_llm(question, description, answer):
    """让大模型把杂乱的问答，提取成标准的数据库结构"""
    prompt = f"""
    你是一个数据清洗专家。请从以下真实的心理咨询问答中，提取出一个具体的、可操作的心理干预建议。
    
    【求助者问题】：{question}
    【详细情况】：{description}
    【咨询师回复】：{answer}

    请必须返回 JSON 格式，包含以下字段：
    {{
        "emotion_type": "从 愤怒, 难过, 恐惧, 高兴, 厌恶, 惊讶, 中性 中选出最符合求助者的1个情绪",
        "category": "从 认知重构, 行动, CBT练习, 正念, 社交, 书籍, 电影, 音乐 中选1个",
        "content": "从咨询师的回复中，提取出一个具体的行动建议（限15字内，如：采用非暴力沟通表达需求）",
        "reason": "提取出这个建议的心理学依据或作用，并结合求助者的具体场景（限30字内）"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"提取失败: {e}")
        return None

def main():
    # 假设你的文件名为 PsyQA_full.json，请确保它和这个 Python 脚本在同一个文件夹下
    file_path = 'PsyQA_example.json' # 如果你的文件名不同，请在这里修改
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}，请检查文件名和路径。")
        return

    print(f"✅ 成功加载数据集，共找到 {len(data)} 条问答记录。")
    print("⏳ 开始自动化清洗并生成 SQL，为了演示我们先处理前 50 条...\n")
    
    sql_statements = []
    
    # 我们只取前 50 条做提取，你也可以改成取更多
    for item in data[:50]:
        question = item.get('question', '')
        description = item.get('description', '')
        
        # 获取咨询师的第一个回复文本
        answers = item.get('answers', [])
        if not answers:
            continue
        answer_text = answers[0].get('answer_text', '')
        
        print(f"正在处理: {question}...")
        result = extract_intervention_with_llm(question, description, answer_text)
        
        if result:
            sql = f"INSERT INTO `recommendations` (`emotion_type`, `category`, `content`, `reason`) VALUES ('{result['emotion_type']}', '{result['category']}', '{result['content']}', '{result['reason']}');"
            sql_statements.append(sql)
            print(f"  └─ 成功提取: [{result['emotion_type']}] {result['content']}")
            
        time.sleep(1.5) # 防止请求过快
        
    # 保存 SQL
    with open("psyqa_interventions.sql", "w", encoding="utf-8") as f:
        f.write("\n".join(sql_statements))
        
    print("\n🎉 处理完毕！请在 Navicat 中运行新生成的 psyqa_interventions.sql")

if __name__ == "__main__":
    main()