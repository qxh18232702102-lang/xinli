import os
import json
import base64
import re
from openai import OpenAI
from zhipuai import ZhipuAI 

def parse_llm_json(text):
    """
    清洗 AI 返回的 JSON 字符串，具备极强的容错能力
    """
    try:
        # 1. 预处理：移除 Markdown 标记
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # 没找到代码块，尝试找首尾大括号
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]
        
        text = text.strip()
        
        # 2. 第一次尝试：标准解析
        return json.loads(text, strict=False)

    except json.JSONDecodeError:
        try:
            print("⚠️ JSON 解析初步失败，尝试暴力修复换行符...")
            # 3. 暴力修复：将所有真实换行符替换为转义符，防止报错
            # 这一步能解决 "Invalid control character" 错误
            text = text.replace('\n', '\\n').replace('\r', '')
            return json.loads(text, strict=False)
        except:
            # 4. 最终兜底：如果还是挂了，返回一个包含原始内容的字典
            return {
                "cloud_emotion": "解析异常",
                "advice": text, # 把原始内容直接展示给用户
                "rec_category": "AI原始回复",
                "rec_content": "格式需人工解读",
                "rec_reason": "JSON格式错误"
            }
# ================= 配置区 =================
DEEPSEEK_KEY = "sk-4fa691e4f66940d3b797f02b404c478b"
BASE_URL = "https://api.deepseek.com"

# 🔴 请务必确认这里填的是智谱 API Key
ZHIPU_KEY = "12288a3ab74b46bf994a6dc5717ba38c.TJx4SoWOXfevwHwL" 

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=BASE_URL)
vision_client = ZhipuAI(api_key=ZHIPU_KEY) 

def check_crisis_risk(text):
    try:
        system_prompt = "You are a crisis intervention expert. Please analyze the input and output a JSON object with a boolean field 'is_crisis'."
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            temperature=0.0, response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content).get("is_crisis", False)
    except: return False

def analyze_intent_and_select_tool(text, history):
    # 🔥 1. 彻底删除之前的关键词硬编码 (那些 if any... 全部不要了)
    
    try:
        # 🔥 2. 使用新的“高情商”提示词
        system_prompt = """
                你是一个极其敏锐的心理咨询分诊督导。你的任务是严格把关，决定用户当前是需要【日常聊天陪伴 (Normal_Chat)】，还是需要【启用潜意识测试工具 (Deep_Therapy)】。
                
                🔴 核心准则：心理测试（如房树人、完形填空）会消耗用户的心理能量。在错误的时机弹出测试，会造成严重的破坏。

                🚫 【绝对禁止弹窗 (强制返回 Normal_Chat)】：
                1. 用户明确表达极度疲惫、崩溃、无力（例如：“累瘫了”、“一句话都不想说”、“不想活了”）。
                2. 用户正在进行高浓度的情绪宣泄，带有强烈的愤怒或悲伤（例如：“气死我了”、“我一直在哭”）。
                3. 用户清晰地陈述了具体事件的因果关系，即使情绪负面（例如：“我因为考试考砸了所以很难受”）。
                4. 用户只是进行简单的日常问候或回复你的上一句话。

                🟢 【黄金弹窗时机 (允许返回 Deep_Therapy 并指定工具)】：
                只有当用户具备一定的心理能量，且符合以下任一情况时：
                1. **述情障碍/言语受限**：用户感到痛苦，但无法用语言准确描述原因（例如：“不知道怎么形容”、“说不上来哪里不对劲”、“心里堵得慌但不知道为什么”）。此时可推荐 `htp_drawing` (房树人) 或 `sentence_completion` (完形填空)。
                2. **强迫性重复/死胡同**：用户在反复纠结同一个无解的问题，意识到自己的模式但无法打破（例如：“我总是做同一个梦”、“每次都搞砸”）。此时可推荐 `sentence_completion`。
                3. **高度防御下的隔离**：用户在描述一个本该悲伤的事件，但语气出奇的冷漠和理智化（例如：“虽然分手了，但我理智分析过这很正常”）。此时可推荐 `htp_drawing` 破防。
                4. **主动要求**：用户明确提出想做测试、画画或玩卡牌。

                🔴 "response_to_user" 字段要求：
                这是一句【温暖、自然、像真人一样】的引导语。绝对禁止使用机械的官方套话。
                - 如果是 Deep_Therapy，语气要像邀请老朋友尝试新方法：“感觉你心里好像装了很多事，要不我们试着画幅画，看看能不能把这些情绪理清楚？”
                - 如果是 Normal_Chat，可以留空或给出简短的安抚。

                请严格按照以上规则进行判断，返回 JSON 格式：
                {
                    "intent_type": "Deep_Therapy" 或 "Normal_Chat",
                    "suggested_tool": "htp_drawing" 或 "sentence_completion" 或 "none",
                    "response_to_user": "给用户的简短引导语",
                    "reason": "严格对照上述规则给出的判断依据"
                }
                """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": f"用户输入: {text}"}
            ],
            temperature=0.5,  # 🔥 关键：提高温度，增加灵活性
            response_format={ "type": "json_object" }
        )
        
        # 使用之前修好的解析函数
        return parse_llm_json(response.choices[0].message.content)

    except Exception as e:
        print(f"❌ 分诊模型出错: {e}")
        return {"intent_type": "Normal_Chat", "suggested_tool": "none", "response_to_user": "", "reason": "API Error"}
def consult_llm(prompt_text):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Output JSON."}, 
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7, response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except: return None

# =========================================================
# 🔥🔥🔥 核心修复：智谱 GLM-4V 视觉分析 (静默版) 🔥🔥🔥
# =========================================================
def analyze_htp_with_vision(image_path):
    # 只打印英文日志，防止 Windows 控制台报错
    print(f"[Vision] Processing image path: {image_path}")
    
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        response = vision_client.chat.completions.create(
            model="glm-4v", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            你是一位温暖、循循善诱的资深心理咨询师（CBT流派）。
                            用户刚刚上传了一张【房树人(HTP)】心理投射绘画。
                            请仔细观察画面细节（如线条力度、画面大小、位置、缺失部分、涂抹痕迹）进行专业分析。

                            请严格按照以下 JSON 格式返回（不要输出 markdown，JSON 必须压缩为单行，字符串内不要使用真实换行符，请使用 \\n 代替）：
                            {
                                "cloud_emotion": "分析出的核心情绪词(如: 焦虑/防御/渴望关注)",
                                "advice": "请输出一段【对话式】的咨询回复。要求：1. 先共情，肯定用户愿意画画的行为。2. 指出画面中1-2个具体的特征并尝试解读。3. 【关键】最后必须抛出一个引导性问题，引导用户继续在输入框里和你对话。",
                                "rec_category": "本周作业",
                                "rec_content": "具体的心理练习任务",
                                "rec_reason": "练习目的"
                            }
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            }
                        }
                    ]
                }
            ]
        )
        
# --- 这里是替换后的代码 ---
        
        # 1. 获取原始文本
        raw_content = response.choices[0].message.content
        
        # 2. 调用上面的清洗函数进行解析
        result_dict = parse_llm_json(raw_content)
        
        print("[Vision] Analysis success. JSON parsed.")
        return result_dict

        # --- 替换结束 ---

    except Exception as e:
        # 只打印错误类型，不打印可能包含中文的具体内容
        print(f"[Vision Error] An error occurred: {type(e)}")
        
        # 兜底返回
        return {
            "cloud_emotion": "潜意识整合",
            "advice": "（视觉模型连接波动）你的画作已收到。绘画本身就是一种疗愈，通过线条，我们能看到潜意识的流动。建议保持这种觉察。",
            "rec_category": "自我关怀",
            "rec_content": "正念呼吸",
            "rec_reason": "平复内心"
        }