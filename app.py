import os
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_file
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
import tempfile
import random
import io
import re
from datetime import datetime
from collections import Counter
import jieba

# PDF与绘图库
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 模型加载区 =================
try:
    from emotion_analyzer.emotion_analyzer import EmotionAnalyzer
    from emotion_face_fer import FaceEmotionRecognizer
    from llm_service import consult_llm, analyze_intent_and_select_tool, analyze_htp_with_vision
    
    print("✅ 正在初始化 AI 引擎...")
    text_model = EmotionAnalyzer()
    face_model = FaceEmotionRecognizer("models/EmoNet_V1.pth") 
    
    # 🔥🔥🔥 新增：V3.0 语义向量检索模型 🔥🔥🔥
    # 使用 shibing624 开源的极佳中文文本匹配模型，体积小，速度快
    print("✅ 正在加载语义向量模型 (Sentence Transformer)...")
    embedder = SentenceTransformer('shibing624/text2vec-base-chinese')
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")

# ================= 配置区 =================
app = Flask(__name__)
app.secret_key = 'set_your_secret_key_here'
CORS(app)

# ================= 登录管理配置 =================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后访问'

# ================= 数据库配置 =================
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1327', 
    'db': 'emotion_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# ================= 危机干预关键词 =================
CRISIS_KEYWORDS = [
    "想死", "不想活", "自杀", "结束生命", "活着没意思", "毫无意义", 
    "毁灭", "绝望", "跳楼", "割腕", "在此了结", "再见世界"
]
CRISIS_REPLY = """亲爱的，我检测到你现在的情绪非常低落，甚至有些危险。
请记住，这只是暂时的阴霾，不是人生的终点。
这个世界虽然不完美，但包含我在内的很多人都在乎你。
请试着深呼吸，或者拨打下方的援助热线，让我们陪你走过这段黑暗。"""

# ================= 模型加载区 =================
try:
    from emotion_analyzer.emotion_analyzer import EmotionAnalyzer
    from emotion_face_fer import FaceEmotionRecognizer
    # 🔥🔥🔥 引入新增的 Agent 函数 🔥🔥🔥
    from llm_service import consult_llm, analyze_intent_and_select_tool,analyze_htp_with_vision
    
    print("✅ 正在初始化 AI 引擎...")
    text_model = EmotionAnalyzer()
    face_model = FaceEmotionRecognizer("models/EmoNet_V1.pth") 
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    text_model = None
    face_model = None

# ================= 用户类 =================
class User(UserMixin):
    def __init__(self, id, username, password_hash, email=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(user_data['id'], user_data['username'], user_data['password_hash'], user_data.get('email'))
    finally:
        conn.close()
    return None

# ================= 辅助函数 =================
HEALING_CORPUS = {
    "高兴": ["看到你开心，我也觉得世界变亮了！✨", "保持这份好心情，你笑起来真好看。"],
    "难过": ["抱抱你，想哭就哭出来吧，我在呢。🫂", "乌云总会散去，明天又是新的一天。"],
    "愤怒": ["深呼吸... 1... 2... 3... 别气坏了身子。🌬️", "去喝杯水降降火，不值得为别人的错误买单。"],
    "恐惧": ["别怕，这只是暂时的，你比想象中更勇敢。🛡️", "我在你身边，一切都会好起来的。"],
    "惊讶": ["生活总是充满了意想不到的小插曲。", "保持好奇心是好事哦。"],
    "厌恶": ["离讨厌的事物远一点，保护好心情。", "去洗把脸，转换一下心情吧。"],
    "中性": ["平平淡淡才是真，享受此刻的宁静。", "又是平和的一天。"]
}

def get_ai_reply(emotion):
    candidates = HEALING_CORPUS.get(emotion, HEALING_CORPUS["中性"])
    return random.choice(candidates)

def map_emotion_to_score(emotion):
    mapping = { "高兴": 2, "惊讶": 1, "中性": 0, "难过": -1, "厌恶": -1, "恐惧": -2, "愤怒": -2 }
    return mapping.get(emotion, 0)

def save_history(content, emotion, healing):
    if not current_user.is_authenticated: return
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            sql = "INSERT INTO history (user_id, input_content, emotion_type, healing_text, create_time) VALUES (%s, %s, %s, %s, NOW())"
            cursor.execute(sql, (current_user.id, content, emotion, healing))
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    except Exception as e:
        print(f"❌ 保存历史失败: {e}")
    finally:
        if conn: conn.close()

# 🔥 核心修改 1：函数参数增加 is_refresh=False
def get_smart_recommendation(user_text, emotion, is_refresh=False):
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            # 【阶段 1：粗排召回】取出该情绪下的所有方案
            sql = "SELECT category, content, reason FROM recommendations WHERE emotion_type = %s"
            cursor.execute(sql, (emotion,))
            candidates = list(cursor.fetchall())
            
        if not candidates:
            return []

        # ==========================================
        # 🔥 定义硬核专业疗法目录
        # ==========================================
        pro_categories = [
            'CBT练习', 'CBT认知行为疗法', 'DBT技巧', 'DBT辩证行为疗法', 
            'ACT疗法', 'ACT接纳承诺疗法', '叙事疗法', '焦点解决', 
            'SFBT焦点解决', '完形疗法', '完形/格式塔疗法', 
            '正念流派', '内在小孩疗法', '积极心理学', '非暴力沟通 (NVC)', 
            '系统脱敏/暴露疗法'
        ]

        if len(candidates) <= 3 or not user_text:
            # 如果候选太少或无文本，随机打乱顺序
            import random
            random.shuffle(candidates)
        else:
            # 【阶段 2：🔥🔥🔥 V3.0 向量检索 (Vector Search) 重排 🔥🔥🔥】
            docs = [f"{c['content']} {c['reason']}" for c in candidates]
            
            user_embedding = embedder.encode([user_text])
            doc_embeddings = embedder.encode(docs)
            
            cosine_scores = cosine_similarity(user_embedding, doc_embeddings).flatten()
            
            for i, c in enumerate(candidates):
                c['score'] = float(cosine_scores[i])
                
            candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            print("-" * 40)
            print(f"📊 [V3.0 向量检索] 打分排名详情：")
            for idx, c in enumerate(candidates[:5]):
                print(f"第{idx+1}名 | 语义匹配度: {c.get('score', 0):.4f} | {c['category']} - {c['content'][:15]}...")
            print("-" * 40)

        # ==========================================
        # 🔥 阶段 3：【动态阈值 + 万能兜底】组装逻辑
        # ==========================================
        pro_recs = [c for c in candidates if c.get('category', '') in pro_categories]
        normal_recs = [c for c in candidates if c.get('category', '') not in pro_categories]

        universal_fallbacks = [
            {'category': '音乐', 'content': '治愈系白噪音/Lo-Fi歌单', 'reason': '科学证明，轻柔的白噪音能最快切断大脑的灾难化反刍。'},
            {'category': '行动', 'content': '喝一杯温热的水', 'reason': '感受水流过食道的温度，把注意力从脑海的画面拉回到身体感官上。'},
            {'category': '行动', 'content': '做5组深呼吸 (4-7-8呼吸法)', 'reason': '吸气4秒，憋气7秒，呼气8秒。强制激活副交感神经，瞬间降低焦虑。'},
            {'category': '行动', 'content': '去洗个手或洗把冷水脸', 'reason': '心理学上的“清洁效应”：水流拂过皮肤的触感，能帮你洗掉一部分心理上的沉重与混沌感。'},
            {'category': '正念', 'content': '视觉锚点：寻找3件蓝色的物品', 'reason': '极其简单的注意力转移法，强迫大脑去处理视觉信息，能瞬间打断负面情绪的死循环。'},
            {'category': '行动', 'content': '站起来，用力伸个懒腰', 'reason': '改变身体僵硬的姿态，能直接向大脑传递“我处于安全状态”的信号，打破情绪冰冻。'},
            {'category': '自我关怀', 'content': '蝴蝶拍 (Butterfly Hug)', 'reason': '双臂交叉抱住自己，左右手交替轻拍肩膀。这种双侧节律刺激是创伤心理学中极佳的自我安抚技术。'},
            {'category': '行动', 'content': '找张废纸，随便乱涂乱画', 'reason': '不需要画出任何具体的形状，让笔尖在纸上剧烈摩擦的动作本身，就是一个极好的物理宣泄口。'},
            {'category': '行动', 'content': '长长地叹一口气', 'reason': '别憋着！刻意地深呼吸并用力叹气，是身体自带的天然减压阀，能有效释放胸腔的压迫感。'},
            {'category': '音乐', 'content': '大自然纯音 (海浪/雨声/篝火)', 'reason': '恒定且无规律的自然底噪，能像安抚婴儿一样，平复大脑深处杏仁核的警报。'},
            {'category': '自我关怀', 'content': '轻轻拍拍自己的头', 'reason': '在心里对自己说句“今天已经很努力了”。身体上的微小正向触碰，能促进催产素分泌，缓解孤独感。'},
            {'category': '行动', 'content': '抬头看一分钟窗外的天空', 'reason': '拉远视觉焦点，视野的开阔会带动心理空间的开阔，让你意识到当下的烦恼在广阔天地间微不足道。'}
        ]

        THRESHOLD = 0.46 
        qualified_normal = [c for c in normal_recs if c.get('score', 0) >= THRESHOLD]

        import random

        # 🔥 核心修改 2：根据 is_refresh 标志位，决定是死拿前两名，还是盲抽
        if not is_refresh:
            # 场景 A：【首次分析】老老实实拿最高分的前 2 名
            final_recs = qualified_normal[:2]
        else:
            # 场景 B：【点击换一换】从及格池子里挑选
            if len(qualified_normal) >= 4:
                # 尽量避开前2名（前2名刚刚已经看过了），从第3名往后盲抽 2 个
                final_recs = random.sample(qualified_normal[2:], 2)
            elif len(qualified_normal) >= 2:
                # 及格的数据不够多，就在整个及格池子里随机抽 2 个
                final_recs = random.sample(qualified_normal, 2)
            else:
                # 如果及格的不到 2 个，有几个拿几个
                final_recs = qualified_normal

        # 万能兜底：如果选出来的普通方案不够 2 个，从兜底池里抽几个补齐
        if len(final_recs) < 2:
            fallback_samples = random.sample(universal_fallbacks, 2 - len(final_recs))
            final_recs.extend(fallback_samples)

        # 拿走第 1 名得分最高的专业方案（通常专业卡片不需要刷新，因为最对症的就那一个）
        if pro_recs:
            final_recs.append(pro_recs[0])
        else:
            # 极端兜底：如果没有专业方案，补一张普通方案凑齐 3 个
            if len(normal_recs) > 2:
                final_recs.append(normal_recs[2])
        
        # 3. 为了前端展示更霸气，给专业卡片的 Category 加上高亮标识
        for rec in final_recs:
            if rec.get('category', '') in pro_categories:
                if not str(rec['category']).startswith('⭐'):
                    rec['category'] = f"⭐ 专家级干预 | {rec['category']}"

        return final_recs

    except Exception as e:
        print(f"❌ 推荐算法出错: {e}")
        return []
    finally:
        if conn: conn.close()

# ================= 页面路由 =================
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = pymysql.connect(**mysql_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user_data = cursor.fetchone()
                if user_data and check_password_hash(user_data['password_hash'], password):
                    user = User(user_data['id'], user_data['username'], user_data['password_hash'], user_data.get('email'))
                    login_user(user)
                    return redirect(url_for('index'))
                else:
                    flash('用户名或密码错误', 'danger')
        finally:
            conn.close()
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return render_template('register.html')

        conn = pymysql.connect(**mysql_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    flash('该用户名已被占用', 'warning')
                    return render_template('register.html')
                hashed_pw = generate_password_hash(password)
                cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)", (username, email, hashed_pw))
            conn.commit()
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ================= 业务接口 =================

@app.route('/analyze/text', methods=['POST'])
@login_required
def api_analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        chat_history = data.get('history', []) 
        
        # 1. 危机干预检测 (优先级最高)
        for keyword in CRISIS_KEYWORDS:
            if keyword in text:
                save_history(text, "危机", CRISIS_REPLY)
                return jsonify({
                    "is_crisis": True,
                    "emotion": "SOS",
                    "ai_reply": CRISIS_REPLY
                })

        # =========================================================
        # 🔥🔥🔥 2. AI 咨询师分诊 (Agent Router) [新增功能] 🔥🔥🔥
        # =========================================================
        # 让 AI 决定是“普通聊天”还是“启动深度治疗”
        
        agent_decision = analyze_intent_and_select_tool(text, chat_history)
        
        # 如果 AI 觉得需要深度治疗 (房树人 / 完形填空)
        if agent_decision.get('intent_type') == 'Deep_Therapy':
            tool = agent_decision.get('suggested_tool')
            reply = agent_decision.get('response_to_user')
            reason = agent_decision.get('reason')
            
            save_history(text, "深度治疗", reply)
            
            return jsonify({
                "is_crisis": False,
                "is_therapy_mode": True,      # 告诉前端：进入治疗模式
                "tool_type": tool,            # 告诉前端：用什么工具
                "emotion": "潜意识探索",       # 状态栏显示
                "ai_reply": reply,
                "recommendations": [],        # 治疗时暂不推荐娱乐内容
                "cloud_analysis": {           # 利用 Layer 3 区域显示治疗思路
                    "emotion": "诊疗中",
                    "advice": "正在启动专业心理干预工具",
                    "rec_category": "心理工具",
                    "rec_content": "房树人" if tool == 'htp_drawing' else "潜意识投射",
                    "rec_reason": reason
                }
            })

        # =========================================================
        # 3. 如果不是治疗模式，走原来的普通分析流程
        # =========================================================
        local_result = text_model.analyze(text)
        emotion = local_result['最终情绪']
        reply = get_ai_reply(emotion)
        recommendations = get_smart_recommendation(text, emotion)
        cloud_data = None
        
        # 4. Layer 3 云端大模型辅助 (DeepSeek)
        if len(text) > 0:
            history_str = ""
            if chat_history:
                history_str = "\n【之前的对话历史】:\n" + "".join([f"{msg['role']}: {msg['content']}\n" for msg in chat_history[-4:]])

# 🔥🔥🔥 核心人设升级 V5.0 (全流派动态闭环版) 🔥🔥🔥
            prompt = f"""
            {history_str}
            【用户当前输入】: "{text}"
            【本地模型初步识别情绪】: "{emotion}"
            
            🔴 你的终极身份：
            你是一位【温暖、包容且经验丰富的全科心理咨询师】。你精通人本主义（共情与陪伴）、CBT（认知行为）以及荣格流派（意象解析）。
            你深知咨询的本质：在心理愈合的过程中，“被看见和被接纳”永远排在“分析和挖掘”的前面。

            🔴 核心应对策略 (请根据上下文，严格选择对应的模式，并执行该模式的【动态闭环】)：

            【模式 1：纯粹抱持与心理急救模式】
            👉 触发条件：用户表达极度疲惫、崩溃、难受、挫败（如“想静静”、“太累了”），或处于高防御、纯宣泄状态。
            👉 应对闭环：
               - 环节 A【无条件接纳】：完全接住用户的情绪，不做任何评判。告诉TA“你有权利感到难过/疲惫”。
               - 环节 B【提供心理毛毯】：强调你的陪伴与存在。话语要像毛毯一样包裹住TA，消除孤独感。
               - 环节 C【出口（见好就收）】：
                 -> 状态 1（用户仍在深渊）：如果用户还在哭泣或宣泄，【绝对禁止提问】。告诉TA“不想说话也没关系，我就在这里陪着你”。
                 -> 状态 2（情绪已触底反弹）：如果用户叹气、或者情绪稍微平复，极其轻柔地引导TA进行微小的现实锚定（如：“也许你可以喝口温水，或者什么都不做，听听下面为你推荐的白噪音。”）
            
            【模式 2：温和探索与认知重构模式】
            👉 触发条件：用户情绪相对平稳，主动描述具体事件的困惑，且没有强烈的抗拒感。
            👉 应对闭环：
               - 环节 A【接住与镜映】：复述或提炼用户的核心意思，让TA觉得“我被听懂了”。
               - 环节 B【识别认知卡点】：在TA的描述中捕捉“认知偏差”（如非黑即白、灾难化想象）或未被满足的“深层需求”。
               - 环节 C【出口（抛锚或收网）】：
                 -> 状态 1（继续深入）：抛出下一个不同视角的启发式问题（如：“有没有另一种可能……”）。
                 -> 状态 2（收网赋能）：如果用户话语中出现“顿悟”或“释怀”信号，【立刻停止提问】！转为全面的肯定，并引导TA查看个性化方案。

            【模式 3：深度意象与潜意识投射模式】
            👉 触发条件：用户主动使用了强烈比喻（如“像掉进黑洞”），或明确在描述画作（房树人）、OH卡牌、梦境时。
            👉 应对闭环：
               - 环节 A【现象学捕捉】：不急于解梦或分析，先纯粹地好奇这个意象的客观细节。
               - 环节 B【感觉具象化提问】：针对意象细节提问 1 个问题（如：“那个黑洞里有光线吗？水是冰凉的还是温暖的？”）。
               - 环节 C【出口（现实映射）】：
                 -> 当用户描述完细节后，将意象与TA的现实情绪连接起来。（如：“听起来，画里那棵孤独的树，很像你刚才提到的在公司里孤立无援的感受，对吗？”）帮助潜意识意识化。

            🔴 你的表达铁律：
            1. 去除所有AI感和客套话，像一个认识很久的温柔长辈或知己。
            2. 每次回复【最多只能包含一个问号】。绝对禁止连珠炮式的追问。
            3. 允许留白，克制“好为人师”的冲动。
            
            请严格分析当前对话上下文，评估用户当前处于哪个模式的哪个环节，然后返回 JSON 格式：
            {{
                "cloud_emotion": "精准的情绪词 (如: 疲惫的难过 / 渴望被抱持 / 释怀的平静)",
                "advice": "这是你要对用户说的话 (100字以内。根据上述闭环严格执行，控制问号数量)",
                "rec_category": "推荐类别 (如: 音乐/冥想/自我关怀)",
                "rec_content": "推荐具体内容",
                "rec_reason": "推荐理由"
            }}
            """
            
            cloud_res = consult_llm(prompt)
            
            if cloud_res:
                cloud_data = {
                    "emotion": cloud_res.get("cloud_emotion", emotion),
                    "advice": cloud_res.get("advice", reply),
                    "rec_category": cloud_res.get("rec_category", "AI推荐"),
                    "rec_content": cloud_res.get("rec_content", ""),
                    "rec_reason": cloud_res.get("rec_reason", "")
                }
                if cloud_data["advice"]: reply = cloud_data["advice"]

        # 5. 保存历史
        history_id = save_history(text, emotion, reply)
        
        return jsonify({
            "is_crisis": False,
            "is_therapy_mode": False,
            "history_id": history_id,
            "emotion": emotion, "ai_reply": reply, "recommendations": recommendations,
            "direction": local_result.get('情感极性', ''), "fine": local_result.get('具体情绪', ''),
            "cloud_analysis": cloud_data
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/image', methods=['POST'])
@login_required
def api_analyze_image():
    # 1. 基础校验
    if 'file' not in request.files: 
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    
    # 🔥 关键新增：同时获取图片类型，以及前端随图片一起发来的【伴随文本】
    img_type = request.form.get('img_type', 'face') 
    text = request.form.get('text', '').strip() # 用户配图写的文字
    
    # 2. 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        path = tmp.name
    
    try:
        # =================================================
        # 🎨 分支 A：房树人绘画模式 (调用智谱 GLM-4V)
        # =================================================
        if img_type == 'htp':
            print("🎨 [处理模式] 启动 GLM-4V 视觉分析...")
            vision_result = analyze_htp_with_vision(path)
            reply = vision_result.get('advice', '分析生成中...')
            emotion = vision_result.get('cloud_emotion', '潜意识整合')
            save_history("[房树人绘画]", emotion, reply)
            return jsonify({
                "is_crisis": False, 
                "is_therapy_mode": True,
                "emotion": emotion, 
                "ai_reply": reply, 
                "recommendations": [],
                "cloud_analysis": {
                    "emotion": "视觉分析完成",
                    "advice": "AI已识别画作细节",
                    "rec_category": vision_result.get('rec_category'),
                    "rec_content": vision_result.get('rec_content'),
                    "rec_reason": vision_result.get('rec_reason')
                }
            })

        # =================================================
        # 📸 分支 B：🔥🔥 多模态融合模式 (人脸 + 文本仲裁) 🔥🔥
        # =================================================
        else:
            print(f"📸 [处理模式] 视觉情绪识别 (伴随文本: {text})")
            
            # 步骤 1：提取面部真实情绪 (底层潜意识)
            face_emotion, _ = face_model.predict_emotion(path)
            
            # 步骤 2：如果没有附带文字，走原来的纯看脸逻辑
            if not text:
                reply = get_ai_reply(face_emotion)
                save_history("[纯图片分析]", face_emotion, reply)
                return jsonify({
                    "is_crisis": False, "is_therapy_mode": False,
                    "emotion": face_emotion, "ai_reply": reply, 
                    "recommendations": get_smart_recommendation("情绪低落", face_emotion)
                })

            # 步骤 3：如果有文字，提取文本情绪 (表层意识)
            text_result = text_model.analyze(text)
            text_emotion = text_result['最终情绪']
            
            # 步骤 4：🔥🔥 核心仲裁机制 (冲突判定) 🔥🔥
            negative_emotions = ["难过", "恐惧", "厌恶", "愤怒"]
            positive_emotions = ["高兴", "中性"]
            
            final_emotion = face_emotion # 默认微表情最诚实，以脸为准！
            fusion_reply = ""
            
            # 冲突场景 A：嘴硬掩饰 (文字积极/中性，人脸负面) -> 比如文字"我很好"，脸"难过"
            if text_emotion in positive_emotions and face_emotion in negative_emotions:
                final_emotion = f"压抑的{face_emotion}"
                fusion_reply = f"虽然你发文字说“{text}”，但我从你的表情里看到了一丝【{face_emotion}】。在我这里不需要强颜欢笑，如果心里藏着委屈，可以随时对我说。"
                
            # 冲突场景 B：自嘲/玩笑 (文字负面，人脸积极) -> 比如文字"气死我了"，脸"高兴"
            elif text_emotion in negative_emotions and face_emotion in positive_emotions:
                final_emotion = "开玩笑"
                fusion_reply = f"哈哈，虽然你嘴上说着“{text}”，但你的表情出卖了你哦，看来并没有真的往心里去呢。"
                
            # 场景 C：表里如一
            else:
                fusion_reply = f"我看到了你脸上的【{face_emotion}】，也听到了你的心声。{get_ai_reply(face_emotion)}"

            # 步骤 5：去 V3.0 向量库拉取方案 (用文字做语义检索，用真实面部情绪做分类过滤)
            recs = get_smart_recommendation(text, face_emotion)
            
            # 保存历史并返回
            save_history(f"[图文融合] {text}", final_emotion, fusion_reply)
            
            return jsonify({
                "is_crisis": False, 
                "is_therapy_mode": False,
                "emotion": final_emotion, # 返回仲裁后的最终情绪
                "ai_reply": fusion_reply, # 返回极其走心的定制回复
                "recommendations": recs
            })

    finally:
        try: os.remove(path)
        except: pass

@app.route('/get/refresh_recommend', methods=['POST'])
@login_required
def api_refresh():
    try:
        data = request.json
        # 1. 尝试获取前端传来的用户输入文本（为了 V3.0 向量检索）
        # 如果前端没传，就默认给个空字符串
        user_text = data.get('text', '') 
        
        # 2. 获取当前的情绪分类
        emotion = data.get('emotion', '中性')
        
        # 3. 调用全新的 V3.0 智能推荐函数！（注意这里传入了两个参数）
        fresh_recs = get_smart_recommendation(user_text, emotion, is_refresh=True)
        
        return jsonify({"recommendations": fresh_recs})
        
    except Exception as e:
        print(f"❌ 刷新推荐卡片失败: {e}")
        return jsonify({"error": "内部服务器错误"}), 500
@app.route('/get/history', methods=['GET'])
@login_required
def api_get_history():
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            sql = "SELECT emotion_type, create_time FROM history WHERE user_id = %s ORDER BY create_time DESC LIMIT 10"
            cursor.execute(sql, (current_user.id,))
            data = cursor.fetchall()
            data.reverse()
            dates = [item['create_time'].strftime("%m-%d %H:%M") for item in data]
            scores = [map_emotion_to_score(item['emotion_type']) for item in data]
            emotions = [item['emotion_type'] for item in data]
            return jsonify({"dates": dates, "scores": scores, "emotions": emotions})
    finally:
        conn.close()

@app.route('/get/calendar_data')
@login_required
def get_calendar_data():
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            sql = "SELECT emotion_type, create_time FROM history WHERE user_id = %s"
            cursor.execute(sql, (current_user.id,))
            records = cursor.fetchall()

        date_map = {}
        for r in records:
            date_str = r['create_time'].strftime('%Y-%m-%d')
            if date_str not in date_map: date_map[date_str] = []
            date_map[date_str].append(r['emotion_type'])

        heatmap_data = []
        for date_str, emotions in date_map.items():
            most_common = max(set(emotions), key=emotions.count)
            score = map_emotion_to_score(most_common)
            heatmap_data.append([date_str, score])

        return jsonify({"data": heatmap_data})
    except Exception as e:
        print(e)
        return jsonify({"data": []})
    finally:
        if conn: conn.close()

@app.route('/get/wordcloud')
@login_required
def get_wordcloud():
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            sql = "SELECT input_content FROM history WHERE user_id = %s AND input_content NOT LIKE '[%%'"
            cursor.execute(sql, (current_user.id,))
            data = cursor.fetchall()
            
        full_text = " ".join([d['input_content'] for d in data])
        stop_words = {'的', '了', '我', '是', '在', '也', '就', '都', '和', '有', '去', '今天', '啊', '吗', '很', '真', '什么', '但是', '一个', '自己'}
        words = jieba.cut(full_text)
        filtered_words = [w for w in words if len(w) > 1 and w not in stop_words]
        counter = Counter(filtered_words)
        cloud_data = [{"name": k, "value": v} for k, v in counter.most_common(30)]
        return jsonify({"data": cloud_data})
    except Exception as e:
        print(e)
        return jsonify({"data": []})
    finally:
        if conn: conn.close()

@app.route('/report/download')
@login_required
def generate_report():
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            sql = "SELECT * FROM history WHERE user_id = %s AND DATE(create_time) = CURDATE() ORDER BY create_time ASC"
            cursor.execute(sql, (current_user.id,))
            records = cursor.fetchall()

        if not records:
            return "今日暂无数据，请先进行几次分析后再生成报告。", 400

        dates = [r['create_time'].strftime("%H:%M") for r in records]
        scores = [map_emotion_to_score(r['emotion_type']) for r in records]
        emotions = [r['emotion_type'] for r in records]
        most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "无"
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        font_name = 'Helvetica'
        try:
            pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
            font_name = 'SimHei'
        except:
            try:
                pdfmetrics.registerFont(TTFont('SimHei', 'simhei.ttf'))
                font_name = 'SimHei'
            except: pass

        c.setFont(font_name, 24)
        c.drawCentredString(width/2, height - 60, "情绪健康诊断日报")
        c.setFont(font_name, 10)
        c.drawCentredString(width/2, height - 90, f"用户: {current_user.username}  |  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.line(50, height - 100, width - 50, height - 100)

        c.setFont(font_name, 14)
        c.drawString(50, height - 140, "【今日概览】")
        c.setFont(font_name, 12)
        summary = f"您今天共进行了 {len(records)} 次情绪监测。主导情绪为“{most_common_emotion}”。"
        c.drawString(50, height - 165, summary)

        plt.figure(figsize=(8, 3.5))
        plt.plot(dates, scores, marker='o', linestyle='-', color='#6c5ce7', linewidth=2)
        plt.title('Emotion Trend (Today)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.yticks([-2, -1, 0, 1, 2], ['Neg+', 'Neg', 'Neutral', 'Pos', 'Pos+'])
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120)
        img_buffer.seek(0)
        plt.close()
        
        img = ImageReader(img_buffer)
        c.drawImage(img, 50, height - 450, width=500, height=220)

        y = height - 480
        c.setFont(font_name, 14)
        c.drawString(50, y, "【详细记录】")
        y -= 30
        
        c.setFont(font_name, 10)
        c.setFillColor(colors.lightgrey)
        c.rect(50, y-5, 500, 20, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(60, y, "时间")
        c.drawString(120, y, "情绪")
        c.drawString(180, y, "内容/AI建议摘要")
        y -= 25

        for r in records[:12]:
            c.drawString(60, y, r['create_time'].strftime("%H:%M"))
            c.drawString(120, y, r['emotion_type'])
            content = r['input_content'] if r['input_content'] else "[图片]"
            advice = r['healing_text']
            full_text = f"{content} -> {advice}"
            if len(full_text) > 35: full_text = full_text[:35] + "..."
            c.drawString(180, y, full_text)
            y -= 20
        
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(width/2, 30, "Generated by Emotion Intelligence System (Voice + Cloud + Edge)")

        c.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"Report_{current_user.username}_{datetime.now().date()}.pdf", mimetype='application/pdf')

    except Exception as e:
        print(e)
        return f"生成失败: {e}", 500
    finally:
        if conn: conn.close()
# ================= 后台管理系统路由 =================

# 1. 后台首页 (Dashboard)
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.username != 'admin': return redirect(url_for('index'))
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) as count FROM history WHERE DATE(create_time) = CURDATE()")
            today_analyses = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) as count FROM history WHERE emotion_type IN ('危机', '深度治疗')")
            total_crisis = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) as count FROM recommendations")
            total_resources = cursor.fetchone()['count']
            
            cursor.execute("SELECT emotion_type, COUNT(*) as value FROM history GROUP BY emotion_type")
            emotion_distribution = cursor.fetchall()
            
            cursor.execute("""
                SELECT h.create_time, u.username, h.input_content, h.emotion_type 
                FROM history h JOIN users u ON h.user_id = u.id 
                WHERE h.emotion_type IN ('危机', '深度治疗') ORDER BY h.create_time DESC LIMIT 5
            """)
            recent_alerts = cursor.fetchall()
            
        stats = {'total_users': total_users, 'today_analyses': today_analyses, 'total_crisis': total_crisis, 'total_resources': total_resources}
        return render_template('admin_dashboard.html', stats=stats, alerts=recent_alerts, emotion_data=emotion_distribution)
    finally:
        conn.close()

# 2. 用户管理
@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.username != 'admin': return redirect(url_for('index'))
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, email FROM users ORDER BY id DESC")
            users = cursor.fetchall()
        return render_template('admin_users.html', users=users)
    finally:
        conn.close()

# 3. 分析日志记录
@app.route('/admin/logs')
@login_required
def admin_logs():
    if current_user.username != 'admin': return redirect(url_for('index'))
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT h.id, u.username, h.emotion_type, h.input_content, h.healing_text, h.create_time 
                FROM history h JOIN users u ON h.user_id = u.id ORDER BY h.create_time DESC LIMIT 100
            """)
            logs = cursor.fetchall()
        return render_template('admin_logs.html', logs=logs)
    finally:
        conn.close()

# 4. 危机预警中心
@app.route('/admin/crisis')
@login_required
def admin_crisis():
    if current_user.username != 'admin': return redirect(url_for('index'))
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT h.id, u.username, h.emotion_type, h.input_content, h.create_time 
                FROM history h JOIN users u ON h.user_id = u.id 
                WHERE h.emotion_type IN ('危机', '深度治疗') ORDER BY h.create_time DESC
            """)
            alerts = cursor.fetchall()
        return render_template('admin_crisis.html', alerts=alerts)
    finally:
        conn.close()

# 5. 治愈资源库
@app.route('/admin/resources')
@login_required
def admin_resources():
    if current_user.username != 'admin': return redirect(url_for('index'))
    conn = pymysql.connect(**mysql_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, category, content, reason, emotion_type FROM recommendations ORDER BY id DESC")
            resources = cursor.fetchall()
        return render_template('admin_resources.html', resources=resources)
    finally:
        conn.close()

# 6. 系统设置 (静态页面展示用)
@app.route('/admin/settings')
@login_required
def admin_settings():
    if current_user.username != 'admin': return redirect(url_for('index'))
    return render_template('admin_settings.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)