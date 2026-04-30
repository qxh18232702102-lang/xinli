from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_file
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
import os
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

def get_smart_recommendation(user_text, emotion):
    conn = None
    try:
        conn = pymysql.connect(**mysql_config)
        with conn.cursor() as cursor:
            # 【阶段 1：粗排召回】取出该情绪下的所有方案，不再使用 LIMIT 和 RAND()
            sql = "SELECT category, content, reason FROM recommendations WHERE emotion_type = %s"
            cursor.execute(sql, (emotion,))
            candidates = cursor.fetchall()
            
        if not candidates:
            return []
        if len(candidates) <= 2 or not user_text:
            # 如果候选太少或用户没输入文字（比如图片模式），直接随机返回
            import random
            random.shuffle(candidates)
            return candidates[:2]

        # 【阶段 2：TF-IDF 语义向量重排】
        # 1. 对用户输入进行分词
        user_words = " ".join(jieba.cut(user_text))
        
        # 2. 对所有候选方案（内容+理由）进行分词
        docs = [user_words]
        for c in candidates:
            doc_text = f"{c['content']} {c['reason']}"
            docs.append(" ".join(jieba.cut(doc_text)))
            
        # 3. 转化为 TF-IDF 矩阵并计算余弦相似度
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        # 计算用户输入（第0行）与所有候选方案（第1行及以后）的相似度
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # 4. 将打分附加到候选列表中，并按分数降序排列
        for i, c in enumerate(candidates):
            c['score'] = float(cosine_sim[i])
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 【阶段 3：输出】选取最匹配的 2 个方案
        return candidates[:2]

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

# 🔥🔥🔥 核心人设升级 V3.0 (深度投射版) 🔥🔥🔥
            prompt = f"""
            {history_str}
            【用户当前输入】: "{text}"
            【本地模型初步识别情绪】: "{emotion}"
            
            🔴 你的终极身份：/
            1
            你是一位【深谙潜意【0识投射技术的资深心理咨询师】（结合了荣格流派与CBT）。
            你正在与用户进行深度对话，用户可能正在描述一张OH卡牌、一个梦境、或者一幅画。

            🔴 核心应对策略 (必须严格执行)：
            1. **判断是否在描述意象**：
               - 如果用户在描述一个画面（如“看到一个湖”、“梦见被蛇追”、“画了一棵树”），**绝对禁止**直接给出建议（如“去放松”、“深呼吸”）。
               - **必须进行“现象学追问”**：抓住画面中的 1-2 个细节进行反问，引导用户向内探索。
               - 例如：用户说“有个湖”。你应回：“那个湖的水是死水还是活水？水面下似乎隐藏着什么吗？”

            2. **判断是否在表达情绪**：
               - 如果用户直接说“我很累”，则使用共情 + 启发式提问。

            3. **说话风格**：
               - 温暖、好奇、深邃。
               - 像一个耐心的倾听者，而不是急于解决问题的教练。
               - 使用“我很好奇...”、“这让你想起了...”、“似乎...”这样的引导词。

            请返回 JSON 格式：
            {{
                "cloud_emotion": "精准的情绪词 (如: 平静下的暗涌 / 迷茫)",
                "advice": "这是你要对用户说的话 (100字以内，重点在于【提问】和【引导】，而不是给建议)",
                "rec_category": "推荐类别 (如: 冥想练习/书籍/电影)",
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
    
    # 🔥 关键点：获取前端传来的图片类型（是人脸 face 还是房树人 htp？）
    img_type = request.form.get('img_type', 'face') 

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
            
            # 调用 llm_service.py 里的新函数看图
            vision_result = analyze_htp_with_vision(path)
            
            # 解析结果
            reply = vision_result.get('advice', '分析生成中...')
            emotion = vision_result.get('cloud_emotion', '潜意识整合')
            
            # 保存历史
            save_history("[房树人绘画]", emotion, reply)
            
            # 返回给前端（注意 is_therapy_mode=True，保持金色特效）
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
        # 📸 分支 B：普通人脸模式 (调用本地 FaceEmotionRecognizer)
        # =================================================
        else:
            print("📸 [处理模式] 普通人脸识别")
            
            # 调用本地模型检测表情
            emotion, _ = face_model.predict_emotion(path)
            reply = get_ai_reply(emotion)
            
            # 保存历史
            save_history("[图片分析]", emotion, reply)
            
            # 返回给前端（is_therapy_mode=False，普通绿色特效）
            return jsonify({
                "is_crisis": False, 
                "is_therapy_mode": False,
                "emotion": emotion, 
                "ai_reply": reply, 
                "recommendations": get_recommendation(emotion)
            })

    finally:
        # 清理临时文件
        try: os.remove(path)
        except: pass

@app.route('/get/refresh_recommend', methods=['POST'])
@login_required
def api_refresh():
    data = request.get_json()
    return jsonify({"recommendations": get_recommendation(data.get('emotion', '中性'))})

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