import json
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self, data_path='cbt_data.json'):
        self.data_path = data_path
        self.documents = []
        self.contents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """加载知识库并进行预处理（分词+向量化）"""
        if not os.path.exists(self.data_path):
            print("⚠️ 警告：未找到知识库文件，RAG 功能将不可用。")
            return

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 准备语料
            self.contents = [item['content'] for item in data]
            # 对内容进行分词，用于 TF-IDF 计算
            self.documents = [" ".join(jieba.cut(item['topic'] + " " + item['content'])) for item in data]
            
            # 初始化向量化器 (这是"学术"的部分：Vector Space Model)
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print(f"✅ RAG 引擎已就绪，加载了 {len(self.contents)} 条 CBT 专业知识。")
            
        except Exception as e:
            print(f"❌ 知识库加载失败: {e}")

    def search(self, query, top_k=1):
        """检索最相关的知识"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return None

        # 1. 对用户输入进行分词
        query_cut = " ".join(jieba.cut(query))
        
        # 2. 转化为向量
        query_vec = self.vectorizer.transform([query_cut])
        
        # 3. 计算余弦相似度 (Cosine Similarity)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 4. 找到匹配度最高的索引
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        # 设定一个阈值，如果相关性太低（比如在瞎聊），就不硬凑专业知识
        if best_score < 0.1: 
            return None
            
        print(f"🔍 [RAG检索] 用户输入: {query[:10]}... | 匹配知识点: {self.contents[best_idx][:10]}... | 相似度: {best_score:.4f}")
        return self.contents[best_idx]

# 单例模式，方便外部调用
rag_engine = SimpleRAG()