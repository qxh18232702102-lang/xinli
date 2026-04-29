class EmotionRecommender:
    def __init__(self):
        self.rules = {
            '高兴': ['流行音乐', '轻小说', '喜剧电影'],
            '难过': ['治愈音乐', '励志文章', '温情电影'],
            '愤怒': ['冥想视频', '舒缓音乐'],
            '恐惧': ['安全心理课程'],
            '惊讶': ['新闻推荐'],
            '厌恶': ['情绪疏导短片'],
            '中性': ['热门综艺', '科技资讯']
        }

    def recommend(self, emotion):
        return self.rules.get(emotion, ["暂无推荐"])
