import requests
import json

# 这是你自己电脑的测试地址
url = "http://127.0.0.1:5000/analyze/text"

# 模拟发送一段文本
payload = {
    "text": "我今天丢了钱，感觉好难过，不想说话。"
}

print(f"正在发送请求到 {url} ...")
print(f"发送内容: {payload['text']}")

try:
    # 发送 POST 请求
    response = requests.post(url, json=payload)
    
    # 打印状态码 (200 表示成功)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ 测试成功！服务器返回如下：")
        print("-" * 30)
        print(f"识别情绪: 【{data['emotion']}】")
        print("-" * 30)
        print("📚 推荐内容：")
        for item in data['recommendations']:
            print(f"- [{item['category']}] {item['title']}: {item['content']}")
        print("-" * 30)
        # print("完整JSON:", json.dumps(data, ensure_ascii=False, indent=2)) # 调试用
    else:
        print("❌ 请求失败:", response.text)

except Exception as e:
    print("❌ 发生错误，请检查后端是否开启:", e)