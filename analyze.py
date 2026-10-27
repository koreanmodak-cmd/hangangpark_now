import os
import json
import requests
import sys
from io import BytesIO
from PIL import Image
from datetime import datetime, timezone, timedelta
from google import genai

# 한국 시간(KST) 설정
kst = timezone(timedelta(hours=9))
now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

# 1. 깃허브 금고에서 API 키 불러오기 및 최신 Client 생성
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ 에러: GEMINI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# 2. 테스트용 CCTV 이미지 (Unsplash)
CCTV_URL = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800"

def run_analysis():
    try:
        print("CCTV 이미지 다운로드 중...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(CCTV_URL, headers=headers, timeout=15)
        response.raise_for_status()
        
        # PIL 이미지 객체로 변환
        image = Image.open(BytesIO(response.content))

        # 3. 최신 Gemini 모델(gemini-2.5-flash) 호출
        prompt = """
        당신은 한강공원 실시간 상태를 분석하는 정밀 관측 AI입니다.
        제공된 CCTV 이미지를 분석하여 날씨, 가시거리, 혼잡도 상태를 판독하세요.
        반드시 아래 JSON 형식으로만 응답하고, 다른 텍스트나 마크다운(```json)을 절대 포함하지 마세요.
        {
            "weather": "맑음",
            "visibility": "매우 좋음",
            "crowd": "여유",
            "reason": "시야가 깨끗하고 가시거리가 확보됨"
        }
        """
        
        print("Gemini Vision AI 분석 요청 중...")
        result = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )
        
        result_text = result.text.strip()
        
        # 마크다운 텍스트 보정
        if "```" in result_text:
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(result_text)
        
        # 메타데이터 추가
        data["location"] = "여의도 한강공원"
        data["updated_at"] = now
        
        # 4. data.json 저장
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
            
        print(f"✅ [{now}] 분석 성공! data.json이 정상적으로 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 분석 중 치명적 에러 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()
