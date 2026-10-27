import os
import json
import requests
import sys
from datetime import datetime, timezone, timedelta
import google.generativeai as genai

# 한국 시간(KST) 설정
kst = timezone(timedelta(hours=9))
now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

# 1. 깃허브 금고에서 API 키 불러오기
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ 에러: GEMINI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# 2. 임시 테스트용 CCTV 이미지
CCTV_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Han_River_Bridge.jpg/800px-Han_River_Bridge.jpg"

def run_analysis():
    try:
        print("CCTV 이미지 다운로드 중...")
        response = requests.get(CCTV_URL, timeout=10)
        response.raise_for_status()
        image_data = {"mime_type": "image/jpeg", "data": response.content}

        # 3. Gemini Vision 모델 호출
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        당신은 한강공원 실시간 상태를 분석하는 정밀 관측 AI입니다.
        제공된 CCTV 이미지를 분석하여 날씨, 가시거리, 혼잡도 상태를 판독하세요.
        반드시 아래 JSON 형식으로만 응답하고, 다른 텍스트나 마크다운(```json)을 절대 포함하지 마세요.
        {
            "weather": "맑음",
            "visibility": "매우 좋음",
            "crowd": "여유",
            "reason": "시야가 깨끗하고 사람이 많지 않음"
        }
        """
        
        print("Gemini Vision AI 분석 요청 중...")
        result = model.generate_content([prompt, image_data])
        result_text = result.text.strip()
        
        # 마크다운 텍스트 제거 보정
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
        sys.exit(1) # 깃허브 액션에 에러를 명시적으로 전달하여 빨간색으로 표시되게 함

if __name__ == "__main__":
    run_analysis()
