import os
import json
import requests
from datetime import datetime, timezone, timedelta
import google.generativeai as genai

# 한국 시간(KST) 설정
kst = timezone(timedelta(hours=9))
now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

# 1. 깃허브 금고에서 API 키 불러오기
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# 2. 임시 테스트용 CCTV 이미지 (추후 공공데이터 API 주소로 교체)
CCTV_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Han_River_Bridge.jpg/800px-Han_River_Bridge.jpg"

def run_analysis():
    try:
        print("CCTV 이미지 다운로드 중...")
        response = requests.get(CCTV_URL)
        response.raise_for_status()
        image_data = {"mime_type": "image/jpeg", "data": response.content}

        # 3. Gemini Vision 모델 호출
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 견고한 JSON 출력을 위한 정밀 프롬프트
        prompt = """
        당신은 한강공원 실시간 상태를 분석하는 정밀 관측 AI입니다.
        제공된 CCTV 이미지를 분석하여 날씨, 가시거리, 혼잡도 상태를 판독하세요.
        반드시 아래 JSON 형식으로만 응답하고, 다른 텍스트나 마크다운(```json)을 절대 포함하지 마세요.
        {
            "weather": "날씨 상태 (예: 맑음, 흐림, 비 옴, 야간 맑음)",
            "visibility": "가시거리 상태 (예: 매우 좋음, 나쁨)",
            "crowd": "혼잡도 상태 (예: 여유, 혼잡, 매우 혼잡)",
            "reason": "위와 같이 판단한 객관적 근거 1문장"
        }
        """
        
        result = model.generate_content([prompt, image_data])
        result_text = result.text.strip()
        
        # AI가 마크다운을 붙여서 응답할 경우를 대비한 안전장치(예외 처리)
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
            
        # JSON 텍스트를 파이썬 데이터로 변환
        data = json.loads(result_text)
        
        # 메타데이터(위치, 업데이트 시간) 추가
        data["location"] = "여의도 한강공원 (테스트)"
        data["updated_at"] = now
        
        # 4. 분석 결과를 data.json 파일로 저장
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
            
        print(f"[{now}] 분석 성공! data.json 파일이 업데이트되었습니다.")
        
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    run_analysis()
