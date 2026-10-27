import os
import json
import requests
import sys
import time
import base64
from datetime import datetime, timezone, timedelta

# 한국 시간(KST) 설정
kst = timezone(timedelta(hours=9))
now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

# 1. API 키 확인
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("❌ 에러: GEMINI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# 2. 테스트용 이미지
CCTV_URL = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800"

def run_analysis():
    try:
        print("CCTV 이미지 다운로드 중...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(CCTV_URL, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 이미지를 텍스트(Base64)로 변환하여 구글에 직접 전송할 준비
        img_b64 = base64.b64encode(response.content).decode("utf-8")

        # 3. 구글 Gemini REST API 직접 호출 (라이브러리 미사용)
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={API_KEY}"
        
        prompt = """
        당신은 한강공원 실시간 상태를 분석하는 정밀 관측 AI입니다.
        제공된 이미지를 분석하여 날씨, 가시거리, 혼잡도 상태를 판독하세요.
        반드시 JSON 형식으로만 응답해야 합니다.
        """
        
        # JSON 요청 구조체
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }],
            # AI가 무조건 JSON 형태로만 답변하도록 강제 (오류 방지)
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        print("Gemini Vision AI 직접 통신 요청 중 (gemini-3.8-flash)...")
        max_retries = 3
        result_text = ""
        
        for attempt in range(max_retries):
            try:
                # 💡 핵심: 20초 하드 타임아웃 설정 (7분 무한루프 원천 차단)
                res = requests.post(
                    api_url, 
                    json=payload, 
                    headers={'Content-Type': 'application/json'}, 
                    timeout=20
                )
                
                res.raise_for_status()  # 400, 500번대 에러 발생 시 즉시 except로 던짐
                
                # 성공 시 결과 추출
                result_json = res.json()
                result_text = result_json['candidates'][0]['content']['parts'][0]['text']
                break
                
            except Exception as api_err:
                print(f"⚠️ API 응답 지연 또는 서버 에러 (시도 {attempt+1}/{max_retries}): {api_err}")
                if attempt < max_retries - 1:
                    print("🔄 10초 후 재시도합니다...")
                    time.sleep(10)
                else:
                    raise Exception("구글 서버가 완전히 혼잡하여 분석에 실패했습니다.")

        # 4. 데이터 저장
        data = json.loads(result_text)
        data["location"] = "여의도 한강공원"
        data["updated_at"] = now
        
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
            
        print(f"✅ [{now}] 분석 성공! data.json이 정상적으로 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 분석 중 치명적 에러 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()
