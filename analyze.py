import os
import json
import requests
import sys
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta

# 한국 시간(KST) 설정
kst = timezone(timedelta(hours=9))
now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

DATA_FILE = "data.json"
CCTV_URL = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800"

def analyze_image_locally(img_bytes):
    """
    GitHub Actions CPU 내부에서 작동하는 순수 OpenCV 비전 엔진
    - 가시거리: Gaussian Blur + Laplacian Variance (노이즈 제거 후 엣지 변량 계산)
    - 날씨/조명: HSV 채도(Saturation) & 명도(Value) 히스토그램 분석
    - 혼잡도: Canny Edge Density (객체/차량/인파 윤곽선 밀도 산출)
    """
    # 1. 메모리 바이너리를 OpenCV 마트릭스로 디코딩
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("이미지 디코딩에 실패했습니다.")

    # 2. 색상 공간 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. 핵심 수치 측정
    # A. 가시거리 (노이즈 제거 후 Laplacian 계산)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    
    # B. 날씨 및 조명 (HSV 채도/명도)
    mean_sat = float(np.mean(hsv[:, :, 1]))  # 채도
    mean_val = float(np.mean(hsv[:, :, 2]))  # 명도
    
    # C. 혼잡도 (윤곽선 밀도 비율)
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = float((np.count_nonzero(edges) / edges.size) * 100)

    # 4. 조건별 로컬 룰 엔진 (Rule Engine)
    is_night = mean_val < 50

    if is_night:
        weather = "야간 (공원 조명 작동)"
    elif mean_sat < 40 and mean_val < 130:
        weather = "흐림 / 우천 가능성"
    elif mean_sat < 65:
        weather = "구름 많음"
    else:
        weather = "맑음"

    if laplacian_var > 80:
        visibility = "매우 좋음"
    elif laplacian_var > 35:
        visibility = "보통"
    else:
        visibility = "나쁨 (안개/시야 미흡)"

    if edge_ratio > 8.0:
        crowd = "혼잡"
    elif edge_ratio > 4.0:
        crowd = "보통"
    else:
        crowd = "여유"

    reason = f"로컬 분석 완료 [채도: {mean_sat:.1f}, 밝기: {mean_val:.1f}, 엣지밀도: {edge_ratio:.2f}%]"

    return {
        "location": "여의도 한강공원",
        "weather": weather,
        "visibility": visibility,
        "crowd": crowd,
        "reason": reason,
        "updated_at": now
    }

def run():
    print(f"[{now}] Local Edge-Vision 파이프라인 가동 (0% External API 의존성)")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(CCTV_URL, headers=headers, timeout=10)
        res.raise_for_status()
        
        result_data = analyze_image_locally(res.content)
        print("✅ 로컬 비전 연산 성공!")
        
    except Exception as e:
        print(f"⚠️ 이미지 수집/처리 예외 발생: {e}")
        result_data = {
            "location": "여의도 한강공원",
            "weather": "점검 중",
            "visibility": "점검 중",
            "crowd": "보통",
            "reason": f"CCTV 스트림 연동 대기 중 ({e})",
            "updated_at": now
        }

    # data.json 파일 작성
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump([result_data], f, ensure_ascii=False, indent=2)
        
    print(f"📄 [{DATA_FILE}] 정상 생성 완료.")

if __name__ == "__main__":
    run()
