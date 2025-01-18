import pytesseract
import cv2
import re
import pandas as pd
from PIL import Image
import numpy as np

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_business_card_info(image_path):
    # PIL을 사용하여 이미지를 열고 OpenCV 형식으로 변환
    pil_image = Image.open(image_path)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 이미지가 정상적으로 로드되지 않았을 경우
    if image is None:
        print("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
        return None
    
    # OCR로 텍스트 추출
    text = pytesseract.image_to_string(image)
    
    # 정규표현식으로 필요한 정보 필터링
    company_name = re.search(r"(?<=회사명: ).+", text)  # '회사명:' 등 특정 키워드를 찾아서 분류 가능
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.search(r"\b\d{2,3}-\d{3,4}-\d{4}\b", text)
    address = re.search(r"(?<=주소: ).+", text)  # '주소:' 키워드를 활용해 패턴 감지

    # 데이터가 없으면 기본값 설정
    data = {
        "회사명": company_name.group(0) if company_name else "알 수 없음",
        "이메일": email.group(0) if email else "없음",
        "전화번호": phone.group(0) if phone else "없음",
        "주소": address.group(0) if address else "없음"
    }
    return data

# 예제 실행
image_path = r'D:\project\worktable\명함데이터추출\image.jpg'
business_card_data = extract_business_card_info(image_path)

if business_card_data:
    print("추출된 명함 데이터:\n", business_card_data)

    # 엑셀 파일로 저장
    df = pd.DataFrame([business_card_data])
    df.to_excel('명함_데이터.xlsx', index=False)
