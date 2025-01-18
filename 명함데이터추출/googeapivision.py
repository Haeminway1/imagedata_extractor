from google.cloud import vision
import re
import io

def extract_business_card_info(image_path):
    # Google Cloud Vision 클라이언트 생성
    client = vision.ImageAnnotatorClient()

    # 이미지 파일 로드
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # 텍스트 추출 요청
    response = client.document_text_detection(image=image)
    text_annotations = response.text_annotations

    if not text_annotations:
        print("텍스트를 찾을 수 없습니다.")
        return None

    # 추출된 전체 텍스트
    full_text = text_annotations[0].description
    print("추출된 전체 텍스트:\n", full_text)

    # 정규표현식을 사용하여 필요한 정보 추출
    company_name = re.search(r"회사[:\s]*([^\n]+)", full_text)  # '회사' 키워드가 있을 경우
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", full_text)
    phone = re.search(r"\b\d{2,3}-\d{3,4}-\d{4}\b", full_text)
    address = re.search(r"주소[:\s]*([^\n]+)", full_text)  # '주소' 키워드가 있을 경우

    # 추출된 데이터 정리
    data = {
        "회사명": company_name.group(1) if company_name else "알 수 없음",
        "이메일": email.group(0) if email else "없음",
        "전화번호": phone.group(0) if phone else "없음",
        "주소": address.group(1) if address else "없음"
    }

    return data

# 예제 실행
image_path = '명함 이미지 경로를 입력하세요'
business_card_data = extract_business_card_info(image_path)

if business_card_data:
    print("추출된 명함 데이터:\n", business_card_data)
