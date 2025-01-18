from PIL import Image
import cv2
import numpy as np

image_path = r"D:\project\worktable\명함데이터추출\image.jpg"

# PIL을 사용하여 이미지를 열고 OpenCV 형식으로 변환
pil_image = Image.open(image_path)
image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

if image is None:
    print("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
else:
    print("이미지 로드 성공.")
