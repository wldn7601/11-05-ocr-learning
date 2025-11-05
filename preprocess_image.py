import cv2
import numpy as np
import os

# 일반 이미지 전처리
def preprocess_image(input_path, output_path="./images/preprocessed.png", debug=False):
    # 이미지 열기
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_path}")

    # 블러 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 기울기 보정
    coords = np.column_stack(np.where(blur < 255))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = blur.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(blur, M, (w, h), borderValue=(255, 255, 255))

    # 특징 추출
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(deskewed, -1, kernel)

    # 적용
    binary = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 파일을 경로에 만들기
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, binary)

    if debug:
        print(f"[INFO] 기울기 각도: {angle:.2f}°")
        print(f"[INFO] 전처리 완료 -> {output_path}")

    return output_path


def preprocess_nutrition_label(input_path, output_path="./images/preprocessed_nutrition.png", debug=False):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_path}")

    # 1. 회색조 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 명암 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. 약한 블러 (노이즈 제거)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. 강한 이진화 (배경 제거)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 색 반전 (글자가 검정색일 때만)
    white_ratio = np.mean(binary)
    if white_ratio > 127:  # 배경이 밝으면 반전
        binary = cv2.bitwise_not(binary)

    # 6. 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, binary)

    if debug:
        print(f"[INFO] 전처리 완료 → {output_path}")

    return output_path

