from preprocess_image import preprocess_image, preprocess_nutrition_label
import easyocr
from paddleocr import PaddleOCR
import editdistance
import difflib


# 원본 텍스트랑 정확도 판별
def calculate_accuracy(reference, hypothesis):
    """문자열 전체 유사도 기반 Accuracy"""
    return difflib.SequenceMatcher(None, reference, hypothesis).ratio() * 100

# CER 계산
def calculate_cer(reference, hypothesis):
    """Character Error Rate (편집 거리 기반)"""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    distance = editdistance.eval(ref, hyp)
    return (distance / max(len(ref), 1)) * 100

# WER 계산
def calculate_wer(reference, hypothesis):
    """Word Error Rate (편집 거리 기반)"""
    ref = reference.split()
    hyp = hypothesis.split()
    distance = editdistance.eval(ref, hyp)
    return (distance / max(len(ref), 1)) * 100



# OCR 실행 및 평가
def run_ocr_and_evaluate():
    # === 1. 이미지 전처리 ===
    preprocessed_path = preprocess_nutrition_label(
        './images/nutrition.png',
        "./images/preprocessed_nutrition.png",
        debug=True
    )

    # === 2. Ground Truth (정답 텍스트) ===
    ground_truth = """
    총 내용량 40 g 185 kcal
    나트륨 200 mg 10 %
    지방 7 g 13 %
    탄수화물 15 g 5 %
    트랜스지방 0 g
    당류 7 g 7 %
    포화지방 5 g 33 %
    콜레스테롤 0 mg 0 %
    단백질 15 g 27 %
    """.replace("\n", " ").strip()

    # === 3. EasyOCR 실행 ===
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    easy_result = reader.readtext(preprocessed_path)
    easy_text = " ".join([text for (_, text, _) in easy_result])

    # === 4. PaddleOCR 실행 ===
    ocr = PaddleOCR(lang='korean')
    paddle_result = ocr.ocr(preprocessed_path)
    paddle_text = " ".join([line[1][0] for line in paddle_result[0]])

    # === 5. 평가 지표 계산 ===
    easy_acc = calculate_accuracy(ground_truth, easy_text)
    paddle_acc = calculate_accuracy(ground_truth, paddle_text)

    easy_cer = calculate_cer(ground_truth, easy_text)
    paddle_cer = calculate_cer(ground_truth, paddle_text)

    easy_wer = calculate_wer(ground_truth, easy_text)
    paddle_wer = calculate_wer(ground_truth, paddle_text)

    # === 6. 결과 리턴 ===
    return {
        "ground_truth": ground_truth,
        "easy_text": easy_text,
        "paddle_text": paddle_text,
        "easy_accuracy": easy_acc,
        "paddle_accuracy": paddle_acc,
        "easy_cer": easy_cer,
        "paddle_cer": paddle_cer,
        "easy_wer": easy_wer,
        "paddle_wer": paddle_wer
    }


# 단독 실행 (테스트용)
if __name__ == "__main__":
    result = run_ocr_and_evaluate()

    print("\n=== 📊 EasyOCR 평가 ===")
    print(f"정확도: {result['easy_accuracy']:.2f}%")
    print(f"CER: {result['easy_cer']:.2f}%")
    print(f"WER: {result['easy_wer']:.2f}%")

    print("\n=== 📊 PaddleOCR 평가 ===")
    print(f"정확도: {result['paddle_accuracy']:.2f}%")
    print(f"CER: {result['paddle_cer']:.2f}%")
    print(f"WER: {result['paddle_wer']:.2f}%")
