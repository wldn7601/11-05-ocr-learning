# 11-05-ocr-learning

---

## OCR 성능 비교

**PaddleOCR과 TrOCR의 성능을 비교한다.**

---

## 성능 비교

- 원본 텍스트와 추출한 텍스트의 정확도
- CER (Character Error Rate) : 문자 단위로 오인식된 문자 개수를 전체 문자수로 나눈 비율
- WER (Word Error Rate) : 단어 단위로 오인식된 단어의 비율

---

## 실행은 compare_ocr.py에서 하면 된다.

- **compare_ocr.py**를 실행하면 images 폴더에서 **nutrition.png** 이미지를 가져와 전처리한다.

  - 이미지를 전처리하는 코드는 **preprocess_image.py** 파일에 함수로 정의되어 있다.
  - 그럼 images 폴더에 전처리된 이미지 **preprocessed_nutrition.png** 가 생성된다.

- 그 후 **compare_ocr.py** 코드에서 OCR 실행 및 평가를 진행한다.
  - OCR 실행 및 평가는 **run_ocr.py** 파일에 함수로 정의되어 있다.
  - 그럼 각 OCR마다 정확도, CER, WER 값을 추출하여 시각화 이미지를 results 폴더안에 저장한다.
