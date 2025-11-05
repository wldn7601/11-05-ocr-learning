import os
import matplotlib.pyplot as plt
import numpy as np
from run_ocr import run_ocr_and_evaluate


# 성능 비교 시각화
def visualize_ocr_comparison_bar():
    # 1. OCR 성능 결과 불러오기
    result = run_ocr_and_evaluate()

    # 2. 각 지표 추출
    metrics = ["Accuracy (%)", "CER (%)", "WER (%)"]
    easy_values = [result["easy_accuracy"], result["easy_cer"], result["easy_wer"]]
    paddle_values = [result["paddle_accuracy"], result["paddle_cer"], result["paddle_wer"]]

    # 3. 그래프 설정
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, easy_values, width, label="EasyOCR", color="#4B7BE5")
    plt.bar(x + width / 2, paddle_values, width, label="PaddleOCR", color="#2ECC71")

    # 4. 시각적 설정
    plt.ylabel("Score (%)", fontsize=12)
    plt.title("OCR Performance Comparison", fontsize=14, weight="bold")
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 100)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # 5. 각 막대 위에 값 표시
    for i, v in enumerate(easy_values):
        plt.text(i - width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=10)
    for i, v in enumerate(paddle_values):
        plt.text(i + width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=10)

    # 6. 저장
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # 7. 콘솔 출력
    print(f"[INFO] OCR 성능 비교 그래프 저장 완료 → {save_path}\n")

    print("=== 📘 Ground Truth ===")
    print(result["ground_truth"])
    print("\n=== 🔵 EasyOCR 인식 결과 ===")
    print(result["easy_text"])
    print(f"\n정확도: {result['easy_accuracy']:.2f}% | CER: {result['easy_cer']:.2f}% | WER: {result['easy_wer']:.2f}%")

    print("\n=== 🟢 PaddleOCR 인식 결과 ===")
    print(result["paddle_text"])
    print(f"\n정확도: {result['paddle_accuracy']:.2f}% | CER: {result['paddle_cer']:.2f}% | WER: {result['paddle_wer']:.2f}%")


if __name__ == "__main__":
    visualize_ocr_comparison_bar()
