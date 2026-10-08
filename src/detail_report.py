# =============================================================================
# src/detail_report.py - Tạo báo cáo chi tiết (Bonus 3)
#
# Vai trò của file trong hệ thống:
#   - Chạy như một bước riêng trong job Train của cicd.yml, ngay sau khi train.py
#     huấn luyện xong.
#   - Đọc mô hình models/model.joblib và tập holdout, rồi ghi ra
#     outputs/detail.txt gồm: ma trận nhầm lẫn, precision và recall của từng lớp.
#   - File detail.txt được upload cùng report.json thành artifact của lần chạy.
#
# Lưu ý: báo cáo dùng ngưỡng mặc định 0.5 (model.predict), đúng với cách VM
# đang phục vụ mô hình.
# =============================================================================
import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def tao_bao_cao(y_that, du_doan) -> str:
    """
    Tạo nội dung báo cáo dạng văn bản từ nhãn thật và nhãn dự đoán.

    Tham số:
        y_that  : nhãn thật (0 = thu nhập thấp, 1 = thu nhập cao).
        du_doan : nhãn mô hình dự đoán, cùng độ dài với y_that.

    Trả về:
        Chuỗi văn bản nhiều dòng (không cần ảnh, đọc được ngay trong log).
    """
    # labels=[0, 1] để bảng luôn đủ 2 hàng 2 cột, kể cả khi mô hình
    # chưa từng đoán ra một lớp nào
    ma_tran = confusion_matrix(y_that, du_doan, labels=[0, 1])
    tn, fp, fn, tp = ma_tran.ravel()

    # Mỗi mảng có 2 phần tử: phần tử đầu cho lớp 0, phần tử sau cho lớp 1
    precision, recall, f1, so_mau = precision_recall_fscore_support(
        y_that, du_doan, labels=[0, 1], zero_division=0
    )

    dong = []
    dong.append(f"BÁO CÁO CHI TIẾT (ngưỡng 0.5, tập holdout {len(y_that)} mẫu)")
    dong.append("")
    dong.append("Ma trận nhầm lẫn (hàng = nhãn thật, cột = mô hình dự đoán):")
    dong.append(f"{'':<24}{'Đoán thấp':>12}{'Đoán cao':>12}")
    dong.append(f"{'Thật sự thấp (0)':<24}{tn:>12}{fp:>12}")
    dong.append(f"{'Thật sự cao (1)':<24}{fn:>12}{tp:>12}")
    dong.append("")
    dong.append(f"TN = {tn}: thu nhập thấp, đoán đúng là thấp")
    dong.append(f"FP = {fp}: thu nhập thấp, nhưng đoán nhầm là cao (báo nhầm)")
    dong.append(f"FN = {fn}: thu nhập cao, nhưng đoán nhầm là thấp (bỏ sót)")
    dong.append(f"TP = {tp}: thu nhập cao, đoán đúng là cao")
    dong.append("")
    dong.append("Chỉ số theo từng lớp:")
    dong.append(f"{'Lớp':<24}{'precision':>12}{'recall':>12}{'f1':>12}{'số mẫu':>10}")
    ten_lop = ["0 (thu nhập thấp)", "1 (thu nhập cao)"]
    for i in range(2):
        dong.append(
            f"{ten_lop[i]:<24}{precision[i]:>12.4f}{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}{so_mau[i]:>10}"
        )
    return "\n".join(dong)


def main(
    model_path: str = "models/model.joblib",
    eval_path: str = "data/holdout.csv",
    out_path: str = "outputs/detail.txt",
):
    """Nạp mô hình, dự đoán trên holdout, ghi báo cáo ra file và in ra màn hình."""
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    du_doan = model.predict(X_eval)
    noi_dung = tao_bao_cao(y_eval, du_doan)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(noi_dung + "\n")

    # In ra để người xem log của GitHub Actions thấy ngay, không cần tải file
    print(noi_dung)


if __name__ == "__main__":
    main()
