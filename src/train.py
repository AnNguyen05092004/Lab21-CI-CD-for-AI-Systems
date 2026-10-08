# =============================================================================
# src/train.py - Huấn luyện mô hình và ghi nhận kết quả vào MLflow
#
# Vai trò của file trong hệ thống:
#   - Bước 1: chạy trên máy cá nhân để thí nghiệm các bộ siêu tham số.
#   - Bước 2 và 3: GitHub Actions chạy file này, rồi đọc outputs/report.json
#     để quyết định có qua cổng chất lượng hay không, và upload
#     models/model.joblib lên cloud để VM phục vụ.
#   - Bonus 5: cảnh báo khi dữ liệu huấn luyện bị lệch tỷ lệ lớp.
#   - Bonus 2: quét ngưỡng quyết định để tìm ngưỡng cho F1 cao nhất.
# =============================================================================
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Ngưỡng chất lượng của lab là f1_score, KHÔNG phải accuracy.
# Lý do: bộ dữ liệu Adult có tỷ lệ lớp 75/25. Một mô hình đoán bừa
# "thu nhập thấp" cho mọi mẫu vẫn đạt accuracy 0.75 mà không học được gì.
F1_THRESHOLD = 0.65

# Bonus 5: tỷ lệ lớp dương (thu nhập > 50K) tham chiếu của bộ dữ liệu gốc,
# và biên độ lệch cho phép (5 điểm phần trăm = 0.05).
TY_LE_DUONG_THAM_CHIEU = 0.248
BIEN_DO_CANH_BAO = 0.05


def kiem_tra_lech_du_lieu(ty_le_duong: float) -> bool:
    """
    Bonus 5: so sánh tỷ lệ lớp dương của dữ liệu huấn luyện với tỷ lệ tham chiếu.

    Trả về True nếu lệch quá 5 điểm phần trăm (và in cảnh báo rõ ràng vào log),
    trả về False nếu bình thường. Hàm chỉ CẢNH BÁO, không dừng pipeline.
    """
    do_lech = abs(ty_le_duong - TY_LE_DUONG_THAM_CHIEU)
    if do_lech > BIEN_DO_CANH_BAO:
        print(
            f"CẢNH BÁO LỆCH DỮ LIỆU: tỷ lệ lớp dương là {ty_le_duong:.1%}, "
            f"lệch {do_lech:.1%} so với mức tham chiếu {TY_LE_DUONG_THAM_CHIEU:.1%} "
            f"(cho phép tối đa {BIEN_DO_CANH_BAO:.0%}). Hãy kiểm tra lại nguồn dữ liệu."
        )
        return True
    print(
        f"Tỷ lệ lớp dương {ty_le_duong:.1%}: bình thường "
        f"(tham chiếu {TY_LE_DUONG_THAM_CHIEU:.1%})."
    )
    return False


def quet_nguong(y_that, xac_suat):
    """
    Bonus 2: thử các ngưỡng từ 0.10 đến 0.90 (bước 0.05) và chọn ngưỡng cho F1 cao nhất.

    Tham số:
        y_that   : nhãn thật (0 hoặc 1) của tập holdout.
        xac_suat : xác suất mô hình dự đoán lớp dương, mỗi mẫu một số trong [0, 1].

    Trả về:
        (ngưỡng_tốt_nhất, f1_tại_ngưỡng_đó). Nếu nhiều ngưỡng bằng điểm nhau,
        lấy ngưỡng nhỏ nhất.
    """
    nguong_tot_nhat, f1_tot_nhat = 0.5, -1.0
    # np.round để tránh sai số số thực kiểu 0.30000000000000004
    for nguong in np.round(np.arange(0.10, 0.9001, 0.05), 2):
        # Xác suất từ ngưỡng trở lên thì gán nhãn 1, ngược lại gán nhãn 0
        du_doan = (xac_suat >= nguong).astype(int)
        # zero_division=0: nếu một ngưỡng không dự đoán mẫu dương nào thì F1 = 0, không báo lỗi
        f1_nguong = f1_score(y_that, du_doan, zero_division=0)
        if f1_nguong > f1_tot_nhat:
            nguong_tot_nhat, f1_tot_nhat = float(nguong), float(f1_nguong)
    return nguong_tot_nhat, f1_tot_nhat


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho GradientBoostingClassifier.
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá (holdout).

    Trả về:
        f1 (float): điểm F1 của lớp dương (thu nhập > 50K) tại ngưỡng mặc định 0.5,
        tính trên tập holdout. Cổng chất lượng vẫn dựa trên con số này.
    """

    # Đọc dữ liệu huấn luyện và dữ liệu đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tách đặc trưng (X) và nhãn (y); cột "target" là nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: kiểm tra phân phối lớp của dữ liệu huấn luyện TRƯỚC khi huấn luyện
    ty_le_duong = float(y_train.mean())
    kiem_tra_lech_du_lieu(ty_le_duong)

    # Mỗi lần gọi train() là một "run" trong MLflow (một dòng trong bảng so sánh)
    with mlflow.start_run():

        # Ghi nhận siêu tham số để sau này biết run nào dùng tham số nào
        mlflow.log_params(params)

        # Huấn luyện; random_state=42 để kết quả lặp lại được giữa các lần chạy
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Chấm điểm trên holdout (dữ liệu mô hình chưa từng thấy).
        # f1_score mặc định tính cho lớp dương (target = 1); KHÔNG truyền average=
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: lấy XÁC SUẤT thay vì nhãn, rồi quét các ngưỡng quyết định
        xac_suat = model.predict_proba(X_eval)[:, 1]
        nguong_tot, f1_tot = quet_nguong(y_eval, xac_suat)

        # Ghi điểm số và mô hình vào MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_rate", ty_le_duong)
        mlflow.log_metric("best_threshold", nguong_tot)
        mlflow.log_metric("best_f1", f1_tot)
        mlflow.sklearn.log_model(model, "model")

        # In kết quả ra màn hình (dòng đầu giữ nguyên định dạng cũ)
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(
            f"Ngưỡng tốt nhất: {nguong_tot:.2f} -> F1 {f1_tot:.4f} "
            f"(ngưỡng mặc định 0.50 -> F1 {f1:.4f})"
        )

        # Lưu chỉ số ra outputs/report.json (GitHub Actions đọc file này ở cổng chất lượng)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "positive_rate": ty_le_duong,
                    "best_threshold": nguong_tot,
                    "best_f1": f1_tot,
                },
                f,
            )

        # Lưu mô hình ra models/model.joblib (sẽ được upload lên cloud ở Bước 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # Trả về f1 để nơi gọi train() (ví dụ unit test) đọc được kết quả
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
