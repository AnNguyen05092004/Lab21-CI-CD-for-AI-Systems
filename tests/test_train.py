# =============================================================================
# tests/test_train.py - Kiểm thử tự động cho hàm train()
#
# Vai trò của file trong hệ thống:
#   - Đây là "cổng đầu tiên" của pipeline CI/CD (job 1: Unit Test). Nếu test
#     hỏng, GitHub Actions dừng ngay: không huấn luyện thật, không triển khai.
#   - Test chạy trên dữ liệu GIẢ sinh ngẫu nhiên trong bộ nhớ, nên không cần
#     kết nối cloud hay chạy `dvc pull`.
# =============================================================================
import os
import json
import numpy as np
import pandas as pd
from src.train import train, kiem_tra_lech_du_lieu, quet_nguong

# 10 đặc trưng đầu vào, đúng thứ tự mà mô hình và API /score sử dụng
FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tạo bộ dữ liệu nhỏ có cùng cấu trúc (schema) với bộ Adult để dùng trong test.

    `tmp_path` do pytest cấp: một thư mục tạm, tự xóa sau khi test kết thúc.
    Dữ liệu là số ngẫu nhiên, nên test này chỉ kiểm tra "code chạy đúng",
    chứ không kiểm tra "mô hình giỏi hay dở".
    """
    # Bộ sinh số ngẫu nhiên với seed cố định -> mỗi lần chạy ra cùng một dữ liệu
    rng = np.random.default_rng(0)
    n = 200

    # 200 dòng x 10 cột, mỗi giá trị ngẫu nhiên trong khoảng [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))

    # Nhãn chỉ có HAI lớp (0 và 1), nên cận trên là 2 (không bao gồm 2)
    y = rng.integers(0, 2, size=n)

    # Ghép thành bảng dữ liệu, thêm cột "target" là nhãn
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # 160 dòng đầu để huấn luyện, 40 dòng cuối làm tập holdout (đánh giá)
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path, monkeypatch):
    """Kiểm tra hàm train() trả về một số thực nằm trong khoảng [0.0, 1.0]."""
    # Chuyển thư mục làm việc sang thư mục tạm, để các file outputs/, models/,
    # mlruns/ do train() tạo ra không làm bẩn thư mục dự án của bạn
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)

    # Dùng siêu tham số rất nhỏ để test chạy nhanh
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path, monkeypatch):
    """Kiểm tra file outputs/report.json được tạo và có đủ hai chỉ số."""
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    # File phải tồn tại, và bên trong phải có cả f1_score lẫn accuracy
    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report
    assert "positive_rate" in report
    assert "best_threshold" in report
    assert "best_f1" in report


def test_model_file_created(tmp_path, monkeypatch):
    """Kiểm tra file models/model.joblib được tạo sau khi huấn luyện."""
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.joblib")


def test_canh_bao_lech_du_lieu():
    """Bonus 5: lệch quá 5 điểm phần trăm thì cảnh báo, trong biên độ thì không."""
    # 50% so với mức tham chiếu 24.8%: lệch hơn 25 điểm -> phải cảnh báo
    assert kiem_tra_lech_du_lieu(0.50) is True
    # 25% gần sát 24.8% -> bình thường
    assert kiem_tra_lech_du_lieu(0.25) is False


def test_quet_nguong_chon_nguong_tot():
    """Bonus 2: với xác suất tách bạch hai lớp, F1 tốt nhất phải bằng 1.0."""
    # Hai mẫu âm có xác suất thấp (0.1; 0.2), hai mẫu dương có xác suất cao (0.8; 0.9)
    y_that = pd.Series([0, 0, 1, 1])
    xac_suat = np.array([0.1, 0.2, 0.8, 0.9])
    nguong, f1_tot = quet_nguong(y_that, xac_suat)
    assert f1_tot == 1.0
    # Ngưỡng tốt phải nằm giữa hai nhóm xác suất
    assert 0.2 < nguong <= 0.8
