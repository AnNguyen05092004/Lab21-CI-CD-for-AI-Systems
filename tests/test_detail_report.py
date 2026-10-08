# =============================================================================
# tests/test_detail_report.py - Kiểm thử hàm tạo báo cáo chi tiết (Bonus 3)
#
# Vai trò: đảm bảo các con số TN/FP/FN/TP và precision/recall trong báo cáo
# đúng với một ví dụ nhỏ mà ta tính tay được.
# =============================================================================
from src.detail_report import tao_bao_cao


def test_tao_bao_cao_dem_dung_ma_tran_nham_lan():
    """Với ví dụ 8 mẫu, các ô của ma trận và precision/recall phải khớp tính tay."""
    # Nhãn thật:    0 0 0 1 1 1 1 0
    # Dự đoán:      0 0 1 1 1 0 1 0
    # TN = 3 (mẫu 0, 1, 7), FP = 1 (mẫu 2), FN = 1 (mẫu 5), TP = 3 (mẫu 3, 4, 6)
    y_that = [0, 0, 0, 1, 1, 1, 1, 0]
    du_doan = [0, 0, 1, 1, 1, 0, 1, 0]

    noi_dung = tao_bao_cao(y_that, du_doan)

    assert "TN = 3" in noi_dung
    assert "FP = 1" in noi_dung
    assert "FN = 1" in noi_dung
    assert "TP = 3" in noi_dung
    # Lớp cao: precision = 3 / (3 + 1) = 0.75 và recall = 3 / (3 + 1) = 0.75
    assert "0.7500" in noi_dung
