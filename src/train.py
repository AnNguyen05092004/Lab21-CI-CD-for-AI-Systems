# =============================================================================
# src/train.py - Huan luyen mo hinh + ghi nhan ket qua vao MLflow
#
# Vai tro trong he thong:
#   - Buoc 1: chay tren may ca nhan de thi nghiem tham so.
#   - Buoc 2/3: GitHub Actions chay file nay, doc outputs/report.json de
#     quyet dinh quality gate, va upload models/model.joblib len cloud.
# =============================================================================
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y); cot "target" la nhan
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Moi lan goi train() = mot "run" trong MLflow (mot dong trong bang so sanh)
    with mlflow.start_run():

        # TODO 3: Ghi nhan sieu tham so de sau nay biet run nao dung tham so nao
        mlflow.log_params(params)

        # TODO 4: Huan luyen. random_state=42 de ket qua lap lai duoc
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Cham diem tren holdout (du lieu mo hinh chua tung thay)
        # f1_score mac dinh tinh cho lop duong (target=1), KHONG truyen average=
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # TODO 6: Ghi diem + luu mo hinh vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 8: Luu metrics ra outputs/report.json (GitHub Actions doc file nay)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f)

        # TODO 9: Luu mo hinh ra models/model.joblib (se upload len cloud o Buoc 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1 de ham goi (vi du unit test) doc duoc ket qua
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
