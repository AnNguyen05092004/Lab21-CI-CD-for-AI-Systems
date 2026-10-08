# =============================================================================
# src/serve.py - API suy luan chay tren VM (FastAPI)
#
# Vai tro trong he thong:
#   - Khoi dong: tai model.joblib tu cloud storage ve may.
#   - GET  /healthz : job Release cua GitHub Actions goi de kiem tra server song.
#   - POST /score   : nhan 10 dac trung, tra ve nhan du doan.
# Chay duoi systemd (xem /etc/systemd/system/income-api.service tren VM).
# =============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    # TODO 1: Tao storage.Client() (tu doc GOOGLE_APPLICATION_CREDENTIALS)
    client = storage.Client()

    # TODO 2: Lay bucket va blob tuong ung
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)

    # TODO 3: Tai file model xuong may (dam bao thu muc dich ton tai)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    print("Model da duoc tai xuong tu cloud storage.")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    # TODO 6: Sai so luong dac trung -> loi 400 (loi cua nguoi goi, khong phai server)
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400, detail="Expected 10 features (adult income)"
        )

    # TODO 7: Du doan; model.predict tra ve mang, lay phan tu dau va ep ve int
    pred = int(model.predict([req.features])[0])

    # TODO 8: 0 -> thu nhap thap, 1 -> thu nhap cao
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
