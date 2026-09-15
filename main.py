"""
Potato Quality Detection API
-----------------------------
FastAPI service that wraps the trained YOLOv8 potato good/bad
detection model behind a simple HTTP endpoint.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then test with:
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@sample.jpg"
"""

import io
import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("potato-api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best.pt"          # swap to "models/best.onnx" if desired
CONFIDENCE_THRESHOLD = 0.25
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE_MB = 10

ALLOWED_VIDEO_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
MAX_VIDEO_SIZE_MB = 200

# Holds the loaded model. Populated once at startup, not per-request.
model_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan: load model once when the server starts, not on every request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s ...", MODEL_PATH)
    try:
        model_state["model"] = YOLO(MODEL_PATH)
        logger.info("Model loaded. Classes: %s", model_state["model"].names)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise
    yield
    model_state.clear()
    logger.info("Model unloaded, shutting down.")


app = FastAPI(
    title="Potato Quality Detection API",
    description="Detects good vs bad potatoes on a conveyor belt from an image.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Simple liveness/readiness check."""
    is_ready = "model" in model_state
    return {"status": "ok" if is_ready else "model not loaded", "model_loaded": is_ready}


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs the potato detection model, and returns
    detected boxes with class label and confidence.
    """
    # --- Input validation ---------------------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Decode image ---------------------------------------------------
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    img_array = np.array(image)

    # --- Run inference ---------------------------------------------------
    model = model_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        results = model.predict(img_array, conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Inference failed.")

    # --- Format response ---------------------------------------------------
    detections = []
    good_count = 0
    bad_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            detections.append(
                {
                    "class": cls_name,
                    "confidence": round(confidence, 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

            if cls_name.lower() == "bad":
                bad_count += 1
            else:
                good_count += 1

    return JSONResponse(
        {
            "filename": file.filename,
            "total_detections": len(detections),
            "good_count": good_count,
            "bad_count": bad_count,
            "detections": detections,
        }
    )


# ---------------------------------------------------------------------------
# Video prediction endpoint — returns an annotated video (boxes drawn in),
# the same style of output your Colab script produces with save=True.
# ---------------------------------------------------------------------------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Accepts a video file, runs detection on every frame, and returns an
    annotated video (boxes + labels drawn on each frame) as a downloadable
    file — equivalent to the Colab `model.predict(save=True)` workflow.
    """
    # --- Input validation ---------------------------------------------------
    if file.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_VIDEO_CONTENT_TYPES)}",
        )

    model = model_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Save the upload to a temp folder first, since YOLO reads video from a
    # file path, not from an in-memory stream.
    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.gettempdir()) / f"potato-video-{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / (file.filename or "input.mp4")

    try:
        size_bytes = 0
        with open(input_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # stream in 1MB chunks
                size_bytes += len(chunk)
                if size_bytes > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video too large. Max allowed: {MAX_VIDEO_SIZE_MB} MB.",
                    )
                out_file.write(chunk)

        # Run inference on every frame, save annotated output video.
        try:
            results = model.predict(
                source=str(input_path),
                conf=CONFIDENCE_THRESHOLD,
                save=True,
                project=str(work_dir),
                name="output",
                verbose=False,
            )
        except Exception as e:
            logger.error("Video inference failed: %s", e)
            raise HTTPException(status_code=500, detail="Video inference failed.")

        # Locate the annotated video ultralytics just wrote out.
        output_dir = work_dir / "output"
        video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))
        if not video_files:
            raise HTTPException(status_code=500, detail="Annotated video was not produced.")
        output_video_path = video_files[0]

        return FileResponse(
            path=output_video_path,
            media_type="video/mp4",
            filename=f"annotated_{file.filename or 'output.mp4'}",
        )

    finally:
        # Note: FileResponse streams the file after this function returns,
        # so we don't delete work_dir here — the OS temp folder will
        # accumulate files across requests. For production, add a cleanup
        # job (e.g. delete folders older than N minutes) or stream + delete
        # via a BackgroundTask.
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)