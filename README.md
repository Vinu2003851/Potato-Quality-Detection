# Potato Quality Detection API

FastAPI service wrapping the trained YOLOv8 potato good/bad detection model.

## Setup

1. Create the models folder and drop your downloaded weights in:
   ```
   potato-api/
   ├── main.py
   ├── requirements.txt
   └── models/
       └── best.pt      <- your downloaded weights go here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open the auto-generated interactive docs (Swagger UI):
   ```
   http://localhost:8000/docs
   ```
   You can upload a test image directly from this page — no curl needed.

## Endpoints

### `GET /health`
Quick check that the server is up and the model loaded successfully.

### `POST /predict`
Upload an image, get back detections.

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@potato_frame.jpg"
```

**Example response:**
```json
{
  "filename": "potato_frame.jpg",
  "total_detections": 12,
  "good_count": 10,
  "bad_count": 2,
  "detections": [
    {
      "class": "Potato",
      "confidence": 0.91,
      "box": {"x1": 120.4, "y1": 88.2, "x2": 210.7, "y2": 175.9}
    }
  ]
}
```

## Notes

- The model is loaded **once** at startup (see the `lifespan` function in `main.py`), not on every request — this is what keeps inference fast under load.
- Swap `MODEL_PATH` in `main.py` from `models/best.pt` to `models/best.onnx` once you've validated your ONNX export matches PyTorch output, to get the CPU speed benefit from PO 20.2.
- `MAX_FILE_SIZE_MB` and `ALLOWED_CONTENT_TYPES` are the input validation guardrails required by PO 20.1 — adjust as needed.
- This currently accepts a single **image** per request. If you want to feed it video frames from your conveyor belt footage directly, you'd loop over extracted frames client-side and call `/predict` per frame (or extend the endpoint to accept video — worth doing as a follow-up, not part of the base PO 20.1 spec).
