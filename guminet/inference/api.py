"""Optional local demo: uvicorn factory or create_app(checkpoint)."""

import io
import threading

from PIL import Image, UnidentifiedImageError

from . import GumiPredictor


def create_app(checkpoint, device="auto", max_upload_bytes=5 * 1024 * 1024, max_pixels=20_000_000):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from starlette.concurrency import run_in_threadpool

    predictor = GumiPredictor.from_checkpoint(checkpoint, device)
    lock = threading.Lock()
    app = FastAPI(title="GumiNet experimental demo")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info():
        return dict(
            architecture=predictor.metadata["model_config"],
            classes=predictor.classes,
            input_size=predictor.metadata["input_size"],
            device=str(predictor.device),
        )

    def classify(payload):
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.width * image.height > max_pixels:
                    raise HTTPException(413, "Image pixel limit exceeded")
                if image.format not in ("JPEG", "PNG", "WEBP"):
                    raise HTTPException(415, "Only JPEG, PNG and WebP are supported")
                image.load()
                with lock:
                    return predictor.predict(image)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Invalid image") from exc

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(415, "Unsupported content type")
        payload = await file.read(max_upload_bytes + 1)
        await file.close()
        if len(payload) > max_upload_bytes:
            raise HTTPException(413, "Upload size limit exceeded")
        return await run_in_threadpool(classify, payload)

    return app
