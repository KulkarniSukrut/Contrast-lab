
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
import numpy as np
import time

from image_processing import enhance_image


app = FastAPI(
    title="Contrast Lab API",
    description="Image Contrast Enhancement System",
    version="1.0"
)

# Allow our local React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Contrast Lab API",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/enhance")
async def process_image(
    file: UploadFile = File(...),
    method: str = Form("CLAHE"),
    gamma: float = Form(0.7),
    clip_limit: float = Form(2.0),
    tile_grid_size: int = Form(8)
):

    start_time = time.perf_counter()

    # Validate file size (maximum 15 MB)
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(413, "Maximum upload size is 15 MB.")

    try:
        # Decode using Pillow instead of trusting the extension
        pil_image = Image.open(BytesIO(contents))
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

        if pil_image.width > 6000 or pil_image.height > 6000:
            raise HTTPException(
                400, "Image dimensions cannot exceed 6000 x 6000."
            )

        image = np.array(pil_image)

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            400, "Unsupported or invalid image file."
        )

    try:
        # Apply the selected enhancement algorithm
        kwargs = {}

        if method == "CLAHE":
            kwargs = {
                "clip_limit": clip_limit,
                "tile_grid_size": tile_grid_size
            }

        elif method == "Gamma Correction":
            kwargs = {"gamma": gamma}

        enhanced, selected_method = enhance_image(
            image,
            method,
            **kwargs
        )

    except (ValueError, cv_error()) as error:
        raise HTTPException(400, str(error))

    # Encode the enhanced image as PNG
    output = BytesIO()

    Image.fromarray(enhanced).save(output, format="PNG")
    output.seek(0)

    processing_time = (time.perf_counter() - start_time) * 1000

    return StreamingResponse(
        output,
        media_type="image/png",
        headers={
            "X-Enhancement-Method": selected_method,
            "X-Processing-Time-MS": str(round(processing_time, 2)),
            "X-Image-Width": str(enhanced.shape[1]),
            "X-Image-Height": str(enhanced.shape[0]),
            "Content-Disposition": 'attachment; filename="enhanced.png"'
        }
    )


def cv_error():
    # Return the OpenCV error type without requiring another dependency
    import cv2
    return cv2.error
