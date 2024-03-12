# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, UploadFile, status

from app.schemas import OCROut
from app.vision import predictor
from doctr.io import decode_img_as_tensor
import cv2
import numpy as np

router = APIRouter()


@router.post("/", response_model=List[OCROut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(file: UploadFile = File(...)):
    """Runs docTR OCR model to analyze the input image"""
    # img = decode_img_as_tensor(file.file.read())
    data = file.file.read()
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    out = predictor([img])

    return [
        OCROut(box=(*word.geometry[0], *word.geometry[1]), value=word.value, confidence=word.confidence)
        for block in out.pages[0].blocks
        for line in block.lines
        for word in line.words
    ]
