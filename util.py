"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def pil_to_base64(pil_img, format="JPEG", quality=90):
    """
    Convert PIL Image to base64 data URI string
    """
    buffered = BytesIO()
    if pil_img.mode in ("RGBA", "P") and format.upper() == "JPEG":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format, quality=quality)
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

