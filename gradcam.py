"""
Explainable AI (XAI): Grad-CAM Heatmap Visualization
Gradient-weighted Class Activation Mapping for Pneumonia Detection CNN.
Includes:
- Target-class gradient computation (explaining PREDICTED class: PNEUMONIA or NORMAL)
- Border artifact exclusion (~5% margin ignored during normalization)
- Heat-proportional alpha blending (values < 0.3 fully transparent)
- Anatomical zone activation calculation across 6 image zones
"""

import numpy as np
import tensorflow as tf
import matplotlib
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from util import pil_to_base64


def get_last_conv_layer_name(model):
    """
    Find the name of the last convolutional layer in the Keras model.
    For this CNN, the layer is 'conv2d_58'.
    """
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def create_gradcam_model(model, last_conv_layer_name=None):
    """
    Build a multi-output model that returns:
    [last_conv_layer_activations, final_model_prediction]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )


def compute_gradcam(orig_pil_img, model, grad_model=None, alpha_max=0.85):
    """
    Generates the Grad-CAM heatmap and overlays it onto the original chest X-ray.

    VIVA / ARCHITECTURE NOTES:
    1. Preprocessing:
       - Uses exactly target_size=(64, 64), 3 RGB channels, and preprocess_input(mode='tf')
         which scales pixel intensities to [-1, 1].
    2. Predicted Class Target:
       - For Pneumonia (score >= 0.5): loss = score, showing evidence FOR Pneumonia.
       - For Normal (score < 0.5): loss = (1 - score), showing evidence FOR Normal clear lung fields.
    3. Border Artifact Removal:
       - The outer ~5% margin is excluded when calculating the maximum activation for normalization.
    4. Alpha Proportionality:
       - Activations < 0.3 are 100% transparent (no muddy tint over healthy structures).
       - Activations >= 0.3 ramp smoothly from 0 to alpha_max.
    5. Zone Analysis:
       - Image is divided into 6 zones (upper/middle/lower x image-left/image-right).
       - Reports the zone with highest mean activation without using the word 'opacity'.
    """
    orig_rgb = orig_pil_img.convert("RGB")
    orig_w, orig_h = orig_rgb.size

    # Exactly matching /predict preprocessing
    img_resized = orig_rgb.resize((64, 64))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode="tf")

    if grad_model is None:
        grad_model = create_gradcam_model(model)

    # -------------------------------------------------------------------------
    # Target Class Gradient Calculation:
    # Evaluate predicted class to ensure heatmap explains what drove the prediction.
    # -------------------------------------------------------------------------
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        raw_score_tensor = predictions[:, 0]
        # In graph execution or eager mode, compute loss based on predicted class
        # (TensorFlow condition)
        is_pneu_pred = tf.greater_equal(raw_score_tensor[0], 0.5)
        loss = tf.where(is_pneu_pred, raw_score_tensor, 1.0 - raw_score_tensor)

    score = float(predictions[0, 0].numpy())
    if score >= 0.5:
        driving_class = "PNEUMONIA"
        prediction = "PNEUMONIA"
        confidence = round(score * 100, 1)
    else:
        driving_class = "NORMAL"
        prediction = "NORMAL"
        confidence = round((1.0 - score) * 100, 1)

    pneumonia_prob = round(score * 100, 1)
    normal_prob = round((1.0 - score) * 100, 1)

    # Compute gradients of target class loss w.r.t. convolutional activations
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients (GAP weights)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features that have a POSITIVE influence on target class
    heatmap = tf.maximum(heatmap, 0.0)

    # -------------------------------------------------------------------------
    # Border Artifact Removal:
    # Ignore the outer ~5% border when calculating maximum activation for normalization.
    # -------------------------------------------------------------------------
    H, W = heatmap.shape
    b_h = max(1, int(round(H * 0.05)))
    b_w = max(1, int(round(W * 0.05)))
    inner_region = heatmap[b_h:H - b_h, b_w:W - b_w]
    max_inner = tf.math.reduce_max(inner_region)

    if max_inner > 0:
        heatmap_norm = heatmap / (max_inner + 1e-10)
    else:
        heatmap_norm = heatmap

    heatmap_norm = tf.clip_by_value(heatmap_norm, 0.0, 1.0).numpy()

    # -------------------------------------------------------------------------
    # Computed Zone Activation:
    # 6 zones: (upper, middle, lower thirds) x (image-left, image-right halves)
    # -------------------------------------------------------------------------
    h3 = H // 3
    w2 = W // 2
    zones = {
        "upper zone, image-left": heatmap_norm[0:h3, 0:w2],
        "upper zone, image-right": heatmap_norm[0:h3, w2:W],
        "middle zone, image-left": heatmap_norm[h3:2 * h3, 0:w2],
        "middle zone, image-right": heatmap_norm[h3:2 * h3, w2:W],
        "lower zone, image-left": heatmap_norm[2 * h3:H, 0:w2],
        "lower zone, image-right": heatmap_norm[2 * h3:H, w2:W],
    }
    zone_means = {k: float(np.mean(v)) for k, v in zones.items()}
    highest_zone = max(zone_means, key=zone_means.get)
    attention_summary = f"Highest activation: {highest_zone}"

    # -------------------------------------------------------------------------
    # Heat-Proportional Alpha Overlay:
    # Resize heatmap to original image resolution.
    # Values < 0.3 are completely transparent (alpha = 0).
    # Values >= 0.3 scale proportionally up to alpha_max.
    # -------------------------------------------------------------------------
    heat_pil = Image.fromarray(heatmap_norm).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heat_full = np.array(heat_pil, dtype=np.float32)

    # Alpha map: 0 below 0.3, smooth ramp from 0.3 to 1.0
    alpha_map = np.where(heat_full < 0.3, 0.0, (heat_full - 0.3) / 0.7 * alpha_max)
    alpha_map = np.clip(alpha_map, 0.0, 1.0)[..., np.newaxis]

    # Map heatmap intensities to Jet RGB palette
    cmap = matplotlib.colormaps["jet"]
    heat_uint8 = np.uint8(np.clip(heat_full * 255, 0, 255))
    jet_lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.float32)
    jet_rgb = jet_lut[heat_uint8]

    # Blended superimposed image
    orig_arr = np.array(orig_rgb, dtype=np.float32)
    blended = jet_rgb * alpha_map + orig_arr * (1.0 - alpha_map)
    overlay_pil = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # Transparent PNG heatmap layer for opacity blending
    rgba_heatmap = np.concatenate([jet_rgb, alpha_map * 255], axis=-1).astype(np.uint8)
    jet_rgba_pil = Image.fromarray(rgba_heatmap, mode="RGBA")

    # -------------------------------------------------------------------------
    # Model Confidence Band:
    # -------------------------------------------------------------------------
    if score >= 0.85 or score <= 0.15:
        confidence_band = "High Confidence"
        confidence_band_class = "band-high"
    elif score >= 0.70 or score <= 0.30:
        confidence_band = "Moderate Confidence"
        confidence_band_class = "band-moderate"
    else:
        confidence_band = "Low Confidence (Borderline)"
        confidence_band_class = "band-low"

    return {
        "prediction": prediction,
        "score": round(score, 4),
        "confidence": confidence,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
        "confidence_band": confidence_band,
        "confidence_band_class": confidence_band_class,
        "driving_class": driving_class,
        "driving_caption": f"Regions driving prediction: {driving_class}",
        "overlay_base64": pil_to_base64(overlay_pil, format="JPEG"),
        "heatmap_base64": pil_to_base64(jet_rgba_pil, format="PNG"),
        "original_base64": pil_to_base64(orig_rgb, format="JPEG"),
        "attention_summary": attention_summary,
    }
