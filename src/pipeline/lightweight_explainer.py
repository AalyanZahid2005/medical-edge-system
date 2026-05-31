import base64
import io
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def generate_lightweight_heatmap_base64(image_path, size=(384, 384)):
    """
    Dependency-free visual explanation fallback.

    Uses PIL only:
    - resizes uploaded image
    - computes edge/intensity saliency
    - overlays a red heatmap-like mask
    - returns PNG base64 string for API/web display

    This is a lightweight explainability preview, not a true backprop Grad-CAM.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img = img.resize(size)

        gray = ImageOps.grayscale(img)

        # Edge-based saliency approximation
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2))

        # Enhance contrast so attention map is visible
        edges = ImageOps.autocontrast(edges)
        edges = ImageEnhance.Contrast(edges).enhance(2.0)

        # Build red heatmap layer
        red = Image.new("RGB", size, (255, 40, 20))
        heat = Image.composite(red, Image.new("RGB", size, (0, 0, 0)), edges)

        # Blend heatmap over original image
        overlay = Image.blend(img, heat, alpha=0.38)

        out = io.BytesIO()
        overlay.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")

    except Exception as e:
        print("[LIGHTWEIGHT EXPLAINER WARNING]", e)
        return None
