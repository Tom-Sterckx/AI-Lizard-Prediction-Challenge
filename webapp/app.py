from __future__ import annotations

import base64
import cgi
import html
import io
import json
import os
import sys
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Any


# Register custom layers from the notebook so the ensemble model can be loaded
@tf.keras.utils.register_keras_serializable(package="Lizard")
class HorizontalFlip(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.image.flip_left_right(inputs)


@tf.keras.utils.register_keras_serializable(package="Lizard")
class WeightedProbabilityEnsemble(tf.keras.layers.Layer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_weights = [float(weight) for weight in weights]

    def call(self, inputs):
        if len(inputs) != len(self.ensemble_weights):
            raise ValueError(
                f"Expected {len(self.ensemble_weights)} inputs, got {len(inputs)}"
            )
        weighted_sum = tf.cast(inputs[0], dtype=tf.float32) * self.ensemble_weights[0]
        for i, weight in enumerate(self.ensemble_weights[1:], 1):
            weighted_sum += tf.cast(inputs[i], dtype=tf.float32) * weight
        return weighted_sum

    def get_config(self):
        config = super().get_config()
        config["weights"] = self.ensemble_weights
        return config


@tf.keras.utils.register_keras_serializable(package="Lizard")
class AverageProbabilities(tf.keras.layers.Layer):
    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError("AverageProbabilities expects exactly two tensors.")
        return (tf.cast(inputs[0], tf.float32) + tf.cast(inputs[1], tf.float32)) / 2.0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAIN_DIR = PROJECT_ROOT / "train"
IMG_SIZE = (448, 448)
MODEL_PATTERNS = ("best_lizard_model_*.keras", "*.keras")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MODEL_CACHE: dict[Path, tuple[float, tf.keras.Model]] = {}


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def read_model_metadata(model_path: Path) -> dict:
    metadata_path = model_metadata_path(model_path)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def image_size_for_model(model_path: Path) -> tuple[int, int]:
    metadata = read_model_metadata(model_path)
    image_size = metadata.get("img_size")
    if (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(isinstance(value, int) for value in image_size)
    ):
        return image_size[0], image_size[1]
    return IMG_SIZE


def list_model_paths() -> list[Path]:
    seen: set[Path] = set()
    model_paths: list[Path] = []
    for pattern in MODEL_PATTERNS:
        for path in sorted(ARTIFACTS_DIR.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                model_paths.append(path)
    return model_paths


def class_names(model_path: Path | None = None) -> list[str]:
    if model_path is not None:
        metadata = read_model_metadata(model_path)
        metadata_class_names = metadata.get("class_names")
        if isinstance(metadata_class_names, list) and metadata_class_names:
            return [str(name) for name in metadata_class_names]
    if not TRAIN_DIR.exists():
        return [str(index) for index in range(7)]
    return sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_uploaded_image(image_bytes: bytes, image_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    original = Image.open(io.BytesIO(image_bytes))
    original = ImageOps.exif_transpose(original).convert("RGB")
    resized = original.resize(image_size, Image.Resampling.BICUBIC)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    batch = np.clip(batch, 0.0, 255.0)
    return original, batch


def unwrap_single_output(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def load_cached_model(model_path: Path) -> tf.keras.Model:
    """Load model without strict custom object checking."""
    model_path = model_path.resolve()
    modified_time = model_path.stat().st_mtime
    cached = MODEL_CACHE.get(model_path)
    if cached is not None and cached[0] == modified_time:
        return cached[1]
    
    # Load with safe_mode=False to handle models with custom layers from the notebook
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False
    )
    
    MODEL_CACHE[model_path] = (modified_time, model)
    return model


def call_layer(layer: tf.keras.layers.Layer, x: Any) -> Any:
    try:
        return unwrap_single_output(layer(x, training=False))
    except TypeError:
        return unwrap_single_output(layer(x))


def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    preferred_names = {
        "convnext_tiny",
        "efficientnetb0",
        "efficientnetv2-b0",
        "efficientnetv2-s",
    }
    for layer in model.layers:
        if layer.name in preferred_names and isinstance(layer, tf.keras.Model):
            return layer
    nested_models = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential)
    ]
    if nested_models:
        return nested_models[0]
    raise ValueError("Could not find a nested Keras backbone in this model.")


def output_rank(layer: tf.keras.layers.Layer) -> int | None:
    output = getattr(layer, "output", None)
    if output is None:
        return None
    if isinstance(output, (list, tuple)):
        output = output[0]
    shape = getattr(output, "shape", None)
    if shape is None:
        return None
    return len(shape)


def find_last_spatial_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if output_rank(layer) == 4:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_spatial_layer(layer)
            except ValueError:
                pass
    raise ValueError("Could not find a 4D convolutional feature layer for Grad-CAM.")


def model_slices(model: tf.keras.Model, backbone: tf.keras.Model) -> tuple[list[tf.keras.layers.Layer], list[tf.keras.layers.Layer]]:
    backbone_index = model.layers.index(backbone)
    pre_layers = [
        layer
        for layer in model.layers[1:backbone_index]
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    post_layers = model.layers[backbone_index + 1 :]
    return pre_layers, post_layers


def heatmap_to_color_image(heatmap_array: np.ndarray) -> Image.Image:
    heatmap_array = np.asarray(heatmap_array, dtype=np.float32)
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * heatmap_array - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heatmap_array - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heatmap_array - 1.0), 0.0, 1.0)
    color_heatmap = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(255 * color_heatmap))


def is_ensemble_model(model: tf.keras.Model) -> bool:
    branch_models = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("lizard_")
    ]
    return len(branch_models) >= 2 and any("probability_average" in layer.name for layer in model.layers)


def branch_weights(model: tf.keras.Model, branch_count: int) -> list[float]:
    for layer in model.layers:
        if hasattr(layer, "ensemble_weights"):
            weights = [float(value) for value in getattr(layer, "ensemble_weights")]
            if len(weights) >= branch_count:
                weights = weights[:branch_count]
                total = sum(weights)
                if total > 0:
                    return [weight / total for weight in weights]
    return [1.0 / branch_count] * branch_count


def gradcam_for_model(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    original_size: tuple[int, int],
    target_class_index: int | None = None,
) -> tuple[np.ndarray, int, Image.Image]:
    backbone = find_backbone(model)
    target_layer = find_last_spatial_layer(backbone)
    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
    )
    pre_layers, post_layers = model_slices(model, backbone)
    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = image_tensor
        for layer in pre_layers:
            x = call_layer(layer, x)
        conv_outputs, features = feature_model(x, training=False)
        conv_outputs = unwrap_single_output(conv_outputs)
        features = unwrap_single_output(features)
        tape.watch(conv_outputs)
        predictions = features
        for layer in post_layers:
            predictions = call_layer(layer, predictions)
        predictions = unwrap_single_output(predictions)
        if target_class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        else:
            class_index = int(target_class_index)
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise ValueError("Grad-CAM gradients were empty for the selected model.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np), mode="L")
    heatmap_img = heatmap_img.resize(original_size, Image.Resampling.BICUBIC)
    heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    return predictions.numpy()[0], class_index, heatmap_to_color_image(heatmap_arr)


def ensemble_gradcam_for_model(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    original_size: tuple[int, int],
) -> tuple[np.ndarray, int, Image.Image]:
    predictions = unwrap_single_output(model(tf.convert_to_tensor(image_batch, dtype=tf.float32), training=False))
    probabilities = predictions.numpy()[0]
    class_index = int(np.argmax(probabilities))

    branches = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("lizard_")
    ]
    if not branches:
        return gradcam_for_model(model, image_batch, original_size, target_class_index=class_index)

    weights = branch_weights(model, len(branches))
    flipped_batch = image_batch[:, :, ::-1, :]
    combined_heatmap = np.zeros((original_size[1], original_size[0]), dtype=np.float32)

    for branch_model, weight in zip(branches, weights):
        _, _, original_heatmap = gradcam_for_model(branch_model, image_batch, original_size, target_class_index=class_index)
        _, _, flipped_heatmap = gradcam_for_model(branch_model, flipped_batch, original_size, target_class_index=class_index)
        original_gray = np.asarray(original_heatmap.convert("L"), dtype=np.float32) / 255.0
        flipped_gray = np.asarray(ImageOps.mirror(flipped_heatmap.convert("L")), dtype=np.float32) / 255.0
        combined_heatmap += weight * ((original_gray + flipped_gray) / 2.0)

    combined_min = float(np.min(combined_heatmap))
    combined_max = float(np.max(combined_heatmap))
    if combined_max > combined_min:
        combined_heatmap = (combined_heatmap - combined_min) / (combined_max - combined_min)
    else:
        combined_heatmap = np.zeros_like(combined_heatmap)

    return probabilities, class_index, heatmap_to_color_image(combined_heatmap)


def predict_with_gradcam(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    original_size: tuple[int, int],
) -> tuple[np.ndarray, int, Image.Image]:
    if is_ensemble_model(model):
        return ensemble_gradcam_for_model(model, image_batch, original_size)
    return gradcam_for_model(model, image_batch, original_size)


def predict_without_gradcam(model: tf.keras.Model, image_batch: np.ndarray) -> tuple[np.ndarray, int, Image.Image]:
    predictions = unwrap_single_output(model(tf.convert_to_tensor(image_batch, dtype=tf.float32), training=False))
    probabilities = predictions.numpy()[0]
    class_index = int(np.argmax(probabilities))
    blank_heatmap = Image.new("RGB", (1, 1), color=(0, 0, 0))
    return probabilities, class_index, blank_heatmap


def blend_overlay(original: Image.Image, heatmap: Image.Image, alpha: float = 0.42) -> Image.Image:
    heatmap = heatmap.resize(original.size, Image.Resampling.BICUBIC).convert("RGB")
    return Image.blend(original.convert("RGB"), heatmap, alpha)


def render_page(result: str = "", selected_model: str = "") -> bytes:
    models = list_model_paths()
    options = []
    for model_path in models:
        selected = " selected" if str(model_path) == selected_model else ""
        label = html.escape(str(model_path.relative_to(PROJECT_ROOT)))
        options.append(f'<option value="{html.escape(str(model_path))}"{selected}>{label}</option>')

    model_help = ""
    if not models:
        model_help = (
            '<p class="notice">Geen <code>.keras</code> checkpoint gevonden in '
            '<code>artifacts/</code>. Run eerst de final-training cell in Tom\'s notebook.</p>'
        )

    page = f"""<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lizard Grad-CAM</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Lizard Classifier</h1>
      <form action="/predict" method="post" enctype="multipart/form-data" id="predict-form">
        <label>
          Model
          <select name="model_path" required>
            {''.join(options)}
          </select>
        </label>
        <label>
          Afbeelding
          <input type="file" name="image" accept="image/*" required>
        </label>
        <button type="submit" id="submit-button">
          <span class="button-text">Voorspel en toon Grad-CAM</span>
          <span class="button-loading" aria-hidden="true">
            <span class="spinner"></span>
            Aan het denken
          </span>
        </button>
      </form>
      <p class="loading-note" id="loading-note" role="status" aria-live="polite">
        Model wordt geladen en de afbeelding wordt geanalyseerd. Dit kan even duren.
      </p>
      {model_help}
    </section>
    {result}
  </main>
  <script>
    const form = document.getElementById("predict-form");
    const button = document.getElementById("submit-button");
    if (form && button) {{
      form.addEventListener("submit", () => {{
        button.disabled = true;
        form.classList.add("is-loading");
      }});
    }}
  </script>
</body>
</html>"""
    return page.encode("utf-8")


def render_result(
    model_path: Path,
    original: Image.Image,
    overlay: Image.Image,
    probabilities: np.ndarray,
    predicted_index: int,
    warning: str = "",
) -> str:
    names = class_names(model_path)
    top_indices = np.argsort(probabilities)[::-1][:5]
    rows = []
    for index in top_indices:
        label = names[index] if index < len(names) else str(index)
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{float(probabilities[index]):.3f}</td>"
            "</tr>"
        )
    predicted_label = names[predicted_index] if predicted_index < len(names) else str(predicted_index)
    warning_html = f'<p class="warning">{html.escape(warning)}</p>' if warning else ""
    return f"""
    <section class="result">
      <div class="summary">
        <p class="eyebrow">{html.escape(str(model_path.relative_to(PROJECT_ROOT)))}</p>
        <h2>{html.escape(predicted_label)}</h2>
        <p>Confidence: {float(probabilities[predicted_index]):.3f}</p>
        {warning_html}
      </div>
      <div class="images">
        <figure>
          <img src="{image_to_data_uri(original)}" alt="Uploaded lizard">
          <figcaption>Origineel</figcaption>
        </figure>
        <figure>
          <img src="{image_to_data_uri(overlay)}" alt="Grad-CAM overlay">
          <figcaption>Grad-CAM</figcaption>
        </figure>
      </div>
      <table>
        <thead><tr><th>Klasse</th><th>Score</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


class LizardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/static/styles.css":
            self.serve_static(PROJECT_ROOT / "webapp" / "static" / "styles.css", "text/css")
            return
        self.respond(render_page())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > MAX_UPLOAD_BYTES:
                raise ValueError("Upload is te groot.")

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            model_path = Path(form.getfirst("model_path", "")).resolve()
            allowed_models = {path.resolve() for path in list_model_paths()}
            if model_path not in allowed_models:
                raise ValueError("Ongeldig of ontbrekend modelpad.")

            image_item = form["image"]
            image_bytes = image_item.file.read()
            
            model = load_cached_model(model_path)
            
            # Get the actual input shape from the model
            model_input_shape = model.input_shape
            if model_input_shape and len(model_input_shape) >= 3:
                img_h, img_w = model_input_shape[1], model_input_shape[2]
            else:
                img_h, img_w = image_size_for_model(model_path)
            
            original, image_batch = load_uploaded_image(image_bytes, (img_h, img_w))
            
            warning = ""
            try:
                probabilities, predicted_index, heatmap = predict_with_gradcam(model, image_batch, original.size)
            except Exception as gradcam_error:
                probabilities, predicted_index, heatmap = predict_without_gradcam(model, image_batch)
                warning = (
                    "Voorspelling gelukt, maar Grad-CAM kon niet worden berekend voor dit model. "
                    f"Technische reden: {gradcam_error}"
                )
            overlay = original.copy() if warning else blend_overlay(original, heatmap)
            result = render_result(model_path, original, overlay, probabilities, predicted_index, warning=warning)
            self.respond(render_page(result=result, selected_model=str(model_path)))
        except Exception as exc:
            escaped = html.escape(str(exc))
            result = f'<section class="panel error"><h2>Er ging iets mis</h2><p>{escaped}</p></section>'
            self.respond(render_page(result=result), status=500)

    def serve_static(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    base_port = int(os.environ.get("PORT", "8888"))
    
    class ReuseAddrHTTPServer(ThreadingHTTPServer):
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 0)
            super().server_bind()
    
    # Try to bind to a port, with fallback ports
    port = base_port
    max_attempts = 5
    server = None
    
    for attempt in range(max_attempts):
        try:
            server = ReuseAddrHTTPServer(("127.0.0.1", port), LizardHandler)
            print(f"Lizard Grad-CAM app running at http://127.0.0.1:{port}")
            break
        except OSError as e:
            print(f"Port {port} in use or blocked, trying {port + 1}...")
            port += 1
            if attempt == max_attempts - 1:
                print(f"Could not find an available port after {max_attempts} attempts")
                raise
    
    if server:
        print("Press Ctrl+C to stop.")
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
