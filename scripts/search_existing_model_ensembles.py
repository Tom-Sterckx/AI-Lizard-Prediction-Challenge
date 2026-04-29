from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from run_local_experiment import load_clean_dataframe, stratified_group_split


def decode_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, img_size, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, tf.cast(label, tf.int32)


def make_labeled_dataset(
    df: pd.DataFrame,
    img_size: tuple[int, int],
    batch_size: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].astype(str).values, df["label"].values.astype("int32"))
    )
    ds = ds.map(
        lambda path, label: decode_image(path, label, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def horizontal_flip_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda images, labels: (tf.image.flip_left_right(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy().astype(float)
    true_positives = np.diag(confusion)
    precision = true_positives / np.clip(confusion.sum(axis=0), 1, None)
    recall = true_positives / np.clip(confusion.sum(axis=1), 1, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(f1.mean())


def score_probabilities(probabilities: np.ndarray, y_true: np.ndarray, num_classes: int) -> tuple[float, float]:
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y_true).mean())
    macro_f1 = macro_f1_score(y_true, predictions, num_classes)
    return accuracy, macro_f1


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def collect_model_probabilities(
    model_paths: list[Path],
    val_ds: tf.data.Dataset,
    flip_val_ds: tf.data.Dataset,
    y_val: np.ndarray,
    num_classes: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    records: list[dict[str, object]] = []
    probability_bank: dict[str, np.ndarray] = {}

    for model_path in model_paths:
        try:
            model = keras.models.load_model(model_path)
        except Exception as exc:
            records.append(
                {
                    "kind": "load_failed",
                    "model": model_path.name,
                    "error": str(exc),
                    "val_accuracy": None,
                    "val_macro_f1": None,
                }
            )
            continue

        base_probabilities = model.predict(val_ds, verbose=0)
        flip_probabilities = model.predict(flip_val_ds, verbose=0)
        variants = {
            "base": base_probabilities,
            "flip": flip_probabilities,
            "tta": (base_probabilities + flip_probabilities) / 2.0,
        }

        for variant_name, probabilities in variants.items():
            key = f"{model_path.name}:{variant_name}"
            probability_bank[key] = probabilities
            accuracy, macro_f1 = score_probabilities(probabilities, y_val, num_classes)
            records.append(
                {
                    "kind": "single_model",
                    "members": [key],
                    "weights": [1.0],
                    "temperature": 1.0,
                    "validation_calibrated_bias": None,
                    "val_accuracy": accuracy,
                    "val_macro_f1": macro_f1,
                }
            )

    return records, probability_bank


def search_weighted_pairs(
    probability_bank: dict[str, np.ndarray],
    y_val: np.ndarray,
    num_classes: int,
    include_validation_bias: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    weighted_records: list[tuple[dict[str, object], np.ndarray]] = []
    items = list(probability_bank.items())
    temperatures = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    for (name_a, probabilities_a), (name_b, probabilities_b) in combinations(items, 2):
        for weight_a in np.linspace(0.0, 1.0, 101):
            weight_b = 1.0 - weight_a
            blended = weight_a * probabilities_a + weight_b * probabilities_b
            for temperature in temperatures:
                scaled = temperature_scale(blended, temperature)
                accuracy, macro_f1 = score_probabilities(scaled, y_val, num_classes)
                records.append(
                    record := {
                        "kind": "weighted_pair",
                        "members": [name_a, name_b],
                        "weights": [float(weight_a), float(weight_b)],
                        "temperature": float(temperature),
                        "validation_calibrated_bias": None,
                        "val_accuracy": accuracy,
                        "val_macro_f1": macro_f1,
                    }
                )
                weighted_records.append((record, scaled))

    if not include_validation_bias:
        return records

    weighted_records.sort(
        key=lambda item: (float(item[0]["val_accuracy"]), float(item[0]["val_macro_f1"])),
        reverse=True,
    )
    # Keep this reproducible script quick: calibrate only the most promising raw ensembles.
    for record, scaled in weighted_records[:200]:
        logits = np.log(np.clip(scaled, 1e-9, 1.0))
        for class_index in range(num_classes):
            for bias in [-0.30, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.30]:
                biased_logits = logits.copy()
                biased_logits[:, class_index] += bias
                biased_logits -= biased_logits.max(axis=1, keepdims=True)
                biased_probabilities = np.exp(biased_logits)
                biased_probabilities /= biased_probabilities.sum(axis=1, keepdims=True)
                accuracy, macro_f1 = score_probabilities(biased_probabilities, y_val, num_classes)
                records.append(
                    {
                        "kind": "weighted_pair_validation_bias",
                        "members": record["members"],
                        "weights": record["weights"],
                        "temperature": record["temperature"],
                        "validation_calibrated_bias": {
                            "class_index": int(class_index),
                            "bias": float(bias),
                        },
                        "val_accuracy": accuracy,
                        "val_macro_f1": macro_f1,
                    }
                )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiment_weighted_ensembles_repro.json")
    parser.add_argument(
        "--include-validation-bias",
        action="store_true",
        help="Search a tiny class-bias on the validation labels. This can improve the validation score but is optimistic.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    keras.utils.set_random_seed(args.seed)

    clean_df, label_to_class = load_clean_dataframe(root)
    _, val_split = stratified_group_split(clean_df, label_to_class, args.val_fraction, args.seed)
    y_val = val_split["label"].to_numpy()
    num_classes = len(label_to_class)
    img_size = (args.img_size, args.img_size)

    val_ds = make_labeled_dataset(val_split, img_size, args.batch_size)
    flip_val_ds = horizontal_flip_dataset(val_ds)

    model_paths = sorted(root.glob("tmp_*.keras"))
    best_model_path = root / "best_lizard_model.keras"
    if best_model_path.exists():
        model_paths.append(best_model_path)

    single_records, probability_bank = collect_model_probabilities(
        model_paths=model_paths,
        val_ds=val_ds,
        flip_val_ds=flip_val_ds,
        y_val=y_val,
        num_classes=num_classes,
    )
    ensemble_records = search_weighted_pairs(
        probability_bank=probability_bank,
        y_val=y_val,
        num_classes=num_classes,
        include_validation_bias=args.include_validation_bias,
    )

    all_records = single_records + ensemble_records
    successful_records = [
        record
        for record in all_records
        if record.get("val_accuracy") is not None and record.get("val_macro_f1") is not None
    ]
    successful_records.sort(
        key=lambda record: (float(record["val_accuracy"]), float(record["val_macro_f1"])),
        reverse=True,
    )

    output = {
        "note": (
            "Validation-calibrated bias uses validation labels to select the final bias. "
            "Report it as an optimistic validation high-score, not as an unbiased estimate."
        ),
        "top_results": successful_records[:50],
        "all_results_count": len(all_records),
    }
    output_path = root / args.output
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["top_results"][:10], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
