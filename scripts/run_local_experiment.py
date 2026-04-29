from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers


warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_image_lookup(directory: Path) -> dict[str, Path]:
    image_paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return {path.name.casefold(): path for path in image_paths}


def resolve_image_path(image_id: object, lookup: dict[str, Path]) -> Path | None:
    image_id = str(image_id)
    candidates = [image_id, Path(image_id).name]
    if Path(image_id).suffix == "":
        candidates.extend(f"{image_id}{extension}" for extension in sorted(IMAGE_EXTENSIONS))
    for candidate in candidates:
        resolved = lookup.get(candidate.casefold())
        if resolved is not None:
            return resolved
    return None


def average_hash(path: Path, hash_size: int = 16) -> str:
    with Image.open(path) as img:
        gray = ImageOps.exif_transpose(img).convert("L").resize(
            (hash_size, hash_size),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(gray, dtype=np.float32)
    return "".join("1" if pixel > arr.mean() else "0" for pixel in arr.flatten())


def stratified_group_split(
    df: pd.DataFrame,
    label_to_class: dict[int, str],
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_total = int(round(len(df) * val_fraction))
    target_per_label = df["label"].value_counts().sort_index() * val_fraction
    group_rows = []
    for hash_group, subset in df.groupby("hash_group"):
        label_counts = subset["label"].value_counts().sort_index()
        row = {"hash_group": hash_group, "group_size": len(subset)}
        for label in sorted(label_to_class):
            row[f"label_{label}"] = int(label_counts.get(label, 0))
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_df = (
        group_df.sample(frac=1.0, random_state=seed)
        .sort_values("group_size", ascending=False)
        .reset_index(drop=True)
    )

    val_groups: list[str] = []
    val_total = 0
    current_counts = pd.Series(0.0, index=target_per_label.index)

    for row in group_df.itertuples(index=False):
        row_counts = pd.Series(
            {label: getattr(row, f"label_{label}") for label in target_per_label.index},
            dtype=float,
        )
        remaining_deficit = (target_per_label - current_counts).clip(lower=0)
        group_fills_needed_labels = float((np.minimum(row_counts, remaining_deficit)).sum())

        should_add = False
        if val_total < target_total and group_fills_needed_labels > 0:
            should_add = True
        elif val_total + row.group_size <= target_total and len(val_groups) == 0:
            should_add = True

        if should_add:
            val_groups.append(row.hash_group)
            current_counts += row_counts
            val_total += int(row.group_size)

    if not val_groups:
        val_groups = group_df.head(1)["hash_group"].tolist()

    mask = df["hash_group"].isin(val_groups)
    train_split = df[~mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_split = df[mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_split, val_split


def stratified_random_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for _, subset in df.groupby("label", sort=True):
        subset = subset.sample(frac=1.0, random_state=seed)
        n_val = max(1, int(round(len(subset) * val_fraction)))
        val_parts.append(subset.iloc[:n_val])
        train_parts.append(subset.iloc[n_val:])
    train_split = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_split = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_split, val_split


def make_data_augmentation(strength: float) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08 * strength),
            layers.RandomZoom(0.15 * strength),
            layers.RandomContrast(0.15 * strength),
            layers.RandomTranslation(0.05 * strength, 0.05 * strength),
        ],
        name="augmentation",
    )


def load_clean_dataframe(root: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    train_df = pd.read_csv(root / "train.csv")
    train_lookup = build_image_lookup(root / "train")
    train_df["path"] = train_df["id"].map(lambda image_id: resolve_image_path(image_id, train_lookup))
    train_df = train_df[train_df["path"].notna()].reset_index(drop=True)
    train_df["class_name"] = train_df["path"].map(lambda p: Path(p).parent.name)
    label_to_class = (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )

    aspect_ratios = []
    hash_groups = []
    for path in train_df["path"]:
        with Image.open(path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            width, height = rgb.size
        aspect_ratios.append(width / height)
        hash_groups.append(average_hash(path))

    train_df["aspect_ratio"] = aspect_ratios
    train_df["hash_group"] = hash_groups
    hash_label_counts = train_df.groupby("hash_group")["label"].nunique()
    conflicting_hashes = set(hash_label_counts[hash_label_counts > 1].index)

    clean_df = train_df[
        (~train_df["hash_group"].isin(conflicting_hashes))
        & (train_df["aspect_ratio"] >= 0.25)
        & (train_df["aspect_ratio"] <= 4.0)
    ].copy()
    clean_df = clean_df.reset_index(drop=True)
    return clean_df, label_to_class


def build_backbone(name: str, input_shape: tuple[int, int, int], weights: str | None = "imagenet"):
    if name == "efficientnetb0":
        return keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    if name == "efficientnetb1":
        return keras.applications.EfficientNetB1(include_top=False, weights=weights, input_shape=input_shape)
    if name == "efficientnetv2b0":
        return keras.applications.EfficientNetV2B0(include_top=False, weights=weights, input_shape=input_shape)
    if name == "efficientnetv2b1":
        return keras.applications.EfficientNetV2B1(include_top=False, weights=weights, input_shape=input_shape)
    raise ValueError(f"Unsupported backbone: {name}")


def preprocess_for_backbone(name: str, x: tf.Tensor) -> tf.Tensor:
    if name.startswith("efficientnetv2"):
        return keras.applications.efficientnet_v2.preprocess_input(x)
    return keras.applications.efficientnet.preprocess_input(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--backbone", default="efficientnetb0")
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--split-mode", choices=["group", "stratified"], default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-epochs", type=int, default=12)
    parser.add_argument("--fine-layers", type=int, default=40)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--fine-lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--dense-units", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augmentation-strength", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "grayscale"],
        default="rgb",
        help="Use normal RGB images, or convert images to grayscale and repeat the channel to remove color cues.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    keras.utils.set_random_seed(args.seed)
    autotune = tf.data.AUTOTUNE

    clean_df, label_to_class = load_clean_dataframe(root)
    if args.split_mode == "group":
        train_split, val_split = stratified_group_split(clean_df, label_to_class, args.val_fraction, args.seed)
    else:
        train_split, val_split = stratified_random_split(clean_df, args.val_fraction, args.seed)
    val_targets = val_split["label"].to_numpy()
    img_size = (args.img_size, args.img_size)
    augmentation = make_data_augmentation(args.augmentation_strength)
    num_classes = len(label_to_class)
    use_one_hot = args.label_smoothing > 0

    def decode_image(path: tf.Tensor, label: tf.Tensor | None = None):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, img_size, method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image, 0.0, 255.0)
        if args.color_mode == "grayscale":
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.grayscale_to_rgb(image)
        if label is None:
            return image
        label = tf.cast(label, tf.int32)
        if use_one_hot:
            label = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return image, label

    def make_labeled_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            (df["path"].astype(str).values, df["label"].values.astype("int32"))
        )
        ds = ds.map(lambda p, y: decode_image(p, y), num_parallel_calls=autotune)
        if not training:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(len(df), seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size).prefetch(autotune)
        return ds

    train_ds = make_labeled_dataset(train_split, training=True)
    val_ds = make_labeled_dataset(val_split, training=False)

    try:
        backbone = build_backbone(args.backbone, img_size + (3,), weights="imagenet")
    except Exception:
        backbone = build_backbone(args.backbone, img_size + (3,), weights=None)
    backbone.trainable = False

    inputs = keras.Input(shape=img_size + (3,))
    x = augmentation(inputs)
    x = preprocess_for_backbone(args.backbone, x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(args.dropout)(x)
    if args.dense_units > 0:
        regularizer = keras.regularizers.l2(args.weight_decay) if args.weight_decay > 0 else None
        x = layers.Dense(args.dense_units, activation="swish", kernel_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(min(args.dropout + 0.10, 0.70))(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name=f"experiment_{args.name}")

    def build_optimizer(lr: float):
        if args.optimizer == "adamw":
            return keras.optimizers.AdamW(learning_rate=lr, weight_decay=max(args.weight_decay, 1e-5))
        return keras.optimizers.Adam(learning_rate=lr)

    if use_one_hot:
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    else:
        loss_fn = keras.losses.SparseCategoricalCrossentropy()

    class_weights = None
    if args.use_class_weights:
        counts = train_split["label"].value_counts().to_dict()
        total = len(train_split)
        n_classes = len(counts)
        class_weights = {int(label): total / (n_classes * count) for label, count in counts.items()}

    ckpt_path = root / f"tmp_{args.name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.compile(optimizer=build_optimizer(args.head_lr), loss=loss_fn, metrics=["accuracy"])
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    backbone.trainable = True
    for layer in backbone.layers[:-args.fine_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=build_optimizer(args.fine_lr), loss=loss_fn, metrics=["accuracy"])
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model = keras.models.load_model(ckpt_path)
    val_probs = model.predict(val_ds, verbose=0)
    val_preds = val_probs.argmax(axis=1)
    val_accuracy = float((val_preds == val_targets).mean())
    val_confusion = tf.math.confusion_matrix(val_targets, val_preds, num_classes=num_classes).numpy()
    val_tp = np.diag(val_confusion).astype(np.float64)
    val_precision = val_tp / np.clip(val_confusion.sum(axis=0), 1, None)
    val_recall = val_tp / np.clip(val_confusion.sum(axis=1), 1, None)
    val_f1 = 2 * val_precision * val_recall / np.clip(val_precision + val_recall, 1e-12, None)
    val_macro_f1 = float(val_f1.mean())

    flip_val_ds = val_ds.map(
        lambda images, labels: (tf.image.flip_left_right(images), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_probs_flip = model.predict(flip_val_ds, verbose=0)
    val_probs_tta = (val_probs + val_probs_flip) / 2.0
    val_preds_tta = val_probs_tta.argmax(axis=1)
    val_accuracy_tta = float((val_preds_tta == val_targets).mean())
    val_confusion_tta = tf.math.confusion_matrix(val_targets, val_preds_tta, num_classes=num_classes).numpy()
    val_tp_tta = np.diag(val_confusion_tta).astype(np.float64)
    val_precision_tta = val_tp_tta / np.clip(val_confusion_tta.sum(axis=0), 1, None)
    val_recall_tta = val_tp_tta / np.clip(val_confusion_tta.sum(axis=1), 1, None)
    val_f1_tta = 2 * val_precision_tta * val_recall_tta / np.clip(val_precision_tta + val_recall_tta, 1e-12, None)
    val_macro_f1_tta = float(val_f1_tta.mean())

    result = {
        "experiment": args.name,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "split_mode": args.split_mode,
        "color_mode": args.color_mode,
        "head_epochs": args.head_epochs,
        "fine_epochs": args.fine_epochs,
        "fine_layers": args.fine_layers,
        "head_lr": args.head_lr,
        "fine_lr": args.fine_lr,
        "dropout": args.dropout,
        "dense_units": args.dense_units,
        "weight_decay": args.weight_decay,
        "augmentation_strength": args.augmentation_strength,
        "label_smoothing": args.label_smoothing,
        "optimizer": args.optimizer,
        "use_class_weights": args.use_class_weights,
        "best_head_val_accuracy": float(max(history_head.history["val_accuracy"])),
        "best_fine_val_accuracy": float(max(history_fine.history["val_accuracy"])),
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_accuracy_tta": val_accuracy_tta,
        "val_macro_f1_tta": val_macro_f1_tta,
    }
    print("RESULT", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
