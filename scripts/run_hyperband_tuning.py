from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from run_local_experiment import (
    build_backbone,
    load_clean_dataframe,
    make_data_augmentation,
    preprocess_for_backbone,
    stratified_group_split,
)


warnings.filterwarnings("ignore")


def macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy().astype(np.float64)
    true_positives = np.diag(confusion)
    precision = true_positives / np.clip(confusion.sum(axis=0), 1, None)
    recall = true_positives / np.clip(confusion.sum(axis=1), 1, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(f1.mean())


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)


def compute_knn_probabilities(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    temperature: float,
    num_classes: int,
) -> np.ndarray:
    similarities = query_embeddings @ train_embeddings.T
    top_idx = np.argpartition(similarities, -k, axis=1)[:, -k:]
    top_similarities = np.take_along_axis(similarities, top_idx, axis=1)
    order = np.argsort(top_similarities, axis=1)[:, ::-1]
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_similarities = np.take_along_axis(top_similarities, order, axis=1)
    weights = np.exp(top_similarities / temperature)
    labels = train_labels[top_idx]
    probabilities = np.zeros((query_embeddings.shape[0], num_classes), dtype=np.float64)
    for class_index in range(num_classes):
        probabilities[:, class_index] = (weights * (labels == class_index)).sum(axis=1)
    return probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)


def search_embedding_blend(
    model: keras.Model,
    train_eval_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    train_labels: np.ndarray,
    y_val: np.ndarray,
    val_probabilities_tta: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    feature_model = keras.Model(model.input, model.get_layer("embedding_features").output)
    train_embeddings = l2_normalize(feature_model.predict(train_eval_ds, verbose=0))
    val_embeddings = l2_normalize(feature_model.predict(val_ds, verbose=0))

    best: dict[str, object] = {
        "family": "softmax_tta",
        "val_accuracy": float((val_probabilities_tta.argmax(axis=1) == y_val).mean()),
        "val_macro_f1": macro_f1_from_predictions(y_val, val_probabilities_tta.argmax(axis=1), num_classes),
    }

    for k in [7, 11, 15, 21, 31, 41]:
        if k > len(train_labels):
            continue
        for temperature in [0.07, 0.10, 0.15, 0.20, 0.30]:
            knn_probabilities = compute_knn_probabilities(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                k=k,
                temperature=temperature,
                num_classes=num_classes,
            )
            for alpha in np.linspace(0.0, 0.5, 26):
                probabilities = alpha * val_probabilities_tta + (1.0 - alpha) * knn_probabilities
                predictions = probabilities.argmax(axis=1)
                accuracy = float((predictions == y_val).mean())
                macro_f1 = macro_f1_from_predictions(y_val, predictions, num_classes)
                candidate = {
                    "family": "blend_tta_knn",
                    "k": int(k),
                    "temperature": float(temperature),
                    "alpha": float(alpha),
                    "val_accuracy": accuracy,
                    "val_macro_f1": macro_f1,
                }
                if (accuracy, macro_f1) > (float(best["val_accuracy"]), float(best["val_macro_f1"])):
                    best = candidate
    return best


class TrialPruningCallback(keras.callbacks.Callback):
    def __init__(
        self,
        trial: optuna.Trial,
        validation_data: tf.data.Dataset,
        y_val: np.ndarray,
        num_classes: int,
        stage_offset: int,
        checkpoint_path: Path,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.validation_data = validation_data
        self.y_val = y_val
        self.num_classes = num_classes
        self.stage_offset = stage_offset
        self.checkpoint_path = checkpoint_path
        self.best_score = -1.0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        probabilities = self.model.predict(self.validation_data, verbose=0)
        predictions = probabilities.argmax(axis=1)
        accuracy = float((predictions == self.y_val).mean())
        macro_f1 = macro_f1_from_predictions(self.y_val, predictions, self.num_classes)
        score = 0.5 * accuracy + 0.5 * macro_f1
        step = self.stage_offset + epoch + 1
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
            logs["val_eval_score"] = score
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.checkpoint_path, overwrite=True)
        self.trial.report(score, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {step} with score {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-head-epochs", type=int, default=6)
    parser.add_argument("--max-fine-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", default="experiment_hyperband_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    keras.utils.set_random_seed(args.seed)

    clean_df, label_to_class = load_clean_dataframe(root)
    train_split, val_split = stratified_group_split(clean_df, label_to_class, 0.20, args.seed)
    y_val = val_split["label"].to_numpy()
    train_labels = train_split["label"].to_numpy()
    num_classes = len(label_to_class)
    img_size = (args.img_size, args.img_size)

    def decode_image(path: tf.Tensor, label: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, img_size, method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image, 0.0, 255.0)
        if label is None:
            return image
        return image, tf.cast(label, tf.int32)

    def make_labeled_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            (df["path"].astype(str).values, df["label"].values.astype("int32"))
        )
        ds = ds.map(lambda path, label: decode_image(path, label), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(len(df), seed=args.seed, reshuffle_each_iteration=True)
        else:
            ds = ds.cache()
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_labeled_dataset(train_split, training=True)
    train_eval_ds = make_labeled_dataset(train_split, training=False)
    val_ds = make_labeled_dataset(val_split, training=False)
    class_counts = train_split["label"].value_counts().to_dict()
    class_weights = {
        int(label): len(train_split) / (num_classes * count)
        for label, count in class_counts.items()
    }

    def objective(trial: optuna.Trial) -> float:
        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed + trial.number)

        backbone_name = trial.suggest_categorical("backbone", ["efficientnetb0", "efficientnetv2b0"])
        dropout = trial.suggest_float("dropout", 0.25, 0.55, step=0.05)
        dense_units = trial.suggest_categorical("dense_units", [0, 64, 128])
        augmentation_strength = trial.suggest_float("augmentation_strength", 0.6, 1.3, step=0.1)
        head_lr = trial.suggest_float("head_lr", 1e-4, 6e-4, log=True)
        fine_lr = trial.suggest_float("fine_lr", 5e-6, 6e-5, log=True)
        fine_layers = trial.suggest_categorical("fine_layers", [25, 40, 60, 90])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)

        try:
            backbone = build_backbone(backbone_name, img_size + (3,), weights="imagenet")
        except Exception:
            backbone = build_backbone(backbone_name, img_size + (3,), weights=None)
        backbone.trainable = False

        inputs = keras.Input(shape=img_size + (3,))
        x = make_data_augmentation(augmentation_strength)(inputs)
        x = preprocess_for_backbone(backbone_name, x)
        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        if dense_units > 0:
            x = layers.Dense(
                dense_units,
                activation="swish",
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(min(dropout + 0.10, 0.65))(x)
        x = layers.Activation("linear", name="embedding_features")(x)
        outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        model = keras.Model(inputs, outputs, name=f"hyperband_trial_{trial.number}")

        def make_optimizer(learning_rate: float) -> keras.optimizers.Optimizer:
            if optimizer_name == "adamw":
                return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
            return keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint_path = root / f"tmp_hyperband_trial_{trial.number}.keras"
        head_callback = TrialPruningCallback(
            trial=trial,
            validation_data=val_ds,
            y_val=y_val,
            num_classes=num_classes,
            stage_offset=0,
            checkpoint_path=checkpoint_path,
        )
        model.compile(
            optimizer=make_optimizer(head_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.max_head_epochs,
            class_weight=class_weights,
            callbacks=[head_callback],
            verbose=1,
        )

        backbone.trainable = True
        for layer in backbone.layers[:-fine_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        fine_callback = TrialPruningCallback(
            trial=trial,
            validation_data=val_ds,
            y_val=y_val,
            num_classes=num_classes,
            stage_offset=args.max_head_epochs,
            checkpoint_path=checkpoint_path,
        )
        model.compile(
            optimizer=make_optimizer(fine_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.max_fine_epochs,
            class_weight=class_weights,
            callbacks=[
                fine_callback,
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_accuracy",
                    mode="max",
                    factor=0.3,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1,
                ),
            ],
            verbose=1,
        )

        best_model = keras.models.load_model(checkpoint_path)
        val_probabilities = best_model.predict(val_ds, verbose=0)
        val_predictions = val_probabilities.argmax(axis=1)
        val_accuracy = float((val_predictions == y_val).mean())
        val_macro_f1 = macro_f1_from_predictions(y_val, val_predictions, num_classes)

        flip_val_ds = val_ds.map(
            lambda images, labels: (tf.image.flip_left_right(images), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        val_probabilities_flip = best_model.predict(flip_val_ds, verbose=0)
        val_probabilities_tta = (val_probabilities + val_probabilities_flip) / 2.0
        val_predictions_tta = val_probabilities_tta.argmax(axis=1)
        val_accuracy_tta = float((val_predictions_tta == y_val).mean())
        val_macro_f1_tta = macro_f1_from_predictions(y_val, val_predictions_tta, num_classes)
        blend_result = search_embedding_blend(
            model=best_model,
            train_eval_ds=train_eval_ds,
            val_ds=val_ds,
            train_labels=train_labels,
            y_val=y_val,
            val_probabilities_tta=val_probabilities_tta,
            num_classes=num_classes,
        )

        trial.set_user_attr("model_path", str(checkpoint_path.relative_to(root)))
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("val_macro_f1", val_macro_f1)
        trial.set_user_attr("val_accuracy_tta", val_accuracy_tta)
        trial.set_user_attr("val_macro_f1_tta", val_macro_f1_tta)
        trial.set_user_attr("blend_result", blend_result)

        return max(
            0.5 * val_accuracy + 0.5 * val_macro_f1,
            0.5 * val_accuracy_tta + 0.5 * val_macro_f1_tta,
            0.5 * float(blend_result["val_accuracy"]) + 0.5 * float(blend_result["val_macro_f1"]),
        )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3,
        max_resource=args.max_head_epochs + args.max_fine_epochs,
        reduction_factor=3,
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    trials = []
    for trial in study.trials:
        trials.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    trials.sort(key=lambda row: row["value"] if row["value"] is not None else -1, reverse=True)
    output = {
        "note": "Optuna HyperbandPruner tuning on the duplicate-aware validation split. No validation images are used for training.",
        "best_value": study.best_value if study.best_trial else None,
        "best_trial": {
            "number": study.best_trial.number,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        } if study.best_trial else None,
        "trials": trials,
    }
    output_path = root / args.output
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["best_trial"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
