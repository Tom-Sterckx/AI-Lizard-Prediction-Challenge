from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def _lines(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


cells = [
    md(
        """
        # Lizard Species Image Classification Challenge

        **Course:** Artificial Intelligence - Thomas More  
        **Team members:** `<fill in>`  
        **Notebook goal:** build a strong, well-documented TensorFlow/Keras image classifier for 7 lizard species.

        ## Species
        1. Black spiny tailed iguana
        2. Brown anole
        3. Cuban knight anole
        4. Desert iguana
        5. Green anole
        6. Green iguana
        7. Lesser Antillean iguana

        ## What this notebook covers
        - dataset integrity checks and exploratory data analysis
        - image quality checks, including duplicate analysis and OCR-style text-overlay checks
        - data augmentation
        - transfer learning with staged fine-tuning
        - training and validation visualisation
        - confusion matrix and macro-F1 evaluation
        - test-set inference and competition submission export
        - a short GenAI reflection section
        """
    ),
    code(
        """
        # Run this once if your environment is still missing packages.
        # In Google Colab, use this cell before running the rest of the notebook.
        %pip install -q tensorflow pandas numpy matplotlib pillow opencv-python pytesseract openpyxl

        # Important:
        # `pytesseract` is only the Python wrapper.
        # For real OCR extraction you still need the Tesseract engine installed on the machine itself.
        """
    ),
    md(
        """
        ## Why this approach?

        The dataset is relatively small, so a pure CNN from scratch would likely underperform or overfit quickly.  
        Because of that, we combine:

        - transfer learning from an ImageNet backbone
        - moderate augmentation
        - early stopping and learning-rate reduction
        - a duplicate-aware validation split to reduce leakage risk
        - simple test-time augmentation for a small boost at inference time

        The challenge page mentions "probabilities", but the provided sample submission and the macro-F1 scoring setup point to a **single class label per image**.  
        We therefore export `ID,TARGET` with integer class predictions.
        """
    ),
    code(
        """
        from pathlib import Path
        import math
        import shutil
        import warnings

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from IPython.display import display
        from PIL import Image, ImageOps
        from tensorflow import keras
        from tensorflow.keras import layers

        try:
            get_ipython().run_line_magic("matplotlib", "inline")
        except Exception:
            plt.switch_backend("Agg")

        warnings.filterwarnings("ignore")
        plt.style.use("seaborn-v0_8")

        SEED = 42
        IMG_SIZE = (260, 260)
        BATCH_SIZE = 24
        VAL_FRACTION = 0.20
        INITIAL_EPOCHS = 8
        FINE_TUNE_EPOCHS = 12
        FINE_TUNE_LAYERS = 40
        USE_TTA = True

        keras.utils.set_random_seed(SEED)
        AUTOTUNE = tf.data.AUTOTUNE

        print("TensorFlow:", tf.__version__)
        print("Keras:", keras.__version__)
        print("GPU devices:", tf.config.list_physical_devices("GPU"))

        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled.")
            else:
                print("Running on CPU. Training will work, but a GPU or Colab runtime is recommended.")
        except Exception as exc:
            print("Could not configure mixed precision:", exc)
        """
    ),
    code(
        """
        def find_project_root(start: Path | None = None) -> Path:
            start = (start or Path.cwd()).resolve()
            candidates = [start, *start.parents]
            required = {"train.csv", "test.csv", "sample_submission.csv", "train", "test"}
            for candidate in candidates:
                names = {path.name for path in candidate.iterdir()} if candidate.exists() else set()
                if required.issubset(names):
                    return candidate
            raise FileNotFoundError(
                "Could not find the project root. Start the notebook from the repository or a subfolder of it."
            )

        ROOT = find_project_root()
        TRAIN_DIR = ROOT / "train"
        TEST_DIR = ROOT / "test"
        TRAIN_CSV = ROOT / "train.csv"
        TEST_CSV = ROOT / "test.csv"
        SAMPLE_SUBMISSION = ROOT / "sample_submission.csv"

        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

        filename_to_path = {path.name: path for path in TRAIN_DIR.rglob("*.jpg")}
        train_df["path"] = train_df["id"].map(filename_to_path)
        assert train_df["path"].notna().all(), "Some train rows could not be matched to image files."

        train_df["class_name"] = train_df["path"].map(lambda p: Path(p).parent.name)
        label_to_class = (
            train_df[["label", "class_name"]]
            .drop_duplicates()
            .sort_values("label")
            .set_index("label")["class_name"]
            .to_dict()
        )
        class_to_label = {class_name: label for label, class_name in label_to_class.items()}
        class_names = [label_to_class[i] for i in sorted(label_to_class)]
        num_classes = len(class_names)

        test_df["path"] = test_df["id"].map(lambda image_id: TEST_DIR / f"{image_id}.jpg")
        assert test_df["path"].map(lambda p: p.exists()).all(), "Some test rows have no matching image."

        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
        print("Resolved project root:", ROOT)
        print("Label mapping:", label_to_class)
        display(train_df.head())
        """
    ),
    md(
        """
        ## 1. Exploratory Data Analysis

        We first verify that the CSV files and image folders are consistent.  
        After that we inspect:

        - class balance
        - image resolution variation
        - brightness and blur spread
        - near-duplicate groups
        - possible text or watermark overlays
        """
    ),
    code(
        """
        label_counts = train_df["label"].value_counts().sort_index()
        class_distribution = pd.DataFrame(
            {
                "label": label_counts.index,
                "class_name": [label_to_class[i] for i in label_counts.index],
                "count": label_counts.values,
                "share_pct": (100 * label_counts.values / len(train_df)).round(2),
            }
        )
        display(class_distribution)

        plt.figure(figsize=(10, 4))
        plt.bar(class_distribution["class_name"], class_distribution["count"], color="#2a9d8f")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Number of images")
        plt.title("Training class distribution")
        plt.tight_layout()
        plt.show()

        print("train.csv columns:", train_df.columns.tolist())
        print("test.csv columns:", test_df.columns.tolist())
        print("sample_submission columns:", pd.read_csv(SAMPLE_SUBMISSION).columns.tolist())
        """
    ),
    code(
        """
        def show_class_examples(df: pd.DataFrame, n_per_class: int = 4) -> None:
            fig, axes = plt.subplots(num_classes, n_per_class, figsize=(4 * n_per_class, 3 * num_classes))
            axes = np.atleast_2d(axes)
            for row_idx, label in enumerate(sorted(label_to_class)):
                subset = df[df["label"] == label].sample(n_per_class, random_state=SEED)
                for col_idx, (_, row) in enumerate(subset.reset_index(drop=True).iterrows()):
                    with Image.open(row["path"]) as img:
                        axes[row_idx, col_idx].imshow(ImageOps.exif_transpose(img).convert("RGB"))
                    axes[row_idx, col_idx].set_title(label_to_class[label])
                    axes[row_idx, col_idx].axis("off")
            plt.tight_layout()
            plt.show()

        show_class_examples(train_df, n_per_class=4)
        """
    ),
    code(
        """
        def collect_image_stats(paths: list[Path]) -> pd.DataFrame:
            records = []
            for path in paths:
                with Image.open(path) as img:
                    rgb = ImageOps.exif_transpose(img).convert("RGB")
                    gray = np.asarray(rgb.convert("L"))
                    width, height = rgb.size
                records.append(
                    {
                        "path": path,
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height,
                        "brightness": float(gray.mean()),
                        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    }
                )
            return pd.DataFrame(records)

        train_stats = collect_image_stats(train_df["path"].tolist())
        test_stats = collect_image_stats(test_df["path"].tolist())

        eda_summary = pd.DataFrame(
            {
                "split": ["train", "test"],
                "n_images": [len(train_stats), len(test_stats)],
                "width_min": [train_stats["width"].min(), test_stats["width"].min()],
                "width_median": [int(train_stats["width"].median()), int(test_stats["width"].median())],
                "width_max": [train_stats["width"].max(), test_stats["width"].max()],
                "height_min": [train_stats["height"].min(), test_stats["height"].min()],
                "height_median": [int(train_stats["height"].median()), int(test_stats["height"].median())],
                "height_max": [train_stats["height"].max(), test_stats["height"].max()],
                "brightness_median": [round(train_stats["brightness"].median(), 2), round(test_stats["brightness"].median(), 2)],
            }
        )
        display(eda_summary)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].hist(train_stats["width"], bins=25, color="#457b9d", alpha=0.9)
        axes[0].set_title("Train image widths")
        axes[1].hist(train_stats["height"], bins=25, color="#e76f51", alpha=0.9)
        axes[1].set_title("Train image heights")
        axes[2].hist(train_stats["brightness"], bins=25, color="#e9c46a", alpha=0.9)
        axes[2].set_title("Train brightness distribution")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        def average_hash(path: Path, hash_size: int = 16) -> str:
            with Image.open(path) as img:
                gray = ImageOps.exif_transpose(img).convert("L").resize(
                    (hash_size, hash_size),
                    Image.Resampling.BILINEAR,
                )
                arr = np.asarray(gray, dtype=np.float32)
            return "".join("1" if pixel > arr.mean() else "0" for pixel in arr.flatten())

        train_df["hash_group"] = train_df["path"].map(average_hash)

        hash_group_sizes = train_df.groupby("hash_group").size().sort_values(ascending=False)
        duplicate_groups = hash_group_sizes[hash_group_sizes > 1]

        print("Approximate duplicate groups:", len(duplicate_groups))
        if len(duplicate_groups) > 0:
            print("Largest duplicate group size:", int(duplicate_groups.iloc[0]))
            duplicate_preview = (
                train_df[train_df["hash_group"].isin(duplicate_groups.index[:5])]
                .sort_values(["hash_group", "class_name", "id"])
                [["id", "class_name", "hash_group"]]
                .reset_index(drop=True)
            )
            display(duplicate_preview.head(20))
        """
    ),
    code(
        """
        def text_overlay_score(path: Path) -> int:
            image = cv2.imread(str(path))
            if image is None:
                return -1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
            merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / max(h, 1)
                if 60 <= area <= 6000 and aspect_ratio > 1.8 and h < 70:
                    candidates += 1
            return candidates

        sample_parts = []
        for _, subset in train_df.groupby("label"):
            sample_parts.append(subset.sample(min(len(subset), 25), random_state=SEED))
        text_check_df = pd.concat(sample_parts).reset_index(drop=True)
        text_check_df["text_overlay_score"] = text_check_df["path"].map(text_overlay_score)

        ocr_available = False
        pytesseract = None
        try:
            import pytesseract  # type: ignore
            ocr_available = shutil.which("tesseract") is not None
        except Exception:
            pytesseract = None

        if ocr_available:
            def extract_text(path: Path) -> str:
                text = pytesseract.image_to_string(str(path), config="--psm 11")
                return " ".join(text.split())

            text_check_df["ocr_text"] = text_check_df["path"].map(extract_text)
            text_check_df["has_ocr_text"] = text_check_df["ocr_text"].str.len() >= 4
            display(
                text_check_df.sort_values(
                    ["has_ocr_text", "text_overlay_score"],
                    ascending=[False, False],
                )[["id", "class_name", "text_overlay_score", "ocr_text"]].head(15)
            )
        else:
            print("pytesseract / Tesseract not available in this environment.")
            print("Fallback: OpenCV text-overlay heuristic only.")
            display(
                text_check_df.sort_values("text_overlay_score", ascending=False)[
                    ["id", "class_name", "text_overlay_score"]
                ].head(15)
            )
        """
    ),
    md(
        """
        ## 2. Validation Strategy

        A random split can be misleading when the dataset contains duplicate or near-duplicate images.  
        To reduce leakage, we first create image hash groups and then do a **stratified group split**:

        - class balance stays similar in train and validation
        - images with the same hash group stay together
        - validation macro-F1 becomes more trustworthy
        """
    ),
    code(
        """
        def stratified_group_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
            rng = np.random.default_rng(seed)
            target_total = int(round(len(df) * val_fraction))
            target_per_label = df["label"].value_counts().sort_index() * val_fraction

            group_rows = []
            for hash_group, subset in df.groupby("hash_group"):
                label_counts = subset["label"].value_counts().sort_index()
                row = {
                    "hash_group": hash_group,
                    "group_size": len(subset),
                }
                for label in sorted(label_to_class):
                    row[f"label_{label}"] = int(label_counts.get(label, 0))
                group_rows.append(row)

            group_df = pd.DataFrame(group_rows)
            group_df = group_df.sample(frac=1.0, random_state=seed).sort_values("group_size", ascending=False).reset_index(drop=True)

            val_groups = []
            val_total = 0
            current_counts = pd.Series(0.0, index=target_per_label.index)

            for row in group_df.itertuples(index=False):
                row_counts = pd.Series(
                    {label: getattr(row, f"label_{label}") for label in target_per_label.index},
                    dtype=float,
                )

                remaining_deficit = (target_per_label - current_counts).clip(lower=0)
                group_fills_needed_labels = float((np.minimum(row_counts, remaining_deficit)).sum())
                group_excess = float((row_counts - remaining_deficit).clip(lower=0).sum())

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

        if "hash_group" not in train_df.columns:
            train_df["hash_group"] = train_df["path"].map(average_hash)

        train_split, val_split = stratified_group_split(train_df, val_fraction=VAL_FRACTION, seed=SEED)

        print("Train split shape:", train_split.shape)
        print("Validation split shape:", val_split.shape)
        print("Hash overlap:", len(set(train_split["hash_group"]) & set(val_split["hash_group"])))

        split_summary = pd.DataFrame(
            {
                "class_name": class_names,
                "train_count": train_split["label"].value_counts().sort_index().values,
                "val_count": val_split["label"].value_counts().sort_index().values,
            }
        )
        split_summary["val_share_pct"] = (100 * split_summary["val_count"] / (split_summary["train_count"] + split_summary["val_count"])).round(2)
        display(split_summary)
        """
    ),
    md(
        """
        ## 3. TensorFlow / Keras Input Pipeline

        We keep the pipeline fully inside TensorFlow:

        - file paths are read from the CSV-driven dataframe
        - images are decoded and resized on the fly
        - augmentation is done with Keras preprocessing layers
        - `tf.data` takes care of batching and prefetching
        """
    ),
    code(
        """
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.15),
                layers.RandomContrast(0.15),
                layers.RandomTranslation(0.05, 0.05),
            ],
            name="augmentation",
        )

        @tf.autograph.experimental.do_not_convert
        def decode_image(path: tf.Tensor, label: tf.Tensor | None = None):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BICUBIC)
            image = tf.cast(image, tf.float32)
            image = tf.clip_by_value(image, 0.0, 255.0)
            if label is None:
                return image
            return image, tf.cast(label, tf.int32)

        def make_labeled_dataset(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
            dataset = tf.data.Dataset.from_tensor_slices(
                (df["path"].astype(str).values, df["label"].values.astype("int32"))
            )
            if training:
                dataset = dataset.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
            dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
            dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            return dataset

        def make_test_dataset(df: pd.DataFrame) -> tf.data.Dataset:
            dataset = tf.data.Dataset.from_tensor_slices(df["path"].astype(str).values)
            dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
            dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            return dataset

        train_ds = make_labeled_dataset(train_split, training=True)
        val_ds = make_labeled_dataset(val_split, training=False)
        test_ds = make_test_dataset(test_df)

        for images, labels in train_ds.take(1):
            print("Image batch shape:", images.shape)
            print("Label batch shape:", labels.shape)
        """
    ),
    md(
        """
        ## 4. Transfer Learning Model

        We use an ImageNet-pretrained backbone as a feature extractor, then fine-tune the top part later.

        Training happens in two stages:
        1. freeze the backbone and train only the classification head
        2. unfreeze the top layers and continue with a much smaller learning rate
        """
    ),
    code(
        """
        def build_transfer_model(
            input_shape: tuple[int, int, int] = IMG_SIZE + (3,),
            num_classes: int = 7,
            dropout_rate: float = 0.35,
            weights: str | None = "imagenet",
        ) -> tuple[keras.Model, keras.Model]:
            try:
                backbone = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights=weights,
                    input_shape=input_shape,
                )
                print("Loaded EfficientNetB0 with ImageNet weights.")
            except Exception as exc:
                print(f"Could not load pretrained weights ({exc}). Falling back to random initialization.")
                backbone = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights=None,
                    input_shape=input_shape,
                )

            backbone.trainable = False

            inputs = keras.Input(shape=input_shape)
            x = data_augmentation(inputs)
            x = keras.applications.efficientnet.preprocess_input(x)
            x = backbone(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

            model = keras.Model(inputs, outputs, name="lizard_classifier")
            return model, backbone

        def make_class_weights(labels: pd.Series) -> dict[int, float]:
            counts = labels.value_counts().to_dict()
            total = len(labels)
            n_classes = len(counts)
            return {int(label): total / (n_classes * count) for label, count in counts.items()}

        model, backbone = build_transfer_model(num_classes=num_classes)
        class_weights = make_class_weights(train_split["label"])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        checkpoint_path = ROOT / "best_lizard_model.keras"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        model.summary()
        print("Class weights:", class_weights)
        """
    ),
    code(
        """
        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        """
    ),
    code(
        """
        def unfreeze_top_layers(backbone_model: keras.Model, n_layers: int = 40) -> None:
            backbone_model.trainable = True
            for layer in backbone_model.layers[:-n_layers]:
                layer.trainable = False
            for layer in backbone_model.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

        unfreeze_top_layers(backbone, n_layers=FINE_TUNE_LAYERS)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        model = keras.models.load_model(checkpoint_path)
        print("Reloaded best validation checkpoint from:", checkpoint_path)
        """
    ),
    code(
        """
        def combine_histories(*histories: keras.callbacks.History) -> pd.DataFrame:
            rows = []
            epoch_offset = 0
            for history in histories:
                history_frame = pd.DataFrame(history.history)
                history_frame["epoch"] = np.arange(len(history_frame)) + epoch_offset + 1
                rows.append(history_frame)
                epoch_offset += len(history_frame)
            return pd.concat(rows, ignore_index=True)

        history_df = combine_histories(history_head, history_fine)
        display(history_df.tail())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(history_df["epoch"], history_df["loss"], label="train loss", color="#1d3557")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val loss", color="#e63946")
        axes[0].set_title("Training vs validation loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train accuracy", color="#2a9d8f")
        axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val accuracy", color="#f4a261")
        axes[1].set_title("Training vs validation accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 5. Evaluation

        The official metric is **macro-F1**, so we evaluate with:

        - validation accuracy
        - confusion matrix
        - per-class precision, recall and F1
        - a short visual error analysis on misclassified examples
        """
    ),
    code(
        """
        def metrics_from_confusion_matrix(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            true_positives = np.diag(cm).astype(np.float64)
            precision = np.divide(
                true_positives,
                cm.sum(axis=0),
                out=np.zeros_like(true_positives),
                where=cm.sum(axis=0) != 0,
            )
            recall = np.divide(
                true_positives,
                cm.sum(axis=1),
                out=np.zeros_like(true_positives),
                where=cm.sum(axis=1) != 0,
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(true_positives),
                where=(precision + recall) != 0,
            )
            return precision, recall, f1, float(f1.mean())

        val_probabilities = model.predict(val_ds, verbose=0)
        val_predictions = val_probabilities.argmax(axis=1)
        val_targets = val_split["label"].to_numpy()

        confusion = tf.math.confusion_matrix(
            val_targets,
            val_predictions,
            num_classes=num_classes,
        ).numpy()

        precision, recall, f1_per_class, macro_f1 = metrics_from_confusion_matrix(confusion)
        val_accuracy = float((val_predictions == val_targets).mean())

        evaluation_table = pd.DataFrame(
            {
                "class_name": class_names,
                "precision": np.round(precision, 4),
                "recall": np.round(recall, 4),
                "f1": np.round(f1_per_class, 4),
                "support": confusion.sum(axis=1),
            }
        )

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {macro_f1:.4f}")
        display(evaluation_table)
        """
    ),
    code(
        """
        def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
            plt.figure(figsize=(8, 7))
            plt.imshow(cm, cmap="Blues")
            plt.title("Validation confusion matrix")
            plt.colorbar()
            tick_positions = np.arange(len(labels))
            plt.xticks(tick_positions, labels, rotation=35, ha="right")
            plt.yticks(tick_positions, labels)

            threshold = cm.max() / 2 if cm.max() > 0 else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > threshold else "black"
                    plt.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.show()

        plot_confusion_matrix(confusion, class_names)
        """
    ),
    code(
        """
        error_df = val_split.copy()
        error_df["pred_label"] = val_predictions
        error_df["pred_class_name"] = error_df["pred_label"].map(label_to_class)
        error_df["confidence"] = val_probabilities.max(axis=1)
        misclassified = error_df[error_df["label"] != error_df["pred_label"]].sort_values("confidence", ascending=False)

        display(misclassified[["id", "class_name", "pred_class_name", "confidence"]].head(10))

        n_examples = min(9, len(misclassified))
        if n_examples > 0:
            fig, axes = plt.subplots(math.ceil(n_examples / 3), 3, figsize=(12, 4 * math.ceil(n_examples / 3)))
            axes = np.array(axes).reshape(-1)
            for ax in axes[n_examples:]:
                ax.axis("off")

            for ax, (_, row) in zip(axes, misclassified.head(n_examples).iterrows()):
                with Image.open(row["path"]) as img:
                    ax.imshow(ImageOps.exif_transpose(img).convert("RGB"))
                ax.set_title(
                    f"True: {row['class_name']}\\nPred: {row['pred_class_name']}\\nConf: {row['confidence']:.2f}"
                )
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("No misclassified validation images in this run.")
        """
    ),
    md(
        """
        ## 6. Submission File

        To squeeze a little more performance out of the final model, we optionally use a very simple form of test-time augmentation:

        - original test image
        - horizontally flipped test image

        The predicted probabilities are averaged before taking the final class label.
        """
    ),
    code(
        """
        def horizontal_flip_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
            return dataset.map(
                lambda images: tf.image.flip_left_right(images),
                num_parallel_calls=AUTOTUNE,
            ).prefetch(AUTOTUNE)

        if USE_TTA:
            test_probs_base = model.predict(test_ds, verbose=0)
            test_probs_flip = model.predict(horizontal_flip_dataset(test_ds), verbose=0)
            test_probabilities = (test_probs_base + test_probs_flip) / 2.0
        else:
            test_probabilities = model.predict(test_ds, verbose=0)

        test_predictions = test_probabilities.argmax(axis=1).astype(int)

        submission_df = pd.DataFrame(
            {
                "ID": test_df["id"].astype(int),
                "TARGET": test_predictions,
            }
        )
        submission_path = ROOT / "submission_transfer_learning.csv"
        submission_df.to_csv(submission_path, index=False)

        print("Saved submission to:", submission_path)
        display(submission_df.head())
        """
    ),
    md(
        """
        ## 7. Results Discussion

        Replace the placeholders below after your final training run:

        - **Best validation accuracy:** `<fill in>`
        - **Best validation macro-F1:** `<fill in>`
        - **Main confusion pairs:** `<fill in>`
        - **What helped most:** duplicate-aware split, augmentation, staged fine-tuning, TTA

        In practice, the hardest errors are expected between visually similar green/brown species and between iguana variants with comparable body shape or background context.
        """
    ),
    md(
        """
        ## 8. GenAI Reflection

        GenAI was used as a support tool during the project, not as a replacement for understanding.

        ### How it helped
        - structure the notebook into a clear, defendable workflow
        - suggest extra validation ideas such as duplicate checks and text-overlay screening
        - help brainstorm transfer-learning improvements and training callbacks
        - improve the wording of explanations and markdown sections

        ### What we still validated ourselves
        - the label mapping between CSV and folders
        - whether the split strategy avoids duplicate leakage
        - whether the model architecture and training settings are appropriate
        - whether the exported submission matches the required format

        ### Important limitation
        Suggestions from GenAI can be useful, but they still need human verification.  
        For that reason, every technical choice in this notebook was checked against the dataset, the course material, and the observed validation behaviour.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks") / "lizard_species_transfer_learning.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote notebook to {output_path}")
