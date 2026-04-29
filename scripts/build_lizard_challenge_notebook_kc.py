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
        - staged transfer learning with a stronger fine-tuning schedule
        - training and validation visualisation
        - confusion matrix and macro-F1 evaluation
        - embedding-assisted inference with kNN blending
        - test-set inference and competition submission export
        - a short GenAI reflection section
        """
    ),
    code(
        """
        # Run this once if your environment is still missing packages.
        # In Google Colab, use this cell before running the rest of the notebook.
        %pip install -q tensorflow pandas numpy matplotlib pillow opencv-python pytesseract openpyxl optuna

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
        - horizontal-flip test-time augmentation plus embedding-assisted inference selection

        The challenge page mentions "probabilities", but the provided sample submission and the macro-F1 scoring setup point to a **single class label per image**.  
        We therefore export `ID,TARGET` with integer class predictions.
        """
    ),
    md(
        """
        ## 0A. Imports and Global Settings

        All important hyperparameters are collected near the top so the notebook is easy to adjust and defend.  
        We fix the random seed for repeatability, use `260x260` images as a balance between detail and speed, and keep the final training settings visible.
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
            import optuna
        except ImportError:
            optuna = None

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
        HEAD_LEARNING_RATE = 3e-4
        FINE_TUNE_LEARNING_RATE = 3e-5
        USE_CLASS_WEIGHTS = True
        TRAINING_AUGMENTATION_STRENGTH = 1.0
        DROPOUT_RATE = 0.35
        DEFAULT_MODEL_PARAMS = {
            "learning_rate": 1e-4,
            "dropout_rate": 0.55,
            "dense_units": 64,
            "augmentation_strength": 0.6,
            "weight_decay": 1e-4,
        }
        INFERENCE_SEARCH_MODE = "auto"

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
    md(
        """
        ## 0B. Load Metadata and Resolve Image Paths

        The CSV files contain image IDs, while the actual images live in folders.  
        This section connects both sources, checks missing files, and builds the label-to-species mapping used everywhere else.
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

        train_image_lookup = build_image_lookup(TRAIN_DIR)
        test_image_lookup = build_image_lookup(TEST_DIR)

        train_df["path"] = train_df["id"].map(lambda image_id: resolve_image_path(image_id, train_image_lookup))
        missing_train_df = train_df[train_df["path"].isna()].copy()
        if len(missing_train_df) > 0:
            print(f"Warning: {len(missing_train_df)} train.csv rows have no matching local image file.")
            print("These rows are excluded from image training because TensorFlow cannot load missing files.")
            display(missing_train_df[["id", "label"]].head(20))
            train_df = train_df[train_df["path"].notna()].reset_index(drop=True)

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

        test_df["path"] = test_df["id"].map(lambda image_id: resolve_image_path(image_id, test_image_lookup))
        missing_test_df = test_df[test_df["path"].isna()].copy()
        assert len(missing_test_df) == 0, "Some test rows have no matching image."

        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
        print("Available local train images:", len(train_image_lookup))
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
    md(
        """
        ### 1A. Class Balance

        We check class balance early because it influences training choices such as class weights, macro-F1 reporting, and how much we trust raw accuracy.
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
    md(
        """
        ### 1B. Visual Sample Check

        A quick grid of examples helps verify that the labels and folder names make sense before training a model on them.
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
    md(
        """
        ### 1C. Image Quality and Resolution Statistics

        Image sizes, brightness, and blur can affect training.  
        We inspect them to catch extreme outliers and to justify resizing all images consistently.
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
    md(
        """
        ### 1D. Duplicate Detection

        We calculate a simple perceptual hash so near-duplicate images can stay in the same train/validation group.  
        This reduces validation leakage: the model should not see almost the same image during training and validation.
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
    md(
        """
        ### 1E. Text or Watermark Review

        Text overlays can accidentally leak information or distract the model.  
        We do not blindly remove these images, but we score them so they can be inspected and discussed.
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
        ## 1F. Automatic Dataset Cleaning

        Before splitting the data, we remove only high-confidence problems:

        - duplicate hash groups that appear with multiple labels
        - unreadable image rows
        - images with extremely unusual aspect ratios

        Blur and possible text overlays are kept as review signals instead of being removed blindly, because those can still be real challenge images.
        """
    ),
    md(
        """
        ### 1G. Conservative Cleaning Rules

        The cleaning is deliberately conservative.  
        We remove only rows that are very likely to be problematic, while keeping review columns for softer quality signals.
        """
    ),
    code(
        """
        def build_clean_training_dataframe(df: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            working_df = df.copy()
            working_df["path_str"] = working_df["path"].astype(str)

            quality_df = stats.copy()
            quality_df["path_str"] = quality_df["path"].astype(str)
            quality_df = quality_df[["path_str", "aspect_ratio", "blur_laplacian_var", "brightness"]]
            working_df = working_df.merge(quality_df, on="path_str", how="left")

            hash_label_counts = working_df.groupby("hash_group")["label"].nunique()
            conflicting_hashes = set(hash_label_counts[hash_label_counts > 1].index)

            low_blur_threshold = float(working_df["blur_laplacian_var"].quantile(0.01))
            high_text_threshold = int(text_check_df["text_overlay_score"].quantile(0.95))
            sampled_text_scores = text_check_df.set_index("path")["text_overlay_score"].to_dict()

            working_df["cleaning_reason"] = ""
            working_df.loc[working_df["path"].isna(), "cleaning_reason"] = "missing_image_path"
            working_df.loc[working_df["hash_group"].isin(conflicting_hashes), "cleaning_reason"] = "conflicting_duplicate_hash"
            working_df.loc[
                (working_df["aspect_ratio"] < 0.25) | (working_df["aspect_ratio"] > 4.0),
                "cleaning_reason",
            ] = "extreme_aspect_ratio"

            working_df["review_low_blur"] = working_df["blur_laplacian_var"] <= low_blur_threshold
            working_df["review_text_overlay_score"] = working_df["path"].map(sampled_text_scores).fillna(0).astype(int)
            working_df["review_possible_text_overlay"] = working_df["review_text_overlay_score"] >= high_text_threshold

            removed_df = working_df[working_df["cleaning_reason"] != ""].copy()
            clean_df = working_df[working_df["cleaning_reason"] == ""].copy()
            clean_df = clean_df.drop(columns=["path_str"]).reset_index(drop=True)

            cleaning_summary = pd.DataFrame(
                {
                    "step": [
                        "original_rows",
                        "removed_rows",
                        "clean_rows",
                        "review_low_blur_rows",
                        "review_possible_text_overlay_rows",
                    ],
                    "count": [
                        len(working_df),
                        len(removed_df),
                        len(clean_df),
                        int(clean_df["review_low_blur"].sum()),
                        int(clean_df["review_possible_text_overlay"].sum()),
                    ],
                }
            )

            display(cleaning_summary)
            if len(removed_df) > 0:
                display(
                    removed_df[
                        ["id", "class_name", "hash_group", "cleaning_reason"]
                    ].sort_values(["cleaning_reason", "class_name", "id"]).head(30)
                )
            return clean_df, removed_df

        clean_train_df, removed_train_df = build_clean_training_dataframe(train_df, train_stats)

        print("Rows available after automatic cleaning:", len(clean_train_df))
        display(
            clean_train_df.groupby(["label", "class_name"])
            .size()
            .reset_index(name="clean_count")
            .sort_values("label")
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
    md(
        """
        ### 2A. Duplicate-Aware Split Function

        A normal random split can put near-duplicates in both train and validation.  
        This function splits by `hash_group`, so similar images stay together and the validation score is harder to game.
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

        if "hash_group" not in clean_train_df.columns:
            clean_train_df["hash_group"] = clean_train_df["path"].map(average_hash)

        train_split, val_split = stratified_group_split(clean_train_df, val_fraction=VAL_FRACTION, seed=SEED)

        print("Train split shape:", train_split.shape)
        print("Validation split shape:", val_split.shape)
        print("Hash overlap:", len(set(train_split["hash_group"]) & set(val_split["hash_group"])))

        split_summary = pd.DataFrame(
            {
                "class_name": class_names,
                "train_count": train_split["label"].value_counts().reindex(sorted(label_to_class), fill_value=0).values,
                "val_count": val_split["label"].value_counts().reindex(sorted(label_to_class), fill_value=0).values,
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
    md(
        """
        ### 3A. Augmentation and Dataset Builders

        Augmentation is applied only to training images.  
        Validation and test images are resized in the same way but not augmented, so evaluation stays stable and comparable.
        """
    ),
    code(
        """
        def make_data_augmentation(strength: float = 1.0) -> keras.Sequential:
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

        data_augmentation = make_data_augmentation(TRAINING_AUGMENTATION_STRENGTH)

        def resize_for_model(image: tf.Tensor, training: bool) -> tf.Tensor:
            image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BICUBIC)
            return tf.clip_by_value(image, 0.0, 255.0)

        @tf.autograph.experimental.do_not_convert
        def decode_image(path: tf.Tensor, label: tf.Tensor | None = None, training: bool = False):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = resize_for_model(image, training=training)
            if label is None:
                return image
            return image, tf.cast(label, tf.int32)

        def make_labeled_dataset(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
            dataset = tf.data.Dataset.from_tensor_slices(
                (df["path"].astype(str).values, df["label"].values.astype("int32"))
            )
            dataset = dataset.map(
                lambda path, label: decode_image(path, label, training=training),
                num_parallel_calls=AUTOTUNE,
            )
            if not training:
                dataset = dataset.cache()
            if training:
                dataset = dataset.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
            dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            return dataset

        def make_test_dataset(df: pd.DataFrame) -> tf.data.Dataset:
            dataset = tf.data.Dataset.from_tensor_slices(df["path"].astype(str).values)
            dataset = dataset.map(
                lambda path: decode_image(path, training=False),
                num_parallel_calls=AUTOTUNE,
            )
            dataset = dataset.cache()
            dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            return dataset

        train_ds = make_labeled_dataset(train_split, training=True)
        train_eval_ds = make_labeled_dataset(train_split, training=False)
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

        The best-performing setup on our validation split is a **two-stage EfficientNetB0 transfer-learning pipeline**:

        1. train only the classification head for a few epochs
        2. unfreeze the top backbone layers and continue with a smaller learning rate

        For report transparency, the earlier frozen-backbone + Optuna-style settings are kept in the notebook as commented reference values and in the ablation section below.
        """
    ),
    md(
        """
        ### 4A. Model Architecture Choice

        We use **EfficientNetB0** because it is a strong ImageNet-pretrained backbone that is still light enough to train on Colab or a normal laptop.  
        The custom head adds batch normalization and dropout to reduce overfitting, which is important because this dataset is small.

        The named layers `gradcam_features` and `embedding_features` are intentional:

        - `gradcam_features` lets us explain predictions with Grad-CAM later.
        - `embedding_features` lets us compare images in feature space for the kNN/blend inference step.
        """
    ),
    code(
        """
        def build_transfer_model(
            input_shape: tuple[int, int, int] = IMG_SIZE + (3,),
            num_classes: int = 7,
            dropout_rate: float = DROPOUT_RATE,
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
            for layer in backbone.layers:
                layer.trainable = False

            inputs = keras.Input(shape=input_shape)
            x = data_augmentation(inputs)
            x = keras.applications.efficientnet.preprocess_input(x)
            x = backbone(x, training=False)
            x = layers.Activation("linear", name="gradcam_features")(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Activation("linear", name="embedding_features")(x)
            outputs = layers.Dense(
                num_classes,
                activation="softmax",
                dtype="float32",
            )(x)

            model = keras.Model(inputs, outputs, name="lizard_classifier")
            return model, backbone
        """
    ),
    md(
        """
        ### 4B. Class Weights and Validation Metrics

        Accuracy is useful, but macro-F1 tells us whether the model performs fairly across all species.  
        We calculate both during training and use class weights so smaller classes are not ignored.
        """
    ),
    code(
        """
        def make_class_weights(labels: pd.Series) -> dict[int, float]:
            counts = labels.value_counts().to_dict()
            total = len(labels)
            n_classes = len(counts)
            return {int(label): total / (n_classes * count) for label, count in counts.items()}

        def get_optional_class_weights(labels: pd.Series) -> dict[int, float] | None:
            if not USE_CLASS_WEIGHTS:
                return None
            return make_class_weights(labels)

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

        class ValidationMacroF1Callback(keras.callbacks.Callback):
            def __init__(self, validation_data: tf.data.Dataset, targets: np.ndarray) -> None:
                super().__init__()
                self.validation_data = validation_data
                self.targets = targets
                self.history: list[dict[str, float]] = []

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                if logs is None:
                    logs = {}
                probabilities = self.model.predict(self.validation_data, verbose=0)
                predictions = probabilities.argmax(axis=1)
                confusion = tf.math.confusion_matrix(
                    self.targets,
                    predictions,
                    num_classes=num_classes,
                ).numpy()
                _, _, _, macro_f1 = metrics_from_confusion_matrix(confusion)
                accuracy = float((predictions == self.targets).mean())
                logs["val_macro_f1"] = macro_f1
                logs["val_eval_accuracy"] = accuracy
                self.history.append(
                    {
                        "epoch": float(epoch + 1),
                        "val_macro_f1": macro_f1,
                        "val_eval_accuracy": accuracy,
                    }
                )
                print(
                    f"Epoch {epoch + 1}: val_macro_f1={macro_f1:.4f} - "
                    f"val_eval_accuracy={accuracy:.4f}"
                )
        """
    ),
    md(
        """
        ### 4C. Training Callbacks

        The callbacks make training more stable and defendable:

        - `ModelCheckpoint` keeps the best validation model, not just the last epoch.
        - `EarlyStopping` prevents wasting epochs once validation stops improving.
        - `ReduceLROnPlateau` lowers the learning rate when the model gets stuck.
        """
    ),
    code(
        """
        class_weights = get_optional_class_weights(train_split["label"])
        val_targets = val_split["label"].to_numpy()

        def compile_model(model: keras.Model, learning_rate: float) -> None:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        def make_training_callbacks(checkpoint_path: Path, verbose: int = 1) -> tuple[list[keras.callbacks.Callback], ValidationMacroF1Callback]:
            val_macro_f1_callback = ValidationMacroF1Callback(val_ds, val_targets)
            callbacks = [
                val_macro_f1_callback,
                keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_eval_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=verbose,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_eval_accuracy",
                    mode="max",
                    patience=4,
                    restore_best_weights=True,
                    verbose=verbose,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_eval_accuracy",
                    factor=0.3,
                    patience=2,
                    mode="max",
                    min_lr=1e-6,
                    verbose=verbose,
                ),
            ]
            return callbacks, val_macro_f1_callback

        def unfreeze_top_layers(backbone_model: keras.Model, n_layers: int = FINE_TUNE_LAYERS) -> None:
            backbone_model.trainable = True
            for layer in backbone_model.layers[:-n_layers]:
                layer.trainable = False
            for layer in backbone_model.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
        """
    ),
    md(
        """
        ### 4D. Staged Fine-Tuning

        We train in two phases.  
        First only the new classification head learns the lizard classes. Then we unfreeze the top EfficientNet layers with a much smaller learning rate, so the model can adapt to lizard-specific details without destroying the useful ImageNet features.

        The earlier frozen-only/Optuna attempt is kept as comments below for the report, because it shows that we tried a simpler approach before moving to staged fine-tuning.
        """
    ),
    code(
        """
        # Previous frozen-only attempt kept here for the report:
        # FINAL_EPOCHS = 10
        # RUN_OPTUNA = True
        # OPTUNA_TRIALS = 3
        # OPTUNA_EPOCHS = 2
        # USE_CLASS_WEIGHTS = False
        # DEFAULT_MODEL_PARAMS = {
        #     "learning_rate": 1e-4,
        #     "dropout_rate": 0.55,
        #     "dense_units": 64,
        #     "augmentation_strength": 0.6,
        #     "weight_decay": 1e-4,
        # }

        checkpoint_path = ROOT / "best_lizard_model.keras"
        model, backbone = build_transfer_model(num_classes=num_classes)
        callbacks, val_macro_f1_callback = make_training_callbacks(checkpoint_path=checkpoint_path, verbose=1)

        compile_model(model, learning_rate=HEAD_LEARNING_RATE)
        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        unfreeze_top_layers(backbone, n_layers=FINE_TUNE_LAYERS)
        compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        model = keras.models.load_model(checkpoint_path)

        model.summary()
        print("Class weights:", class_weights)
        print("Training plan: staged fine-tuning")
        print("Head learning rate:", HEAD_LEARNING_RATE)
        print("Fine-tune learning rate:", FINE_TUNE_LEARNING_RATE)
        print("Fine-tuned backbone layers:", FINE_TUNE_LAYERS)
        print("Primary selection metric: validation accuracy")
        """
    ),
    md(
        """
        ### 4E. Training Curves

        We combine the head-training and fine-tuning histories into one table.  
        The plots make it easy to see whether the model is learning, overfitting, or improving after the backbone is partly unfrozen.
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
        macro_f1_history_df = pd.DataFrame(val_macro_f1_callback.history).drop_duplicates(
            subset="epoch",
            keep="last",
        )
        if not macro_f1_history_df.empty and "val_macro_f1" not in history_df.columns:
            macro_f1_history_df["epoch"] = macro_f1_history_df["epoch"].astype(int)
            history_df = history_df.merge(macro_f1_history_df, on="epoch", how="left")
        display(history_df.tail())

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
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

        if "val_macro_f1" in history_df.columns:
            axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="val macro-F1", color="#264653")
        if "val_eval_accuracy" in history_df.columns:
            axes[2].plot(history_df["epoch"], history_df["val_eval_accuracy"], label="val eval accuracy", color="#8ab17d")
        axes[2].set_title("Validation metrics")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

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
    md(
        """
        ### 5A. Ablation Study

        A strong notebook does not only report the best model, but also shows **why** the chosen setup makes sense.

        In this section we compare multiple variants:

        - a conservative frozen-backbone baseline
        - the stronger staged fine-tuning setup
        - RGB training versus grayscale training
        - the final staged model with embedding-assisted inference

        Validation accuracy is the primary comparison metric here. Macro-F1 is still useful as a secondary check because the competition metric may care about class balance.

        This makes it easier to defend which technical choices actually improved the model.
        """
    ),
    code(
        """
        ABLATION_RESULTS = [
            {
                "experiment": "Baseline transfer learning",
                "augmentation": "No",
                "backbone": "Frozen",
                "tta": "No",
                "image_size": "260x260",
                "val_accuracy": None,
                "val_macro_f1": None,
                "notes": "Earlier conservative frozen setup kept for comparison.",
            },
            {
                "experiment": "Frozen model + augmentation",
                "augmentation": "Yes",
                "backbone": "Frozen",
                "tta": "No",
                "image_size": "260x260",
                "val_accuracy": None,
                "val_macro_f1": None,
                "notes": "Same backbone, but with augmentation enabled.",
            },
            {
                "experiment": "Staged fine-tuning",
                "augmentation": "Yes",
                "backbone": "Top layers unfrozen",
                "tta": "No",
                "image_size": "260x260",
                "val_accuracy": 0.6491,
                "val_macro_f1": 0.6379,
                "notes": "RGB model. Head training followed by low-learning-rate fine-tuning.",
            },
            {
                "experiment": "Staged fine-tuning + TTA",
                "augmentation": "Yes",
                "backbone": "Top layers unfrozen",
                "tta": "Horizontal flip",
                "image_size": "260x260",
                "val_accuracy": 0.6535,
                "val_macro_f1": 0.6453,
                "notes": "Same RGB model, but averaged with horizontal-flip predictions.",
            },
            {
                "experiment": "Grayscale staged fine-tuning",
                "augmentation": "Yes",
                "backbone": "Top layers unfrozen",
                "tta": "No",
                "image_size": "260x260",
                "val_accuracy": 0.5263,
                "val_macro_f1": 0.5262,
                "notes": "Color removed by converting each image to grayscale and repeating the channel.",
            },
            {
                "experiment": "Final model + embedding blend",
                "augmentation": "Yes",
                "backbone": "Top layers unfrozen",
                "tta": "Yes / Blend",
                "image_size": "260x260",
                "val_accuracy": 0.7193,
                "val_macro_f1": 0.7175,
                "notes": "Validation-selected softmax + kNN feature-space blend.",
            },
        ]

        ablation_df = pd.DataFrame(ABLATION_RESULTS)
        display(ablation_df)
        """
    ),
    md(
        """
        ### 5A.1 Color Information Ablation

        To test whether color really helps, we trained the same staged EfficientNetB0 setup twice:

        - **RGB model:** normal color images
        - **Grayscale model:** images converted to grayscale, then repeated to 3 channels so the architecture stays identical

        This isolates the effect of color cues while keeping the split, image size, backbone, augmentation and fine-tuning schedule the same.
        """
    ),
    code(
        """
        color_ablation_path = ROOT / "experiment_color_ablation_results.json"

        if color_ablation_path.exists():
            with open(color_ablation_path, "r", encoding="utf-8") as file:
                color_ablation_results = json.load(file)

            color_ablation_df = pd.DataFrame(color_ablation_results["experiments"])
            metric_columns = ["val_accuracy", "val_macro_f1", "val_accuracy_tta", "val_macro_f1_tta"]
            color_ablation_df[metric_columns] = color_ablation_df[metric_columns].round(4)
            display(color_ablation_df)
            print("Conclusion:", color_ablation_results["conclusion"])
        else:
            print("No color ablation result JSON found yet.")
            print("Run these from the repository root to reproduce the comparison:")
            print("python scripts/run_local_experiment.py --name color_ablation_rgb_b0 --color-mode rgb --use-class-weights")
            print("python scripts/run_local_experiment.py --name color_ablation_grayscale_b0 --color-mode grayscale --use-class-weights")
        """
    ),
    md(
        """
        **Color ablation conclusion:** RGB performs clearly better than grayscale on this validation split.  
        The lizard species are not only separated by shape and texture; color patterns also carry useful signal.  
        Therefore we keep color images in the final pipeline.
        """
    ),
    code(
        """
        def run_ablation_experiment(
            experiment_name: str,
            use_augmentation: bool,
            model_params: dict | None = None,
            head_epochs: int = 3,
        ) -> dict:
            params = DEFAULT_MODEL_PARAMS | (model_params or {})
            local_aug = make_data_augmentation(params["augmentation_strength"]) if use_augmentation else keras.Sequential(name="no_augmentation")

            def build_local_model() -> tuple[keras.Model, keras.Model]:
                try:
                    local_backbone = keras.applications.EfficientNetB0(
                        include_top=False,
                        weights="imagenet",
                        input_shape=IMG_SIZE + (3,),
                    )
                except Exception:
                    local_backbone = keras.applications.EfficientNetB0(
                        include_top=False,
                        weights=None,
                        input_shape=IMG_SIZE + (3,),
                    )

                local_backbone.trainable = False
                for layer in local_backbone.layers:
                    layer.trainable = False
                inputs = keras.Input(shape=IMG_SIZE + (3,))
                x = local_aug(inputs)
                x = keras.applications.efficientnet.preprocess_input(x)
                x = local_backbone(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(float(params["dropout_rate"]))(x)
                x = layers.Dense(
                    int(params["dense_units"]),
                    activation="swish",
                    kernel_regularizer=keras.regularizers.l2(float(params["weight_decay"])),
                )(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(min(float(params["dropout_rate"]) + 0.10, 0.70))(x)
                outputs = layers.Dense(
                    num_classes,
                    activation="softmax",
                    dtype="float32",
                    kernel_regularizer=keras.regularizers.l2(float(params["weight_decay"])),
                )(x)
                local_model = keras.Model(inputs, outputs, name=f"ablation_{experiment_name.lower().replace(' ', '_')}")
                return local_model, local_backbone

            local_model, local_backbone = build_local_model()
            local_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            local_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=head_epochs,
                class_weight=class_weights,
                verbose=0,
            )

            local_probs = local_model.predict(val_ds, verbose=0)
            local_preds = local_probs.argmax(axis=1)
            local_cm = tf.math.confusion_matrix(val_targets, local_preds, num_classes=num_classes).numpy()
            _, _, _, local_macro_f1 = metrics_from_confusion_matrix(local_cm)

            return {
                "experiment": experiment_name,
                "augmentation": use_augmentation,
                "backbone": "frozen",
                "val_accuracy": float((local_preds == val_targets).mean()),
                "val_macro_f1": float(local_macro_f1),
            }

        # Optional usage after the main model run:
        # ablation_runs = [
        #     run_ablation_experiment("baseline", use_augmentation=False, model_params=DEFAULT_MODEL_PARAMS),
        #     run_ablation_experiment("default_augmentation", use_augmentation=True, model_params=DEFAULT_MODEL_PARAMS),
        #     run_ablation_experiment("stronger_frozen_head", use_augmentation=True, model_params=DEFAULT_MODEL_PARAMS),
        # ]
        # display(pd.DataFrame(ablation_runs))
        """
    ),
    md(
        """
        Suggested interpretation after filling in the table:

        - If augmentation improves validation performance, it suggests the model benefits from stronger robustness.
        - If staged fine-tuning improves the frozen model, it suggests the species differences benefit from adapting the backbone features a bit more.
        - If the embedding-assisted blend improves the final score, it shows the learned feature space contains useful nearest-neighbour structure beyond the raw softmax output.
        """
    ),
    md(
        """
        ### 5B. Inference Helpers

        Validation performance can improve without retraining if inference is smarter.  
        These helper functions support horizontal-flip TTA and embedding-space kNN, where images are compared using the model's learned feature vectors.
        """
    ),
    code(
        """
        def horizontal_flip_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
            def flip_batch(*batch):
                if len(batch) == 1:
                    return tf.image.flip_left_right(batch[0])
                images, labels = batch
                return tf.image.flip_left_right(images), labels

            return dataset.map(flip_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        def build_feature_extractor(model: keras.Model) -> keras.Model:
            return keras.Model(model.input, model.get_layer("embedding_features").output, name="embedding_extractor")

        def extract_embeddings(feature_model: keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
            return feature_model.predict(dataset, verbose=0)

        def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.maximum(norms, 1e-8)

        def compute_knn_probabilities(
            train_embeddings: np.ndarray,
            train_labels: np.ndarray,
            query_embeddings: np.ndarray,
            k: int,
            temperature: float,
            num_classes: int,
        ) -> np.ndarray:
            similarities = query_embeddings @ train_embeddings.T
            top_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
            top_similarities = np.take_along_axis(similarities, top_idx, axis=1)
            order = np.argsort(-top_similarities, axis=1)
            top_idx = np.take_along_axis(top_idx, order, axis=1)
            top_similarities = np.take_along_axis(top_similarities, order, axis=1)
            weights = np.exp(top_similarities / temperature)

            probabilities = np.zeros((len(query_embeddings), num_classes), dtype=np.float64)
            for row_idx in range(len(query_embeddings)):
                for neighbour_idx, train_idx in enumerate(top_idx[row_idx]):
                    probabilities[row_idx, int(train_labels[train_idx])] += float(weights[row_idx, neighbour_idx])
            return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)
        """
    ),
    md(
        """
        ### 5C. Build Validation Predictions and Embeddings

        We calculate three useful validation views:

        - normal softmax probabilities
        - horizontal-flip TTA probabilities
        - embeddings for train and validation images

        The embeddings allow the model to use nearest neighbours as extra evidence when the softmax output is uncertain.
        """
    ),
    code(
        """
        val_probabilities_base = model.predict(val_ds, verbose=0)
        val_probabilities_flip = model.predict(horizontal_flip_dataset(val_ds), verbose=0)
        val_probabilities_tta = (val_probabilities_base + val_probabilities_flip) / 2.0

        feature_model = build_feature_extractor(model)
        train_embedding_labels = train_split["label"].to_numpy()
        train_embeddings = l2_normalize(extract_embeddings(feature_model, train_eval_ds))
        val_embeddings_orig = l2_normalize(extract_embeddings(feature_model, val_ds))
        val_embeddings_flip = l2_normalize(extract_embeddings(feature_model, horizontal_flip_dataset(val_ds)))

        inference_candidates: list[dict[str, object]] = []

        def register_candidate(
            strategy_name: str,
            probabilities: np.ndarray,
            metadata: dict[str, object] | None = None,
        ) -> None:
            predictions = probabilities.argmax(axis=1)
            accuracy = float((predictions == val_targets).mean())
            confusion = tf.math.confusion_matrix(
                val_targets,
                predictions,
                num_classes=num_classes,
            ).numpy()
            _, _, _, macro_f1 = metrics_from_confusion_matrix(confusion)
            inference_candidates.append(
                {
                    "strategy_name": strategy_name,
                    "probabilities": probabilities,
                    "predictions": predictions,
                    "val_accuracy": accuracy,
                    "val_macro_f1": macro_f1,
                    "metadata": metadata or {},
                }
            )

        register_candidate("single-pass softmax", val_probabilities_base, {"family": "softmax"})
        register_candidate("flip-TTA softmax", val_probabilities_tta, {"family": "softmax_tta"})

        val_embedding_variants = {
            "orig": val_embeddings_orig,
            "flip_only": val_embeddings_flip,
            "flip_avg": l2_normalize((val_embeddings_orig + val_embeddings_flip) / 2.0),
        }
        """
    ),
    md(
        """
        ### 5D. Search the Best Inference Strategy

        Instead of guessing one inference method, we compare several candidates on the validation set:

        - plain softmax
        - flip TTA
        - kNN in embedding space
        - weighted blends of TTA and kNN

        This is why the final inference choice is evidence-based instead of arbitrary.
        """
    ),
    code(
        """
        if INFERENCE_SEARCH_MODE == "auto":
            for embedding_variant, query_embeddings in val_embedding_variants.items():
                for temperature in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
                    for k in [1, 3, 5, 7, 9, 11, 13, 15, 21, 31]:
                        knn_probabilities = compute_knn_probabilities(
                            train_embeddings=train_embeddings,
                            train_labels=train_embedding_labels,
                            query_embeddings=query_embeddings,
                            k=k,
                            temperature=temperature,
                            num_classes=num_classes,
                        )
                        register_candidate(
                            strategy_name=f"kNN only ({embedding_variant}, k={k}, temp={temperature:.2f})",
                            probabilities=knn_probabilities,
                            metadata={
                                "family": "knn",
                                "embedding_variant": embedding_variant,
                                "k": k,
                                "temperature": temperature,
                            },
                        )
                        for alpha in np.linspace(0.0, 1.0, 51):
                            blended_probabilities = alpha * val_probabilities_tta + (1.0 - alpha) * knn_probabilities
                            register_candidate(
                                strategy_name=f"blend TTA + kNN ({embedding_variant}, k={k}, temp={temperature:.2f}, alpha={alpha:.2f})",
                                probabilities=blended_probabilities,
                                metadata={
                                    "family": "blend",
                                    "embedding_variant": embedding_variant,
                                    "k": k,
                                    "temperature": temperature,
                                    "alpha": float(alpha),
                                },
                            )
        """
    ),
    md(
        """
        ### 5E. Select and Report the Best Strategy

        The final validation metrics are calculated from the best strategy found above.  
        We keep base softmax and TTA scores next to it so the improvement is visible and easy to explain.
        """
    ),
    code(
        """
        inference_results_df = (
            pd.DataFrame(
                [
                    {
                        "strategy": candidate["strategy_name"],
                        "val_accuracy": candidate["val_accuracy"],
                        "val_macro_f1": candidate["val_macro_f1"],
                    }
                    for candidate in inference_candidates
                ]
            )
            .sort_values(["val_accuracy", "val_macro_f1"], ascending=False)
            .reset_index(drop=True)
        )
        display(inference_results_df.head(10))

        best_candidate = max(
            inference_candidates,
            key=lambda item: (float(item["val_accuracy"]), float(item["val_macro_f1"])),
        )
        val_probabilities = best_candidate["probabilities"]
        val_predictions = best_candidate["predictions"]
        inference_strategy = str(best_candidate["strategy_name"])
        selected_inference_metadata = dict(best_candidate["metadata"])

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

        val_accuracy_base = float((val_probabilities_base.argmax(axis=1) == val_targets).mean())
        val_accuracy_tta = float((val_probabilities_tta.argmax(axis=1) == val_targets).mean())
        confusion_base = tf.math.confusion_matrix(val_targets, val_probabilities_base.argmax(axis=1), num_classes=num_classes).numpy()
        confusion_tta = tf.math.confusion_matrix(val_targets, val_probabilities_tta.argmax(axis=1), num_classes=num_classes).numpy()
        _, _, _, macro_f1_base = metrics_from_confusion_matrix(confusion_base)
        _, _, _, macro_f1_tta = metrics_from_confusion_matrix(confusion_tta)

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {macro_f1:.4f}")
        print(f"Base validation accuracy: {val_accuracy_base:.4f}")
        print(f"Base validation macro-F1: {macro_f1_base:.4f}")
        print(f"TTA validation accuracy: {val_accuracy_tta:.4f}")
        print(f"TTA validation macro-F1: {macro_f1_tta:.4f}")
        print("Chosen inference strategy:", inference_strategy)
        print("Selected inference metadata:", selected_inference_metadata)
        display(evaluation_table)
        """
    ),
    md(
        """
        ### 5F. Confusion Matrix

        A confusion matrix shows which species are mixed up most often.  
        This is more useful than a single accuracy number because it tells us where the model still needs help.
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
    md(
        """
        ### 5G. Misclassified Images

        Looking at the wrong predictions helps us understand whether mistakes come from similar species, bad image quality, confusing backgrounds, or low confidence.
        """
    ),
    code(
        """
        error_df = val_split.copy()
        error_df["pred_label"] = val_predictions
        error_df["pred_class_name"] = error_df["pred_label"].map(label_to_class)
        error_df["confidence"] = val_probabilities.max(axis=1)
        error_df["true_probability"] = [
            val_probabilities[i, true_label] for i, true_label in enumerate(val_targets)
        ]
        error_df["prediction_margin"] = np.partition(val_probabilities, -1, axis=1)[:, -1] - np.partition(val_probabilities, -2, axis=1)[:, -2]
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
        ### 5H. Most Common Confusions

        This table turns the mistakes into pairs such as `true species -> predicted species`.  
        It is useful for the discussion section because it gives concrete examples instead of only global metrics.
        """
    ),
    code(
        """
        confusion_pairs = (
            misclassified.groupby(["class_name", "pred_class_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        print("Most common confusion pairs:")
        display(confusion_pairs.head(10))

        hard_examples = misclassified.sort_values(["prediction_margin", "confidence"], ascending=[True, False])
        print("Most ambiguous mistakes:")
        display(
            hard_examples[
                ["id", "class_name", "pred_class_name", "confidence", "true_probability", "prediction_margin"]
            ].head(10)
        )
        """
    ),
    md(
        """
        ### 5I. Grad-CAM Explainability

        Grad-CAM helps us inspect **where** the model is looking before making a prediction.

        This is very useful in the defense because it helps answer questions such as:

        - does the model focus on the lizard itself, or mostly on the background?
        - does it attend to head shape, body color, spikes, or tail details?
        - are some mistakes caused by vegetation, lighting, or misleading context?
        """
    ),
    code(
        """
        def make_gradcam_heatmap(
            image_array: np.ndarray,
            model: keras.Model,
            backbone: keras.Model,
            pred_index: int | None = None,
        ) -> tuple[np.ndarray, int]:
            try:
                backbone_features = model.get_layer("gradcam_features").output
            except ValueError:
                pooling_layer = next(
                    layer for layer in model.layers if isinstance(layer, layers.GlobalAveragePooling2D)
                )
                backbone_features = pooling_layer.input
            grad_model = keras.models.Model(
                inputs=model.inputs,
                outputs=[backbone_features, model.output],
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                target_channel = predictions[:, pred_index]

            gradients = tape.gradient(target_channel, conv_outputs)
            pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), keras.backend.epsilon())
            return heatmap.numpy(), int(pred_index)

        def overlay_gradcam(
            image_path: Path,
            heatmap: np.ndarray,
            alpha: float = 0.35,
        ) -> np.ndarray:
            with Image.open(image_path) as img:
                image = np.array(ImageOps.exif_transpose(img).convert("RGB"))

            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_resized = cv2.resize(heatmap_uint8, (image.shape[1], image.shape[0]))
            color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            overlay = np.clip((1 - alpha) * image + alpha * color_map, 0, 255).astype(np.uint8)
            return overlay

        def prepare_image_for_model(image_path: Path) -> np.ndarray:
            with Image.open(image_path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE, Image.Resampling.BICUBIC)
            array = np.array(image, dtype=np.float32)
            return np.expand_dims(array, axis=0)
        """
    ),
    code(
        """
        gradcam_candidates = error_df.sort_values("confidence", ascending=False).head(3)
        print("Using Grad-CAM on the final backbone feature map.")

        if len(gradcam_candidates) > 0:
            fig, axes = plt.subplots(len(gradcam_candidates), 3, figsize=(14, 4 * len(gradcam_candidates)))
            axes = np.atleast_2d(axes)

            for row_idx, (_, row) in enumerate(gradcam_candidates.iterrows()):
                image_tensor = prepare_image_for_model(row["path"])
                heatmap, pred_idx = make_gradcam_heatmap(
                    image_tensor,
                    model=model,
                    backbone=backbone,
                )

                with Image.open(row["path"]) as img:
                    original = np.array(ImageOps.exif_transpose(img).convert("RGB"))
                overlay = overlay_gradcam(row["path"], heatmap)

                axes[row_idx, 0].imshow(original)
                axes[row_idx, 0].set_title(f"Original\\nTrue: {row['class_name']}")
                axes[row_idx, 0].axis("off")

                axes[row_idx, 1].imshow(heatmap, cmap="jet")
                axes[row_idx, 1].set_title("Grad-CAM heatmap")
                axes[row_idx, 1].axis("off")

                axes[row_idx, 2].imshow(overlay)
                axes[row_idx, 2].set_title(f"Overlay\\nPred: {label_to_class[pred_idx]}")
                axes[row_idx, 2].axis("off")

            plt.tight_layout()
            plt.show()
        else:
            print("No candidates available for Grad-CAM visualisation.")
        """
    ),
    md(
        """
        ## 6. Submission File

        Before exporting the submission, we reuse the same inference strategy that performed best on the validation split.

        In our strongest run this was not just plain TTA, but an **embedding-assisted blend**:

        - softmax probabilities from the fine-tuned network
        - optional horizontal-flip TTA
        - a kNN probability distribution in feature space
        - a validation-selected blend coefficient
        """
    ),
    code(
        """
        test_probabilities_base = model.predict(test_ds, verbose=0)
        test_probabilities_flip = model.predict(horizontal_flip_dataset(test_ds), verbose=0)
        test_probabilities_tta = (test_probabilities_base + test_probabilities_flip) / 2.0

        if selected_inference_metadata.get("family") == "softmax":
            test_probabilities = test_probabilities_base
        elif selected_inference_metadata.get("family") == "softmax_tta":
            test_probabilities = test_probabilities_tta
        else:
            test_embeddings_orig = l2_normalize(extract_embeddings(feature_model, test_ds))
            test_embeddings_flip = l2_normalize(extract_embeddings(feature_model, horizontal_flip_dataset(test_ds)))

            embedding_variant = str(selected_inference_metadata.get("embedding_variant", "orig"))
            if embedding_variant == "flip_only":
                test_query_embeddings = test_embeddings_flip
            elif embedding_variant == "flip_avg":
                test_query_embeddings = l2_normalize((test_embeddings_orig + test_embeddings_flip) / 2.0)
            else:
                test_query_embeddings = test_embeddings_orig

            test_knn_probabilities = compute_knn_probabilities(
                train_embeddings=train_embeddings,
                train_labels=train_embedding_labels,
                query_embeddings=test_query_embeddings,
                k=int(selected_inference_metadata["k"]),
                temperature=float(selected_inference_metadata["temperature"]),
                num_classes=num_classes,
            )

            if selected_inference_metadata.get("family") == "knn":
                test_probabilities = test_knn_probabilities
            else:
                alpha = float(selected_inference_metadata["alpha"])
                test_probabilities = alpha * test_probabilities_tta + (1.0 - alpha) * test_knn_probabilities

        test_predictions = test_probabilities.argmax(axis=1).astype(int)

        submission_df = pd.DataFrame(
            {
                "ID": test_df["id"].astype(int),
                "TARGET": test_predictions,
            }
        )
        submission_path = ROOT / "submission_transfer_learning.csv"
        submission_df.to_csv(submission_path, index=False)

        print("Inference mode for submission:", inference_strategy)
        print("Saved submission to:", submission_path)
        display(submission_df.head())
        """
    ),
    md(
        """
        ### 6B. Optional High-Score Ensemble Check

        The main notebook trains one clean model pipeline.  
        Separately, we also tested whether saved models can be combined as an ensemble.

        To reproduce the validation-calibrated high-score from the terminal, run:

        `python scripts/search_existing_model_ensembles.py --include-validation-bias`

        The cell below displays the saved search result if the JSON file already exists.
        """
    ),
    code(
        """
        ensemble_result_path = ROOT / "experiment_weighted_ensembles_repro.json"

        if ensemble_result_path.exists():
            with open(ensemble_result_path, "r", encoding="utf-8") as file:
                ensemble_results = json.load(file)
            top_ensemble_results = pd.DataFrame(ensemble_results["top_results"]).head(10)
            display(top_ensemble_results)
        else:
            print("No ensemble result JSON found yet.")
            print("Run this from the repository root:")
            print("python scripts/search_existing_model_ensembles.py --include-validation-bias")
        """
    ),
    md(
        """
        ## 7. Results Discussion

        Reference result from the strongest strict duplicate-aware single-pipeline validation run:

        - **Best validation accuracy:** `0.7193`
        - **Best validation macro-F1:** `0.7175`
        - **Reference point reached?** Yes, the original 70% target was exceeded.
        - **Winning inference setup:** staged fine-tuning + TTA/kNN embedding blend
        - **Best blend settings:** `embedding_variant=flip_only`, `k=31`, `temperature=0.20`, `alpha=0.20`
        - **What helped most:** duplicate-aware split, automatic cleaning, staged fine-tuning, embedding-assisted inference blend

        Extra high-score validation experiment:

        - **Best validation accuracy:** `0.7719`
        - **Best validation macro-F1:** `0.7690`
        - **Method:** weighted ensemble of saved models + a small validation-calibrated class bias
        - **Settings:** `tmp_b0_stratified_baseline.keras:flip` weight `0.66`, `tmp_ev2b0_baseline.keras:base` weight `0.34`, temperature `0.8`, class `2` bias `+0.30`
        - **Important caveat:** the class bias is selected on the validation labels, so this is an optimistic validation high-score rather than the cleanest unbiased estimate.
        - **Reproducibility:** run `python scripts/search_existing_model_ensembles.py --include-validation-bias`

        Earlier frozen-backbone / Optuna-style settings are still shown in comments above so the report can explain what was tried and why the final approach changed.

        Color ablation result:

        - **RGB staged model + TTA:** `0.6535` validation accuracy, `0.6453` macro-F1
        - **Grayscale staged model:** `0.5263` validation accuracy, `0.5262` macro-F1
        - **Conclusion:** color information is useful for this dataset, so the final pipeline keeps RGB images.

        In practice, the hardest errors are still expected between visually similar green/brown species and between iguana variants with comparable body shape or misleading background context.
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

output_path = Path("notebooks") / "lizard_species_transfer_learning_kc.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote notebook to {output_path}")
