#!/usr/bin/env python
"""
Create a ResNet50 variant of the EfficientNet notebook.

This keeps the notebook structure intact and only swaps the backbone-specific
parts so the comparison stays fair.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EFFICIENTNET_NB = ROOT / "notebooks" / "lizard_species_transfer_learning_kc.ipynb"
RESNET_NB = ROOT / "notebooks" / "lizard_species_transfer_learning_kc_resnet.ipynb"


def transform_notebook() -> None:
    """Transform the EfficientNet notebook into a ResNet50 variant."""
    with open(EFFICIENTNET_NB, "r", encoding="utf-8") as file:
        notebook = json.load(file)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)

        text = text.replace("EfficientNetB0", "ResNet50")
        text = text.replace("efficientnet", "resnet")
        text = text.replace("best_lizard_model.keras", "best_lizard_model_resnet50.keras")
        text = text.replace(
            "submission_transfer_learning.csv",
            "submission_transfer_learning_resnet50.csv",
        )
        text = text.replace("lizard_classifier", "lizard_classifier_resnet50")

        cell["source"] = text.split("\n")
        if cell["source"] and cell["source"][-1] == "":
            cell["source"].pop()

    with open(RESNET_NB, "w", encoding="utf-8") as file:
        json.dump(notebook, file, ensure_ascii=False, indent=1)

    print(f"Created {RESNET_NB}")


if __name__ == "__main__":
    transform_notebook()
