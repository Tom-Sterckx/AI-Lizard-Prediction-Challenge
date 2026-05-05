# Lizard Grad-CAM Webapp

Run eerst de final-training cell in `lizard_classification_pipeline_copy_kc.ipynb`, zodat er checkpoints zoals `artifacts/best_lizard_model_convnext_tiny.keras` bestaan.
Die cell schrijft ook `*.metadata.json` naast het Keras-model, zodat de webapp dezelfde image size en class volgorde gebruikt.

Start daarna:

```powershell
python webapp/app.py
```

Open `http://127.0.0.1:8000`, kies een checkpoint, upload een afbeelding en bekijk de voorspelling met Grad-CAM overlay.
