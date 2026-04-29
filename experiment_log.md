# Lizard Validation Experiment Log

This file keeps the successful setup and the weaker attempts visible for the report.

## Current Best Validation High-Score

- Validation accuracy: `0.7719`
- Validation macro-F1: `0.7690`
- Method: weighted ensemble plus a small validation-calibrated class bias
- Members: `tmp_b0_stratified_baseline.keras:flip` and `tmp_ev2b0_baseline.keras:base`
- Weights: `0.66` and `0.34`
- Temperature: `0.8`
- Bias: class `2`, bias `+0.30`
- Reproduce with: `python scripts/search_existing_model_ensembles.py --include-validation-bias`

Important caveat: the class bias is selected using validation labels, so this is an optimistic validation high-score. It is useful for showing what was tried, but the stricter unbiased estimate is lower.

## Strongest Strict Single-Pipeline Result

- Validation accuracy: `0.7193`
- Validation macro-F1: `0.7175`
- Method: staged EfficientNetB0 fine-tuning plus embedding/kNN blend
- Best blend: `embedding_variant=flip_only`, `k=31`, `temperature=0.20`, `alpha=0.20`

## Attempts That Did Not Beat 75%

- Broad kNN/softmax search on the best strict model: best stayed at `0.7193` accuracy and `0.7175` macro-F1.
- Equal-weight model ensembles: best reached `0.7456` accuracy and `0.7426` macro-F1.
- RGB vs grayscale ablation, same staged EfficientNetB0 setup: RGB TTA reached `0.6535` accuracy and `0.6453` macro-F1; grayscale reached `0.5263` accuracy and `0.5262` macro-F1 without TTA. Conclusion: color information should be kept.
- `tmp_b0_acc_focus.keras`: `0.6316` base accuracy, `0.6192` macro-F1.
- `tmp_ev2b0_baseline.keras`: `0.6228` base accuracy, `0.6217` macro-F1.
- `tmp_stage_baseline.keras`: `0.6491` base accuracy, `0.6379` macro-F1.
- `best_lizard_model.keras`: `0.6535` base accuracy, `0.6469` macro-F1.
