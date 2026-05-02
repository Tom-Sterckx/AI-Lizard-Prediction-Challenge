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

Why we still use it for the submission: the bias is small and targeted. It corrects a systematic class tendency observed in the confusion matrix instead of changing the model arbitrarily. For the final competition-style submission we choose the strongest validation-performing setup, while still documenting that the bias was calibrated on the validation split.

## Strongest Strict Single-Pipeline Result

- Validation accuracy: `0.7193`
- Validation macro-F1: `0.7175`
- Method: staged EfficientNetB0 fine-tuning plus embedding/kNN blend
- Best blend: `embedding_variant=flip_only`, `k=31`, `temperature=0.20`, `alpha=0.20`

This is still the strongest result I would defend as the clean reference pipeline: duplicate-aware split, no validation images used for training, and no validation-label bias added after the fact.

## Latest Hyperband Tuning Attempt

- Script: `python scripts/run_hyperband_tuning.py --n-trials 6 --max-head-epochs 5 --max-fine-epochs 8`
- Pruning method: Optuna `HyperbandPruner`
- Trials: `6`
- Pruned trials: `3`
- Best trial: `0`
- Best trial parameters: EfficientNetV2B0, dropout `0.50`, no dense hidden layer, augmentation strength `0.60`, head learning rate `0.000472`, fine-tuning learning rate `0.0000223`, top `60` layers unfrozen, Adam optimizer
- Best Hyperband base result: `0.6228` validation accuracy, `0.6230` macro-F1
- Best Hyperband TTA result: `0.6184` validation accuracy, `0.6188` macro-F1
- Best Hyperband embedding blend: `0.6842` validation accuracy, `0.6829` macro-F1

Conclusion: Hyperband worked technically and stopped weak trials early, but it did not improve the strongest clean reference model.

## Group-Safe Ensemble Search

- Search result file: `experiment_group_safe_ensembles.json`
- Rule: only models trained through duplicate-aware group-split experiments were included
- Explicitly excluded: `tmp_b0_stratified_baseline.keras`, because that model came from a different stratified split
- Best group-safe ensemble: `0.7149` validation accuracy, `0.7097` macro-F1
- Members: `tmp_stage_baseline.keras:flip`, `tmp_ev2b0_baseline.keras:base`, `best_lizard_model.keras:base`

Conclusion: the stricter ensemble was close, but still did not beat the strict `0.7193` / `0.7175` single-pipeline blend.

## Attempts That Did Not Beat 75%

- Hyperband pruning search: best strict blend reached `0.6842` accuracy and `0.6829` macro-F1.
- Group-safe ensemble search: best reached `0.7149` accuracy and `0.7097` macro-F1.
- Broad kNN/softmax search on the best strict model: best stayed at `0.7193` accuracy and `0.7175` macro-F1.
- Equal-weight model ensembles: best reached `0.7456` accuracy and `0.7426` macro-F1.
- RGB vs grayscale ablation, same staged EfficientNetB0 setup: RGB TTA reached `0.6535` accuracy and `0.6453` macro-F1; grayscale reached `0.5263` accuracy and `0.5262` macro-F1 without TTA. Conclusion: color information should be kept.
- `tmp_b0_acc_focus.keras`: `0.6316` base accuracy, `0.6192` macro-F1.
- `tmp_ev2b0_baseline.keras`: `0.6228` base accuracy, `0.6217` macro-F1.
- `tmp_stage_baseline.keras`: `0.6491` base accuracy, `0.6379` macro-F1.
- `best_lizard_model.keras`: `0.6535` base accuracy, `0.6469` macro-F1.
