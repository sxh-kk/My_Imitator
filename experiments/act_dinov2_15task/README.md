# ACT＋DINOv2 15-task simulation reproduction

This directory runs official 15-task pretraining, seen/unseen evaluation, and equal-budget scratch versus pretrained few-shot adaptation. The upstream ACT/dataset/encoder/environment sources are unchanged.

Completed on 2026-10-09: all three conditions finished 100 epochs, with 80 formal evaluation cells and 800 valid episodes. Final terminal SR: seen 78%, unseen zero-shot 4%, scratch few-shot 85.5%, pretrained few-shot 73%. See [final results and limitations](../../LearningDocs/12_ACT_DINOV2_15TASK_RESULTS.md), [results.json](results.json), and [completion verification](checks/completion-verification.json).

Protocol and differences from the released launcher: [LearningDocs/11_ACT_DINOV2_15TASK_REPRODUCTION.md](../../LearningDocs/11_ACT_DINOV2_15TASK_REPRODUCTION.md).

`plan.json` is the protocol, `status.json` is the last progress snapshot, and `scripts/check_act15_status.py` verifies the actual process as well. Training: 100 epochs, batch256, frozen DINOv2-L with four encoded frames. One seed, simulation only. Sub-SR uses the explicitly authorized provisional threshold 0.95; raw phase peaks are preserved. Unchanged released target metadata statistics are available to evaluation and are disclosed as a protocol limitation.

Downloaded data, assets, feature tensors, checkpoint weights, and videos are local artifacts; logs, configs, provenance and completed numeric results can be published separately.

For GitHub publication, the three full iteration-level `runs/*/train.jsonl` files stay local. The original `runs/*/epochs.jsonl` summaries, run completion records, and per-episode evaluation data are included. The publication script generates `TRAINING_LOG_PROVENANCE.json` with the omitted iteration-log sizes and SHA-256 hashes. Source manifests also retain the recorded L3 evaluator amendment and the excluded original L3 records for traceability.
