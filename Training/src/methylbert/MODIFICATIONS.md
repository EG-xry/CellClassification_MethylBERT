# Changes to MethylBERT

This folder is a modified copy of MethylBERT 2.0.2 by Yunhee Jeong ([CompEpigen/methylbert](https://github.com/CompEpigen/methylbert), MIT License, see `LICENSE` in this folder). The original model classifies each read as tumour or normal. The changes below turn it into an N-class cell-type classifier. Fine-tuning still starts from the published `hanyangii/methylbert_hg19_*` pretrained checkpoints.

## Unchanged files

- `__init__.py`, `utils.py`, `deconvolute.py`
- `data/__init__.py`, `data/vocab.py`, `data/bam.py`, `data/genome.py`
- `deconvolute.py` still implements the original binary tumour-purity method. Cell-type deconvolution for this project lives in `Testing/cell_type_deconvolution.py`.

## Changed files

### `network.py`
- The classification head outputs `num_classes` logits instead of 2.
- The DMR embedding size is set separately from the number of classes (`num_dmr_embeddings`).
- Supported losses: `cross_entropy` (optional class weights and label smoothing), `focal` (multi-class softmax focal loss with per-class weights), `decoupled_focal` (one-vs-rest sigmoid focal loss with separate positive and negative alphas), `bce`.
- Dropout in the read classifier follows `hidden_dropout_prob` (upstream hard-coded 0.05).
- New setters: `set_class_weights`, `set_dropout`, `set_label_smoothing`.

### `function.py`
- Replaced the binary sigmoid focal loss with `softmax_focal_loss` (multi-class).
- Added `sigmoid_focal_loss_decoupled` and `DecoupledFocalLoss`.

### `config.py`
- New fields: `num_classes`, `num_dmr_embeddings`, `label_smoothing`, `focal_weight`, `focal_gamma`, `focal_alpha_pos`, `focal_alpha_neg`, `dropout`, `use_balanced_batches`, `use_balanced_test_eval`.
- Default loss changed to `decoupled_focal` in the model config and `cross_entropy` in the training config.

### `data/dataset.py`
- The label is the read's cell type, not the binary "matches the DMR's cell type" flag.
- Accepts either column order: `dna_seq, methyl_seq, ctype, dmr_label` or `dna_seq, methyl_seq, dmr_label, ctype`.
- Builds a sorted cell-type-to-integer map per file.
- Added `BalancedBatchSampler` (alias `ClassBalancedSampler`): each batch holds the same number of reads from every class.

### `trainer.py`
- Device selection: CUDA, then Apple MPS, then CPU. Mixed precision is off on MPS.
- Multi-class metrics: accuracy, macro and weighted F1, per-class accuracy, confusion matrix.
- Plots and reports written at the end of training: `training_curves.png`, confusion matrices, `final_metrics.txt`, `confidence_analysis.txt`.
- Two checkpoints: highest macro F1 (`bert_best.model/`) and lowest eval loss (`bert.model/`). The final report uses the lowest-loss model.
- Inverse-frequency class weights and focal alphas, computed from the training set when not given.
- Passes an attention mask so padding tokens are ignored.
- Removed interim `save_freq` checkpoints.
- Fixed upstream's `np.concatenate` call on evaluation logits.

### `cli.py`, `data/finetune_data_generate.py`
- `--num_classes` option, `cross_entropy` default loss, device message. Regex strings made raw.

## Known issues

These are documented, not fixed, because the saved results were produced with this code as is.

- **Classifier bias initialisation uses test-set class counts.** `_init_classifier_bias_from_validation` sets the final-layer bias from the class frequencies of the full test file. It leaks only the class proportions, not any read content, but it is still a use of test data during training.
- **Probabilities are passed through softmax twice.** `forward` already returns probabilities, and the trainer and `Testing/cell_type_deconvolution.py` apply softmax again. Argmax predictions and accuracy are unaffected. Reported confidence values are compressed toward uniform (at most about 0.35 for six classes).
- **Some config keys are ignored.** `use_decoupled_focal`, `dropout`, `seed`, and `use_safetensors` are accepted but have no effect. Dropout is 0.1 in every run.
- **Label maps are built per file.** Train and test files must contain exactly the same set of cell types.
- **Class-balanced test evaluation** (`use_balanced_test_eval: true`) scores a class-balanced subsample of the test set, not the whole test set.
- The CLI still lists `focal_bce` and `--save_freq`, which no longer do anything.
