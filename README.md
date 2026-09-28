# Read level cell type classification with MethylBERT

**Can one DNA methylation read tell us which of 39 human cell types it came from?** We fine-tuned MethylBERT on the Loyfer et al. (2023) atlas to find out. For single reads with hard labels, the answer is mostly no.

1. Our first 39 class model scored 99%, but the marker region alone gave away the label; after we removed that leak, accuracy within families of related cell types stayed near chance across 45 runs.
2. Only about 2% of reads fall in a marker region of their own cell type, and 35% of test reads in one family have an identical training read with a different label, so a single hard label per read has little to fit.
3. We split the atlas into lineage families to make the task smaller; it helped in some families (Excitable 0.55, chance 0.33) and barely in others (Structural 0.25, chance 0.17).
4. The group that built MethylBERT then completed the goal: read classification across all 39 types, using data driven soft labels in place of hard ones ([Rizdvanetskyi et al., 2026](https://arxiv.org/abs/2607.04987)).

**Full write-up: [REPORT.md](REPORT.md)**. All 45 runs: [Results/](Results/README.md).

## Results

| Task | Classes | Chance | Accuracy | Macro F1 | Note |
| --- | --- | --- | --- | --- | --- |
| Flat 39 class, first split | 39 | 0.026 | 0.991 | 0.969 | Leak: region ID alone predicts the label |
| Flat 39 class, region balanced | 39 | 0.026 | 0.520 | 0.511 | Matches what the region alone would score (about 0.50) |
| Pancreas acinar vs duct | 2 | 0.500 | 0.833 | 0.825 | Upper bound; split not saved |
| Excitable (cardiomyocyte, neuron, oligodendrocyte) | 3 | 0.333 | 0.548 | 0.542 | Upper bound; read level split |
| Digestive epithelia | 4 | 0.250 | 0.395 | 0.386 | |
| Reproductive epithelia | 4 | 0.250 | 0.378 to 0.394 | 0.372 to 0.382 | 2 runs |
| Blood | 6 | 0.167 | 0.198 to 0.430 | 0.189 to 0.348 | 19 runs; 0.430 is on an unbalanced test set |
| Surface epithelia | 6 | 0.167 | 0.276 to 0.277 | 0.267 to 0.273 | 3 runs |
| Metabolic (liver, pancreas) | 6 | 0.167 | 0.244 to 0.273 | 0.239 to 0.270 | 3 runs |
| Structural (fibroblast, muscle, fat, endothelium) | 6 | 0.167 | 0.217 to 0.246 | 0.175 to 0.228 | 6 runs |

- Accuracy is per read, from the checkpoint with the lowest eval loss.
- Pure samples, which should resolve to one class, came out near uniform or collapsed onto one class ([details](Results/README.md#pure-sample-deconvolution)).

## How train and test were split

- **First split** (the 0.991 row): 80/20 random split of reads within each cell type (`Training/Toolbox/split_data_simple.py`).
- **Later splits**: 75/25 with `Training/Toolbox/split_data_flexible.py`.
  - Whole samples go to one side only for cell types with 4 or more samples.
  - Cell types with 3 or fewer samples are pooled and split by unique read.
  - Which mode built each dataset was not recorded.
- **The Excitable split behaves like a read level split.** 49% of its test reads have an identical read in training ([audit](Results/audit_results.txt)).
- **So reads from the same donor can sit on both sides.** That inflates accuracy. The near chance results stand, but 0.83 and 0.55 are upper bounds.

## Quick start

```bash
git clone https://github.com/EG-xry/CellClassification_MethylBERT.git
cd CellClassification_MethylBERT
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo/run_demo.py
```

- The demo fine-tunes for 60 steps on 780 reads from the Excitable family (`demo/data/`), then evaluates.
- It downloads the pretrained `hanyangii/methylbert_hg19_2l` checkpoint on first use.
- It takes about 40 seconds on an Apple M4 Max and writes metrics and plots to `demo/output/`.
- It shows that the pipeline runs; 60 steps is not a result.

## Repository

```
Preprocessing/          .pat.gz reads + marker regions -> training rows
  analyze_methylation_batch_optimized.py   main script
  Config/               sample-to-cell-type maps (206 atlas samples)
  Data/                 marker region lists (hg38_all_1000.csv, hg38_250.csv)
  Toolbox/              format converters and helpers
Training/
  run_training.py       fine-tuning entry point (python Training/run_training.py -c <config>)
  src/methylbert/       modified MethylBERT (see MODIFICATIONS.md and LICENSE there)
  Config/               configs for each loss and family
  Toolbox/              combine, split, convert, and analysis tools
Testing/                pure-sample deconvolution and CTC scripts
Results/                all 45 runs, split audits, run index
Data/                   read/region agreement and cell type proportion tables
demo/                   one-command demo
REPORT.md               full write-up
```

## Full pipeline

Steps 2 to 4 ask for file names interactively (step 2 opens a small window if tkinter is available).

1. **Preprocess** each atlas sample into rows of `DNA_seq, Methyl_seq, DMR_Label, cType, DMR_cType, Read_Count`. Run from the folder holding the inputs (see Data sources):
   ```bash
   cd Preprocessing/Data
   python ../analyze_methylation_batch_optimized.py \
     --config ../Config/cell_types_config_collective.json --dedupe-reads --parallel 4
   ```
2. **Combine** the per-sample files of one family: `python Training/Toolbox/Combined_CSV.py`.
3. **Split** into train and test: `python Training/Toolbox/split_data_flexible.py`.
   - Mode 4 assigns whole samples by hand; use it to hold out donors.
4. **Convert** to the 4-column training format: `python Training/Toolbox/convert_data_format_flexible.py`.
5. **Train**: `python Training/run_training.py -c Training/Config/<config>.json`.
6. **Deconvolve** a pure sample: `python Testing/cell_type_deconvolution.py --csv <sample>.csv --model_dir <run>/bert.model`.

The CTC scripts in `Testing/` (`process_ctc_pat_files.py`, `predict_cell_type.py`) were written for circulating tumour cell data but never run to results. `predict_cell_type.py` does not pass methylation to the model; do not use it as is.

## Data sources

- **Methylation atlas**: Loyfer et al. (2023), GEO [GSE186458](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458).
  - 206 hg38 `.pat.gz` files, GSM5652176 to GSM5652382 (GSM5652260 not used).
  - Not included in this repository.
- **Marker regions**: `Preprocessing/Data/hg38_all_1000.csv` (about 1,000 per cell type) and `hg38_250.csv` (up to 250 per type).
  - Both derive from the Loyfer atlas markers.
  - The 250 list was lifted to hg38 with UCSC liftOver (`Preprocessing/Toolbox/hglft_genome_12e3e_1b8300.bed`).
  - The exact source table was not recorded.
- **Reference files** (not included):
  - hg38 genome FASTA (UCSC)
  - `CpG.bed.gz` CpG index in wgbstools format
- **Pretrained model**: `hanyangii/methylbert_hg19_{2,4,6}l` on Hugging Face (Jeong et al., 2025).

## Requirements

- Python 3.11.
- Pinned versions are in `requirements.txt`, checked by installing into a clean environment and running the demo.
- Training used CUDA GPUs; the demo also runs on Apple MPS or CPU.

## License and credit

- This repository's code is MIT licensed ([LICENSE](LICENSE)).
- `Training/src/methylbert/` is a modified copy of [MethylBERT](https://github.com/CompEpigen/methylbert) by Yunhee Jeong, MIT licensed.
  - The original license is kept in that folder.
  - The changes are listed in [MODIFICATIONS.md](Training/src/methylbert/MODIFICATIONS.md).
- `Figure_Loyfer.webp` is Figure 2 of Loyfer et al., *Nature* 613, 355–364 (2023), reproduced unchanged under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### References

- Jeong, Y. et al. (2025). MethylBERT enables read-level DNA methylation pattern identification and tumour deconvolution using a Transformer-based model. *Nature Communications* 16, 788.
- Loyfer, N. et al. (2023). A DNA methylation atlas of normal human cell types. *Nature* 613, 355–364.
- Rizdvanetskyi, D., Roos, N. and Lutsik, P. (2026). Data-driven soft labeling scales DNA read classification to whole-body cell-type deconvolution. arXiv:2607.04987.
