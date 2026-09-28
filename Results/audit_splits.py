"""
Read-only audits of the saved train/test splits. No model is trained or evaluated.

Each audit answers one question the report relies on:

  1. leaky   : In the early flat 39-class split, does the DMR region ID alone give away
               the cell type? (It does, which explains the 0.99 accuracy.)
  2. family  : In a family-level split, how many test reads have an identical read in
               training, and how often does that identical read carry a different label?
               (A direct count of the many-to-many read-to-cell-type mapping.)
  3. ontarget: In unbalanced preprocessing output for one sample, what fraction of reads
               fall in a marker region of their own cell type?

Usage (from the repo root):
  python Results/audit_splits.py leaky    --train ../data/train_seq.csv --test ../data/test_seq.csv
  python Results/audit_splits.py family   --train Data/train_Excitable.csv --test Data/test_Excitable.csv
  python Results/audit_splits.py ontarget --csv Preprocessing/Run_Result/skeletal_muscle_speed_dedupe_test.csv
"""
import argparse

import pandas as pd


def load_split(path):
    """Tab-separated training file with columns dna_seq, methyl_seq, ctype, dmr_label (any order)."""
    return pd.read_csv(path, sep="\t", dtype=str, usecols=["dna_seq", "methyl_seq", "ctype", "dmr_label"])


def dmr_lookup_accuracy(train, test):
    """Predict each test read's cell type as the most common training cell type of its DMR region."""
    majority = train.groupby("dmr_label")["ctype"].agg(lambda s: s.value_counts().index[0])
    pred = test["dmr_label"].map(majority)
    covered = pred.notna()
    return (pred[covered] == test.loc[covered, "ctype"]).mean(), covered.mean()


def audit_leaky(args):
    train, test = load_split(args.train), load_split(args.test)
    n_types = train.groupby("dmr_label")["ctype"].nunique()
    acc, cov = dmr_lookup_accuracy(train, test)
    key = ["dna_seq", "methyl_seq", "dmr_label"]
    dup = test.set_index(key).index.isin(train.set_index(key).index).mean()
    print(f"train rows {len(train):,}, test rows {len(test):,}, cell types {train['ctype'].nunique()}")
    print(f"DMR regions in train: {len(n_types):,}; regions with exactly one cell type: {(n_types == 1).mean():.2%}")
    print(f"DMR-lookup accuracy on test (region ID only, no methylation): {acc:.4f} (covers {cov:.2%} of test)")
    print(f"test rows with an identical (dna_seq, methyl_seq, dmr_label) row in train: {dup:.2%}")


def audit_family(args):
    train, test = load_split(args.train), load_split(args.test)
    key = ["dna_seq", "methyl_seq"]
    train_labels = train.groupby(key)["ctype"].agg(set)
    twins = test.set_index(key).index.map(lambda k: train_labels.get(k, set()))
    has_twin = pd.Series([len(t) > 0 for t in twins])
    conflict = pd.Series([len(t - {c}) > 0 for t, c in zip(twins, test["ctype"])])
    acc, cov = dmr_lookup_accuracy(train, test)
    n_types = pd.concat([train, test]).groupby("dmr_label")["ctype"].nunique()
    k = test["ctype"].nunique()
    print(f"train rows {len(train):,}, test rows {len(test):,}, cell types {k} (chance {1 / k:.3f})")
    print(f"DMR regions: {len(n_types):,}; regions containing every cell type: {(n_types == k).mean():.2%}")
    print(f"DMR-lookup accuracy on test (region ID only): {acc:.4f} (covers {cov:.2%} of test)")
    print(f"test rows with an identical (dna_seq, methyl_seq) read in train: {has_twin.mean():.2%}")
    print(f"test rows whose identical training read carries a different cell-type label: {conflict.mean():.2%}")


def audit_ontarget(args):
    df = pd.read_csv(args.csv, usecols=["cType", "DMR_cType", "Read_Count"], dtype={"cType": str, "DMR_cType": str})
    on = df["cType"] == df["DMR_cType"]
    reads = df["Read_Count"].astype(int)
    print(f"sample cell type(s): {sorted(df['cType'].unique())}")
    print(f"rows {len(df):,}, reads {reads.sum():,}, DMR label groups {df['DMR_cType'].nunique()}")
    print(f"on-target rows (read in a marker region of its own cell type): {on.mean():.2%}")
    print(f"on-target reads (weighted by Read_Count): {reads[on].sum() / reads.sum():.2%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="audit", required=True)
    for name in ("leaky", "family"):
        s = sub.add_parser(name)
        s.add_argument("--train", required=True)
        s.add_argument("--test", required=True)
    s = sub.add_parser("ontarget")
    s.add_argument("--csv", required=True)
    args = p.parse_args()
    {"leaky": audit_leaky, "family": audit_family, "ontarget": audit_ontarget}[args.audit](args)
