#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_ph.py
------------------------------------------
Διαβάζει τα Excel:
  - MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx
  - MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx

Κάθε φύλλο (SubXX) έχει σήματα σε ΣΤΗΛΕΣ με ~1001 τιμές από κάτω.
Ο κώδικας εντοπίζει στήλες με "μακρύ" αμιγώς αριθμητικό τμήμα,
εξάγει την (έως) 1001-σημεία κυματομορφή, υπολογίζει persistent
homology (H1) με ripser και αποθηκεύει:
  - CSV (gc_percent vs value)
  - H1 persistence diagram (PNG)
  - H1 barcode (PNG, με fallback αν λείπει plot_barcodes)
  - summary ανά subject + merged all_subjects_summary.csv

Χρήση:
  python xlsx_to_ph.py \
      --xlsx ~/topology/MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx \
      --out  ~/topology/out_post_xlsx_col \
      --label_prefix post

  python xlsx_to_ph.py \
      --xlsx ~/topology/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx \
      --out  ~/topology/out_healthy_xlsx_col \
      --label_prefix healthy

Απαιτήσεις (requirements):
  pandas, numpy, openpyxl, matplotlib, ripser, persim
"""

import os, re, argparse
import numpy as np
import pandas as pd


# ------------------------- helpers ------------------------- #
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def longest_numeric_run(vals: np.ndarray):
    """Επιστρέφει (len, start, stop) του μεγαλύτερου συνεχούς τμήματος με finite τιμές."""
    isn = np.isfinite(vals)
    best = (0, -1, -1)
    i, n = 0, len(vals)
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        if j - i > best[0]:
            best = (j - i, i, j)
        i = j
    return best  # (L, a, b)


def takens_embedding(sig: np.ndarray, m: int = 8, tau: int = 5):
    sig = np.asarray(sig, float).ravel()
    N = len(sig)
    L = N - (m - 1) * tau
    if L <= 0:
        return None
    return np.stack([sig[i:i + L] for i in range(0, m * tau, tau)], axis=1)


def ripser_h1(X: np.ndarray):
    if X is None:
        return np.empty((0, 2))
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms']) > 1 else np.empty((0, 2))


def save_pd_and_barcode(dgm: np.ndarray, out_prefix: str, title: str = "H1"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from persim import plot_diagrams

    # Persistence diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Barcode (δοκίμασε persim.plot_barcodes, αλλιώς custom)
    try:
        from persim import plot_barcodes
        plt.figure()
        plot_barcodes([dgm], title=f"{title} Barcode")
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        plt.figure()
        y = 0
        for bd in np.asarray(dgm):
            if len(bd) < 2:
                continue
            b, de = float(bd[0]), float(bd[1])
            if np.isfinite(b) and np.isfinite(de) and de > b:
                plt.hlines(y, b, de)
                y += 1
        plt.xlabel("filtration")
        plt.ylabel("bars")
        plt.title(f"{title} Barcode (custom)")
        plt.tight_layout()
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()


def save_curve_csv(curve: np.ndarray, subject_id, side: str, varname: str, out_csv: str):
    t = np.linspace(0, 100, len(curve))
    df = pd.DataFrame({
        "subject": subject_id,
        "side": side,
        "variable": varname,
        "gc_percent": t,
        "value": curve
    })
    df.to_csv(out_csv, index=False)


# ------------------ name → side / variable ------------------ #
def detect_side_from_name(colname: str) -> str:
    n = (colname or "").lower()
    # Post-stroke ονοματολογία (Pside/Nside) ή γενικά labels
    if "pside" in n or "paretic" in n or re.search(r"\bP\b", n):
        return "P"
    if "nside" in n or ("non" in n and "pare" in n) or re.search(r"\bN\b", n):
        return "N"
    if "left" in n or re.search(r"\bL\b", n):
        return "L"
    if "right" in n or re.search(r"\bR\b", n):
        return "R"
    return "B"  # both/unknown (healthy Excel)


def detect_var_from_name(colname: str) -> str:
    n = (colname or "").lower()
    # βασικά gait signals (σαγητταία)
    if "ground" in n and "reaction" in n and ("z" in n or "vert" in n):
        return "GroundReactionForce_z"
    if "hip" in n and "angle" in n:
        return "HipAngles_x"
    if "knee" in n and "angle" in n:
        return "KneeAngles_x"
    if "ankle" in n and "angle" in n:
        return "AnkleAngles_x"
    # EMG
    for m in ["gas", "rf", "vl", "bf", "st", "ta", "ers"]:
        if re.search(rf"\b{m}\b", n):
            return m.upper() + "norm"
    # fallback: καθάρισε string
    clean = re.sub(r'\W+', '_', n).strip('_')
    return clean or "unknown"


# ---------------------- main extractor ---------------------- #
def process_xlsx_colwise(
    xlsx_path: str,
    out_root: str,
    label_prefix: str,
    min_run: int = 600,
    m: int = 8,
    tau: int = 5
):
    """
    Column-wise extractor:
      - header=0: ονόματα στηλών
      - βρίσκει long numeric runs (>= min_run)
      - κόβει/γεμίζει σε 1001 δείγματα
      - PH(H1) + διαγράμματα + CSV + per-subject summary + merged summary
    """
    import openpyxl  # noqa: F401 (χρειάζεται για engine='openpyxl')
    ensure_dir(out_root)
    xls = pd.ExcelFile(xlsx_path, engine='openpyxl')

    merged_rows = []

    for sheet in xls.sheet_names:
        if not str(sheet).lower().startswith('sub'):
            continue

        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        subj_match = re.search(r"(\d+)", str(sheet))
        subject_id = int(subj_match.group(1)) if subj_match else str(sheet)

        out_dir = os.path.join(out_root, f"{label_prefix}_sub{int(subject_id):03d}")
        ensure_dir(out_dir)

        per_subject = []

        for col in df.columns:
            series = df[col]
            # Μετατροπή σε numeric (μη αριθμητικά -> NaN)
            vals = pd.to_numeric(series, errors='coerce').to_numpy(float)

            L, a, b = longest_numeric_run(vals)
            if L < min_run:
                continue

            vec = vals[a:b]
            # trim/pad στα 1001 για συνέπεια
            if len(vec) >= 1001:
                vec = vec[:1001]
            else:
                pad = np.full(1001, np.nan)
                pad[:len(vec)] = vec
                vec = pad

            side = detect_side_from_name(str(col))
            var = detect_var_from_name(str(col))

            out_prefix = os.path.join(
                out_dir, f"{label_prefix}_sub{int(subject_id):03d}_{side}_{var}"
            )

            # CSV
            save_curve_csv(vec, subject_id, side, var, out_prefix + ".csv")

            # PH
            X = takens_embedding(vec, m=m, tau=tau)
            dgm1 = ripser_h1(X)
            save_pd_and_barcode(dgm1, out_prefix, f"{label_prefix} Sub{int(subject_id):03d} {side} {var}")

            total_persistence = (
                float(np.nansum(np.clip(dgm1[:, 1] - dgm1[:, 0], 0, None)))
                if dgm1.size else 0.0
            )

            per_subject.append({
                "subject": int(subject_id),
                "side": side,
                "variable": var,
                "h1_points": int(dgm1.shape[0]),
                "h1_total_persistence": total_persistence
            })

        # per-subject summary
        if per_subject:
            sub_sum = os.path.join(out_dir, f"{label_prefix}_sub{int(subject_id):03d}_summary.csv")
            pd.DataFrame(per_subject).to_csv(sub_sum, index=False)
            merged_rows.extend(per_subject)
        else:
            with open(os.path.join(out_dir, f"{label_prefix}_sub{int(subject_id):03d}_DEBUG.txt"), "w") as f:
                f.write("No column with a long numeric run was found in this sheet.\n")

    # merged summary
    if merged_rows:
        all_df = pd.DataFrame(merged_rows)
        all_df.to_csv(os.path.join(out_root, "all_subjects_summary.csv"), index=False)

    print(f"[DONE] Wrote output -> {out_root}")


# --------------------------- CLI ---------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Excel → CSV + PH (H1) (column-wise).")
    ap.add_argument("--xlsx", required=True, help="Διαδρομή του Excel (xlsx).")
    ap.add_argument("--out", required=True, help="Φάκελος εξόδου.")
    ap.add_argument("--label_prefix", default="group",
                    help="Πρόθεμα για ονόματα αρχείων (π.χ. 'post' ή 'healthy').")
    ap.add_argument("--min_run", type=int, default=600,
                    help="Ελάχιστο μήκος συνεχούς αριθμητικού τμήματος (default: 600).")
    ap.add_argument("--m", type=int, default=8, help="Takens embedding dimension (default: 8).")
    ap.add_argument("--tau", type=int, default=5, help="Takens delay (default: 5).")
    args = ap.parse_args()

    process_xlsx_colwise(
        xlsx_path=os.path.expanduser(args.xlsx),
        out_root=os.path.expanduser(args.out),
        label_prefix=args.label_prefix,
        min_run=args.min_run,
        m=args.m,
        tau=args.tau
    )


if __name__ == "__main__":
    main()