#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_ph.py
------------------------------------------
Διαβάζει post-processed Excel (xlsx) με stride-normalized δεδομένα
(ανά καρτέλα/subject), εξάγει καμπύλες (angles/EMG) ανά πλευρά,
και υπολογίζει persistent homology (H1) με Ripser.

Έξοδοι ανά subject:
- CSV καμπύλες (long format): subject,side,variable,gc_percent,value
- PNG: *_H1diagram.png, *_H1barcode.png
- subXXX_summary.csv
Επίσης στο root: all_subjects_summary.csv

Χρήση:
  python xlsx_to_ph.py --xlsx /path/to/file.xlsx --out out_folder --label_prefix post
"""

import argparse, os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

# -------------------- PH helpers --------------------
def takens_embedding(sig, m=8, tau=5):
    sig = np.asarray(sig, dtype=float).ravel()
    N = len(sig)
    L = N - (m - 1) * tau
    if L <= 0:
        raise ValueError("Signal too short for Takens (m,tau)")
    X = np.stack([sig[i:i+L] for i in range(0, m*tau, tau)], axis=1)
    return X

def ripser_h1_diagram(X):
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms']) > 1 else np.empty((0, 2))

def save_pd_and_barcode(dgm, out_prefix, title="H1"):
    import matplotlib.pyplot as plt
    from persim import plot_diagrams
    # Diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Barcode
    try:
        from persim import plot_barcodes
        plt.figure()
        plot_barcodes([dgm], title=f"{title} Barcode")
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        # fallback custom
        plt.figure()
        y = 0
        for b, d in np.asarray(dgm):
            if np.isfinite(b) and np.isfinite(d) and d > b:
                plt.hlines(y, b, d)
                y += 1
        plt.xlabel("filtration"); plt.ylabel("bars")
        plt.title(f"{title} Barcode (custom)")
        plt.tight_layout()
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()

# -------------------- CSV writer --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_curve_long(curve, subject_id, side, varname, out_csv):
    t = np.linspace(0, 100, len(curve))
    df = pd.DataFrame({
        'subject': subject_id,
        'side': side,
        'variable': varname,
        'gc_percent': t,
        'value': curve
    })
    df.to_csv(out_csv, index=False)

# -------------------- Column parsing --------------------
SIDE_PATTERNS = {
    'L': [r'\bL\b', r'_L\b', r'\.L\b', r'\(L\)', r'Left', r'\bLt\b'],
    'R': [r'\bR\b', r'_R\b', r'\.R\b', r'\(R\)', r'Right', r'\bRt\b'],
}
# ενδιαφέροντα variables
ANGLE_VARS = ['Hip', 'Knee', 'Ankle', 'Pelvis']  # sagittal μόνο
EMG_VARS   = ['GAS','RF','VL','BF','ST','TA','ERS']

def detect_side(colname):
    c = str(colname)
    for side, pats in SIDE_PATTERNS.items():
        for pat in pats:
            if re.search(pat, c, flags=re.IGNORECASE):
                return side
    return None  # αν δεν βρεθεί σαφής πλευρά

def normalize_var(colname):
    """Επιστρέφει standardized variable label ή None αν δεν μας ενδιαφέρει"""
    c = str(colname)
    # Angles
    for base in ANGLE_VARS:
        if re.search(fr'\b{base}\b', c, flags=re.IGNORECASE):
            return f"{base}Angles_x"  # sagittal
    # EMG
    for m in EMG_VARS:
        if re.search(fr'\b{m}\b', c, flags=re.IGNORECASE):
            return f"{m}norm" if re.search(r'norm', c, flags=re.IGNORECASE) else m
    return None

def is_time_column(name):
    return str(name).strip().lower() in {'gc_percent','gaitcycle','gait_cycle','time','t','percent'}

def pick_timeseries_columns(df):
    """
    Επιλέγει στήλες που είναι 1D καμπύλες (μήκος ~1001).
    Φιλτράρει μεταδεδομένα/χρονικές στήλες.
    """
    cols = []
    for c in df.columns:
        if is_time_column(c): 
            continue
        # πετάξει προφανώς μη-αριθμητικές στήλες/IDs
        s = pd.to_numeric(df[c], errors='coerce')
        # κράτα στήλες που έχουν αρκετά μη-NaN (~>=100)
        if np.isfinite(s).sum() >= 100:
            cols.append(c)
    return cols

# -------------------- Sheet -> curves --------------------
def extract_curves_from_sheet(df_sheet, subject_label):
    """
    Επιστρέφει list of tuples: (subject, side, varname, curve)
    """
    # Αν υπάρχει στήλη χρόνου, τη χρησιμοποιούμε μόνο για ταξινόμηση
    time_cols = [c for c in df_sheet.columns if is_time_column(c)]
    if time_cols:
        first_t = time_cols[0]
        try:
            df_sheet = df_sheet.sort_values(by=first_t, kind='mergesort')
        except Exception:
            pass

    candidates = pick_timeseries_columns(df_sheet)
    curves = []
    for c in candidates:
        var = normalize_var(c)
        if var is None:
            continue
        side = detect_side(c) or 'B'  # αν δεν βρεθεί πλευρά → 'B' (both/unspecified)
        s = pd.to_numeric(df_sheet[c], errors='coerce').to_numpy(dtype=float)
        # καθάρισε NaN στην αρχή/τέλος
        if np.isfinite(s).sum() < 50:
            continue
        # αν έχει περισσότερα από 1001, κόψε/παρε 1001
        if s.size > 1001:
            s = s[:1001]
        # αν έχει λιγότερα, κάνε simple linear interp σε 1001
        if s.size < 1001:
            x_old = np.linspace(0, 1, s.size)
            x_new = np.linspace(0, 1, 1001)
            s = np.interp(x_new, x_old, np.nan_to_num(s, nan=np.nanmean(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0))
        curves.append((subject_label, side, var, s))
    return curves

# -------------------- Main processing --------------------
def process_xlsx(xlsx_path, out_root, label_prefix=None):
    ensure_dir(out_root)
    # Διάβασε ΟΛΑ τα sheets
    all_sheets = pd.read_excel(xlsx_path, sheet_name=None, engine='openpyxl')
    results = []

    for sheet_name, df in all_sheets.items():
        # subject id από όνομα καρτέλας (π.χ. Sub01 -> 1), αλλιώς φύλα το raw
        m = re.search(r'(\d+)', str(sheet_name))
        if m:
            subj_id = int(m.group(1))
            subj_label = f"sub{subj_id:03d}"
        else:
            subj_label = f"sub_{re.sub(r'[^A-Za-z0-9]+','', sheet_name)}"
        if label_prefix:
            subj_label = f"{label_prefix}_{subj_label}"

        outdir = os.path.join(out_root, subj_label)
        ensure_dir(outdir)

        curves = extract_curves_from_sheet(df, subj_label)
        if not curves:
            # αν δεν βρέθηκαν καμπύλες, γράψε debug
            with open(os.path.join(outdir, f"{subj_label}_DEBUG.txt"), "w") as f:
                f.write("Columns:\n" + "\n".join(map(str, df.columns)))
            continue

        # Για κάθε καμπύλη: σώσε CSV, PH, PNG, summary row
        for (_sub, side, var, s) in curves:
            var_id = re.sub(r'[^A-Za-z0-9_]+','', var)
            side_id = re.sub(r'[^A-Za-z0-9]+','', side)
            base = os.path.join(outdir, f"{subj_label}_{side_id}_{var_id}")
            # CSV
            save_curve_long(s, _sub, side, var, base + ".csv")
            # PH
            try:
                X = takens_embedding(s, m=8, tau=5)
                dgm1 = ripser_h1_diagram(X)
                save_pd_and_barcode(dgm1, base, title=f"{subj_label} {side} {var}")
                tot_pers = float(np.nansum(np.clip(dgm1[:,1]-dgm1[:,0], 0, None))) if dgm1.size else 0.0
                results.append({
                    'subject': _sub,
                    'side': side,
                    'variable': var,
                    'h1_points': int(dgm1.shape[0]),
                    'h1_total_persistence': tot_pers
                })
            except Exception as e:
                results.append({
                    'subject': _sub,
                    'side': side,
                    'variable': var,
                    'error': str(e)
                })

        # per-subject summary
        subj_rows = [r for r in results if r['subject'] == subj_label]
        if subj_rows:
            pd.DataFrame(subj_rows).to_csv(os.path.join(outdir, f"{subj_label}_summary.csv"), index=False)

    # merged summary
    if results:
        pd.DataFrame(results).to_csv(os.path.join(out_root, "all_subjects_summary.csv"), index=False)
    print(f"[DONE] Wrote output -> {os.path.abspath(out_root)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='Path στο Excel (xlsx)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου')
    ap.add_argument('--label_prefix', default=None, help='Πρόθεμα για labels (π.χ. post / healthy)')
    args = ap.parse_args()

    process_xlsx(args.xlsx, args.out, label_prefix=args.label_prefix)

if __name__ == '__main__':
    main()