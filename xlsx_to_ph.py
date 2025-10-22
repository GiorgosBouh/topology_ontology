#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_ph.py (robust)
------------------------------------------
Δουλεύει είτε τα σήματα είναι σε ΓΡΑΜΜΕΣ (1×1001) είτε σε ΣΤΗΛΕΣ (1001×1),
με headers/κενές/merged κελιές. Βγάζει CSV/PNG + per-subject summaries
και στο τέλος all_subjects_summary.csv.

Χρήση:
  python xlsx_to_ph.py --xlsx /path/to/Excel.xlsx --out output_dir --label_prefix post|healthy
"""

import argparse, os, re, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

# -------------------- I/O helpers --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def slug(s: str) -> str: return re.sub(r'[^A-Za-z0-9_]+', '_', str(s) if s is not None else '').strip('_')

def save_curve_csv(curve, subject_id, side, varname, out_csv):
    import pandas as pd, numpy as np
    t = np.linspace(0, 100, len(curve))
    df = pd.DataFrame({'subject':subject_id,'side':side,'variable':varname,'gc_percent':t,'value':curve})
    df.to_csv(out_csv, index=False)

# -------------------- name parsing --------------------
def detect_side_from_text(name: str):
    n = (name or "").lower()
    # explicit sides
    if re.search(r'\b(left|l)\b', n):  return 'L'
    if re.search(r'\b(right|r)\b', n): return 'R'
    if re.search(r'\b(paretic|p)\b', n): return 'P'
    if re.search(r'\b(non[-\s]?paretic|n)\b', n): return 'N'
    # substrings
    for k in [' left', '(left)', '_left', '_l', ' l)']: 
        if k in n: return 'L'
    for k in [' right','(right)','_right','_r',' r)']:
        if k in n: return 'R'
    return None

def detect_variable_from_text(name: str):
    n = (name or "").lower()
    if 'ground' in n and 'reaction' in n and ('z' in n or 'vertical' in n):
        return 'GroundReactionForce_z'
    if 'hip' in n and 'angle' in n:   return 'HipAngles_x'
    if 'knee' in n and 'angle' in n:  return 'KneeAngles_x'
    if 'ankle' in n and 'angle' in n: return 'AnkleAngles_x'
    if 'pelvis' in n and 'angle' in n: return 'PelvisAngles_x'
    for m in ['gas','rf','vl','bf','st','ta','ers']:
        if re.search(r'\b'+m+r'\b', n): return m.upper()+'norm'
    if 'angle' in n:
        if 'hip' in n:   return 'HipAngles_x'
        if 'knee' in n:  return 'KneeAngles_x'
        if 'ankle' in n: return 'AnkleAngles_x'
    return None

# -------------------- numeric scans --------------------
def tofloat(x):
    try: return float(x)
    except: return np.nan

def longest_run(arr):
    isn = np.isfinite(arr); run=best=0
    for v in isn:
        if v: run += 1; best = max(best, run)
        else: run = 0
    return best

def extract_longest_block(arr, min_len=900, target_len=1001):
    """Δώσε array (1D), πάρε το μεγαλύτερο συνεχόμενο μπλοκ finite αριθμών."""
    arr = np.asarray(arr, dtype=float).ravel()
    isn = np.isfinite(arr)
    best = (0, -1, -1)  # length, start, end
    i = 0
    n = len(arr)
    while i < n:
        if not isn[i]:
            i += 1; continue
        j = i
        while j < n and isn[j]: j += 1
        L = j - i
        if L > best[0]: best = (L, i, j)
        i = j
    L, a, b = best
    if L < min_len: return None
    block = arr[a:b]
    if len(block) >= target_len:
        return block[:target_len]
    if len(block) < target_len:
        pad = np.full(target_len, np.nan); pad[:len(block)] = block
        return pad
    return block

# -------------------- PH --------------------
def takens_embedding(sig, m=8, tau=5):
    sig = np.asarray(sig, dtype=float).ravel()
    N = len(sig); L = N - (m-1)*tau
    if L <= 0: return None
    return np.stack([sig[i:i+L] for i in range(0, m*tau, tau)], axis=1)

def ripser_h1(X):
    if X is None: return np.empty((0,2))
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms'])>1 else np.empty((0,2))

def save_pd_and_barcode(dgm, out_prefix, title="H1"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from persim import plot_diagrams
    # diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    # barcode
    try:
        from persim import plot_barcodes
        plt.figure()
        plot_barcodes([dgm], title=f"{title} Barcode")
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        plt.figure()
        y = 0
        for b,d in np.asarray(dgm):
            if np.isfinite(b) and np.isfinite(d) and d>b:
                plt.hlines(y, b, d); y += 1
        plt.xlabel("filtration"); plt.ylabel("bars")
        plt.title(f"{title} Barcode (custom)")
        plt.tight_layout()
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()

# -------------------- parsing in BOTH orientations --------------------
def parse_sheet(df_sheet):
    """
    Επιστρέφει entries: {raw_name, side, variable, curve}
    Πρώτα δοκιμάζει ΟΡΙΖΟΝΤΙΑ (row-wise). Αν λίγα hits, δοκιμάζει ΚΑΘΕΤΑ (col-wise).
    """
    hits = []

    # ---- 1) Row-wise (οριζόντια σήματα) ----
    df = df_sheet.copy()
    # κρατάμε raw ώστε να διαβάσουμε ό,τι υπάρχει
    # Προσπάθησε να βρεις name στη στήλη 0, αλλιώς χρησιμοποίησε την ίδια την τιμή
    if df.shape[1] == 0 or df.shape[0] == 0:
        return hits

    first_col = df.columns[0]
    for _, r in df.iterrows():
        name_cell = r[first_col] if first_col in df.columns else None
        name_s = str(name_cell) if name_cell is not None else ''
        # πάρε τη μεγαλύτερη αριθμητική ακολουθία στην γραμμή
        vals = [tofloat(x) for x in r.values]
        block = extract_longest_block(vals, min_len=900, target_len=1001)
        if block is None:
            continue
        side = detect_side_from_text(name_s)
        var  = detect_variable_from_text(name_s)
        hits.append({'raw_name': name_s, 'side': side, 'variable': var, 'curve': block})

    # Αν βρήκαμε αρκετά, τελειώσαμε
    if len(hits) >= 5:
        return hits

    # ---- 2) Column-wise (κάθετα σήματα) ----
    hits = []
    for col in df_sheet.columns:
        series = df_sheet[col]
        # όνομα από την 1η μη-κενή τιμή πάνω-πάνω (ή το header αν string)
        header_name = None
        # προτίμησε header (αν είναι string και όχι NaN)
        if isinstance(col, str):
            header_name = col
        # αλλιώς σκάναρε τα 5 πρώτα κελιά να βρεις text
        if not header_name:
            for i in range(min(5, len(series))):
                v = series.iloc[i]
                if isinstance(v, str) and v.strip():
                    header_name = v
                    break
        name_s = str(header_name) if header_name else f"col_{col}"

        vals = [tofloat(x) for x in series.values]
        block = extract_longest_block(vals, min_len=900, target_len=1001)
        if block is None:
            continue
        side = detect_side_from_text(name_s)
        var  = detect_variable_from_text(name_s)
        hits.append({'raw_name': name_s, 'side': side, 'variable': var, 'curve': block})

    return hits

# -------------------- per-sheet processing --------------------
def process_sheet(subject_name, df_sheet, out_dir, label_prefix):
    ensure_dir(out_dir)
    parsed = parse_sheet(df_sheet)

    if not parsed:
        dbg = os.path.join(out_dir, f"{label_prefix}_{slug(subject_name)}_DEBUG.txt")
        with open(dbg, "w") as f:
            f.write("No signals parsed in either orientation.\n")
            f.write("Columns:\n")
            f.write("\n".join(map(str, df_sheet.columns)) + "\n\n")
            try: f.write(df_sheet.head(12).to_string())
            except: pass
        return 0

    # συμπλήρωση πλευράς αν λείπει (L,R με σειρά)
    by_var = {}
    for it in parsed:
        v = it['variable'] or slug(it['raw_name'])
        by_var.setdefault(v, []).append(it)
    for v, lst in by_var.items():
        if all(x['side'] in ['L','R','P','N'] for x in lst):
            continue
        seq = ['L','R']
        i = 0
        for x in lst:
            if x['side'] not in ['L','R','P','N']: x['side'] = None
            if x['side'] is None:
                x['side'] = seq[i % len(seq)]; i += 1

    # γράψε outputs
    summary_rows = []
    subj_id = None
    m = re.search(r'(\d+)', str(subject_name))
    if m: subj_id = int(m.group(1))
    else: subj_id = subject_name

    for it in parsed:
        side = it['side'] or 'U'
        var  = it['variable'] or slug(it['raw_name'])
        curve = it['curve']
        out_prefix = os.path.join(out_dir, f"{label_prefix}_sub{subj_id:03d}_{side}_{var}")
        save_curve_csv(curve, subj_id, side, var, out_prefix + ".csv")

        X = takens_embedding(curve, m=8, tau=5)
        dgm1 = ripser_h1(X)
        save_pd_and_barcode(dgm1, out_prefix, f"{label_prefix} Sub{subj_id:03d} {side} {var}")
        tp = float(np.nansum(np.clip(dgm1[:,1]-dgm1[:,0], 0, None))) if dgm1.size else 0.0

        summary_rows.append({'subject':subj_id,'side':side,'variable':var,
                             'h1_points':int(dgm1.shape[0]),'h1_total_persistence':tp})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, f"{label_prefix}_sub{subj_id:03d}_summary.csv"), index=False
        )
    return len(summary_rows)

# -------------------- driver --------------------
def process_xlsx(xlsx_path, out_root, label_prefix='post'):
    import openpyxl
    ensure_dir(out_root)
    xls = pd.ExcelFile(xlsx_path, engine='openpyxl')
    sheets = [s for s in xls.sheet_names if str(s).strip()]
    merged = []

    for sname in sheets:
        if not str(sname).lower().startswith('sub'):
            continue
        # πάρε raw χωρίς header για να μη χαθούν τιμές
        df_raw = pd.read_excel(xls, sheet_name=sname, header=None)
        out_dir = os.path.join(out_root, f"{label_prefix}_{slug(sname)}")
        n = process_sheet(str(sname), df_raw, out_dir, label_prefix=label_prefix)
        summ = os.path.join(out_dir, f"{label_prefix}_{slug(sname)}_summary.csv")
        if os.path.exists(summ):
            merged.append(pd.read_csv(summ))

    if merged:
        all_df = pd.concat(merged, ignore_index=True)
        all_df.to_csv(os.path.join(out_root, "all_subjects_summary.csv"), index=False)
    print(f"[DONE] Wrote output -> {out_root}")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='Excel αρχείο (.xlsx)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου')
    ap.add_argument('--label_prefix', default='post', help='post | healthy (prefix στα αρχεία)')
    args = ap.parse_args()
    process_xlsx(args.xlsx, args.out, label_prefix=args.label_prefix)

if __name__ == '__main__':
    main()