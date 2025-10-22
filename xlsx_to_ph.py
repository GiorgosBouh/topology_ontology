#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_ph.py
------------------------------------------
Διαβάζει τα Excel:
  - MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx
  - MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx

Parser robust:
  • Σκανάρει ΚΑΘΕ SHEET (ένας συμμετέχων)
  • Δουλεύει σε wide-rows μορφή: 1η στήλη = όνομα μεταβλητής, επόμενες ~1001 στήλες = τιμές
  • Αναγνωρίζει side (L/R ή Left/Right ή P/N για stroke)
  • Εξάγει καμπύλες (1001 samples), αποθηκεύει CSV, H1 diagram & barcode
  • Γράφει per-subject summary + merged all_subjects_summary.csv στο τέλος

Χρήση:
  python xlsx_to_ph.py --xlsx /path/to/Excel.xlsx --out output_dir --label_prefix post|healthy
"""

import argparse, os, re, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

# -------------------- utils --------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', s or '').strip('_')

def detect_side_from_text(name: str):
    """Προσπαθεί να εξάγει πλευρά από το όνομα."""
    n = (name or "").lower()
    if any(k in n for k in [' left', '(left)', '_left', ' l ', '_l', ' l)', ' left_', 'left-']):
        return 'L'
    if any(k in n for k in [' right', '(right)', '_right', ' r ', '_r', ' r)', ' right_', 'right-']):
        return 'R'
    if any(k in n for k in ['_p', ' paretic', '(p)', ' p ']):
        return 'P'
    if any(k in n for k in ['_n', ' non-paretic', '(n)', ' n ']):
        return 'N'
    # fallback: δεν δηλώνεται — επέστρεψε None, θα βάλουμε L/R ανά διπλές εμφανίσεις
    return None

def detect_variable_from_text(name: str):
    """
    Εντοπίζει τύπο μεταβλητής από το κείμενο.
    Χαλαρό mapping για Hip/Knee/Ankle Angles, GRF, EMG (GAS/RF/VL/BF/ST/TA/ERS).
    """
    n = (name or "").lower()
    # βασικά joint angles
    if 'hip'   in n and 'angle' in n:   return 'HipAngles_x'
    if 'knee'  in n and 'angle' in n:   return 'KneeAngles_x'
    if 'ankle' in n and 'angle' in n:   return 'AnkleAngles_x'
    # pelvis angle optional
    if 'pelvis' in n and 'angle' in n:  return 'PelvisAngles_x'

    # GRF
    if ('grf' in n or 'ground reaction force' in n) and ('z' in n or 'vertical' in n):
        return 'GroundReactionForce_z'

    # EMG muscles
    for m in ['gas','rf','vl','bf','st','ta','ers']:
        if re.search(r'\b'+m+r'\b', n):
            return m.upper()+'norm'  # normalized channels

    # fallback: δοκίμασε καθαρή λέξη "angle" και hip/knee/ankle αλλιώς
    if 'angle' in n:
        if 'hip' in n: return 'HipAngles_x'
        if 'knee' in n: return 'KneeAngles_x'
        if 'ankle' in n: return 'AnkleAngles_x'

    return None

def takens_embedding(sig, m=8, tau=5):
    sig = np.asarray(sig, dtype=float).ravel()
    N = len(sig); L = N - (m-1)*tau
    if L <= 0: return None
    return np.stack([sig[i:i+L] for i in range(0, m*tau, tau)], axis=1)

def ripser_h1(X):
    if X is None:
        return np.empty((0,2))
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms'])>1 else np.empty((0,2))

def save_pd_and_barcode(dgm, out_prefix, title="H1"):
    import matplotlib.pyplot as plt
    from persim import plot_diagrams
    # diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    # barcode: persim<=0.4 ίσως δεν έχει plot_barcodes – fallback custom
    try:
        from persim import plot_barcodes
        plt.figure()
        plot_barcodes([dgm], title=f"{title} Barcode")
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
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

def save_curve_csv(curve, subject_id, side, varname, out_csv):
    import pandas as pd, numpy as np
    t = np.linspace(0, 100, len(curve))
    df = pd.DataFrame({'subject':subject_id,'side':side,'variable':varname,'gc_percent':t,'value':curve})
    df.to_csv(out_csv, index=False)

# -------------------- row-wise Excel parser --------------------

def row_to_numeric_values(row_values):
    """
    Παίρνει μία γραμμή (λίστα τιμών από pandas) και επιστρέφει το ΜΕΓΑΛΥΤΕΡΟ
    συνεχόμενο μπλοκ από αριθμούς (>=900 τιμές), ως numpy array.
    Αν δεν βρει, επιστρέφει None.
    """
    vals = list(row_values)
    # άσε το πρώτο κελί (συνήθως όνομα), ψάξε από 2ο και μετά
    arr = []
    for x in vals[1:]:
        try:
            fx = float(x)
            arr.append(fx)
        except Exception:
            arr.append(np.nan)
    arr = np.asarray(arr, dtype=float)

    # ψάξε το μεγαλύτερο συνεχόμενο διάστημα με >=900 finite samples
    isn = np.isfinite(arr)
    if not isn.any():
        return None
    # βρες όλα τα segments finite
    max_len = 0; best = None
    i = 0
    while i < len(arr):
        if not np.isfinite(arr[i]):
            i += 1; continue
        j = i
        while j < len(arr) and np.isfinite(arr[j]):
            j += 1
        seg = arr[i:j]
        if seg.size > max_len:
            max_len = seg.size
            best = seg
        i = j
    if best is None or best.size < 900:
        return None
    # ιδανικά θέλουμε 1001 — αν >1001 κόψε, αν <1001 κάνε padding (σπάνια)
    if best.size >= 1001:
        return best[:1001]
    # padding (γραμμικό), αν χρειαστεί
    pad = np.full(1001, np.nan, dtype=float)
    pad[:best.size] = best
    # (προαιρετικά interpolation, εδώ αφήνουμε NaN στο τέλος)
    return pad

def parse_sheet_rowwise(df_sheet):
    """
    Προσπάθεια parsing όταν το sheet είναι σε μορφή:
      [0] variable name | [1..] 1001 samples
    Επιστρέφει λίστα dicts: {'raw_name','side','variable','curve'}
    """
    rows = []
    # ρίξε εντελώς κενές στήλες
    df = df_sheet.copy()
    if df.empty:
        return rows

    # βρες rows με πιθανή μεταβλητή: η 1η στήλη string & >=900 αριθμητικές μετά
    first_col = df.columns[0]
    for idx, r in df.iterrows():
        name_cell = str(r[first_col]) if first_col in df.columns else ''
        if name_cell is None or name_cell.strip()=='' or name_cell.lower().startswith('nan'):
            continue
        # απόπειρα εξαγωγής πλευράς & variable
        side = detect_side_from_text(name_cell)
        var  = detect_variable_from_text(name_cell)

        vals = row_to_numeric_values(list(r.values))
        if vals is None:
            continue  # όχι “γραμμή-σήμα”

        # αν δεν βρήκα side από το κείμενο και η μεταβλητή είναι joint/GRF/EMG,
        # προσπάθησε να μαντέψεις: αν υπάρχει “Left/Right” σε διπλανές γραμμές
        rows.append({
            'raw_name': name_cell.strip(),
            'side': side,
            'variable': var,
            'curve': vals
        })
    return rows

# -------------------- main per-sheet → subject processing --------------------

def process_sheet(subject_name, df_sheet, out_dir, label_prefix):
    """
    Παίρνει ένα pandas sheet, κάνει parsing (row-wise),
    παράγει CSV/PNG και per-subject summary.
    """
    ensure_dir(out_dir)
    parsed = parse_sheet_rowwise(df_sheet)

    # Αν δεν βρήκαμε τίποτα, γράψε DEBUG για να δούμε columns & head
    if not parsed:
        dbg = os.path.join(out_dir, f"{label_prefix}_{subject_name}_DEBUG.txt")
        with open(dbg, "w") as f:
            f.write("No row-wise signals parsed.\n")
            f.write("Columns:\n")
            f.write("\n".join(map(str, df_sheet.columns)) + "\n\n")
            f.write("Head(10):\n")
            try:
                f.write(df_sheet.head(10).to_string())
            except Exception:
                pass
        return 0

    # Αν υπάρχουν μεταβλητές χωρίς side, προσπαθώ να τις ζευγαρώσω:
    # π.χ. αν δύο raw_names για “HipAngles …” στη σειρά, βάλε L/R με τη σειρά εμφάνισης.
    by_var = {}
    for it in parsed:
        var = it['variable'] or slug(it['raw_name'])
        by_var.setdefault(var, []).append(it)
    # συμπλήρωσε sides
    for var, lst in by_var.items():
        # αν όλα έχουν side -> OK
        if all(x['side'] in ['L','R','P','N'] for x in lst):
            continue
        # αλλιώς, δώσε L,R με τη σειρά εμφάνισης
        lr = ['L','R']
        pn = ['P','N']
        # heuristic: joint/GRF -> L/R, stroke EMG ίσως P/N, αλλά θα βάλουμε L/R για γενικότητα
        seq = lr
        i = 0
        for x in lst:
            if x['side'] not in ['L','R','P','N',None]:
                x['side'] = None
            if x['side'] is None:
                x['side'] = seq[i % len(seq)]
                i += 1

    # Γράψε outputs
    import matplotlib
    matplotlib.use('Agg')

    summary_rows = []
    subj_id_num = re.search(r'(\d+)', subject_name)
    subj_id = int(subj_id_num.group(1)) if subj_id_num else subject_name

    for it in parsed:
        side = it['side'] or 'U'  # Unknown
        var  = it['variable'] or slug(it['raw_name'])
        curve = it['curve']
        # σώσε CSV
        out_prefix = os.path.join(out_dir, f"{label_prefix}_sub{subj_id:03d}_{side}_{var}")
        save_curve_csv(curve, subj_id, side, var, out_prefix + ".csv")

        # PH
        X = takens_embedding(curve, m=8, tau=5)
        dgm1 = ripser_h1(X)
        save_pd_and_barcode(dgm1, out_prefix, f"{label_prefix} Sub{subj_id:03d} {side} {var}")
        total_persistence = float(np.nansum(np.clip(dgm1[:,1]-dgm1[:,0], 0, None))) if dgm1.size else 0.0

        summary_rows.append({
            'subject': subj_id,
            'side': side,
            'variable': var,
            'h1_points': int(dgm1.shape[0]),
            'h1_total_persistence': total_persistence
        })

    # per-subject summary
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(os.path.join(out_dir, f"{label_prefix}_sub{subj_id:03d}_summary.csv"), index=False)

    return len(summary_rows)

# -------------------- driver --------------------

def process_xlsx(xlsx_path, out_root, label_prefix='post'):
    import openpyxl
    ensure_dir(out_root)

    xls = pd.ExcelFile(xlsx_path, engine='openpyxl')
    sheets = [s for s in xls.sheet_names if str(s).strip()]
    merged = []

    for sname in sheets:
        # αγνόησε tabs που δεν είναι subjects (π.χ. “Read me”)
        if not re.search(r'\bsub\s*\d+\b', str(sname).lower()):
            # ωστόσο, πολλά φύλλα είναι "Sub01" ακριβώς—θα κρατήσουμε ό,τι ξεκινά με 'Sub'
            if not str(sname).lower().startswith('sub'):
                continue

        df_s = pd.read_excel(xls, sheet_name=sname, header=0)
        out_dir = os.path.join(out_root, f"{label_prefix}_{slug(sname)}")
        n = process_sheet(str(sname), df_s, out_dir, label_prefix=label_prefix)

        # μάζεψε per-subject summary αν υπάρχει
        summ = os.path.join(out_dir, f"{label_prefix}_{slug(sname)}_summary.csv")
        if os.path.exists(summ):
            m = pd.read_csv(summ)
            merged.append(m)

    # merged all_subjects_summary
    if merged:
        all_df = pd.concat(merged, ignore_index=True)
        all_df.to_csv(os.path.join(out_root, "all_subjects_summary.csv"), index=False)

    print(f"[DONE] Wrote output -> {out_root}")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='Excel αρχείο (.xlsx)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου')
    ap.add_argument('--label_prefix', default='post', help='Prefix: post | healthy (μπαίνει στα ονόματα αρχείων)')
    args = ap.parse_args()

    process_xlsx(args.xlsx, args.out, label_prefix=args.label_prefix)

if __name__ == '__main__':
    main()