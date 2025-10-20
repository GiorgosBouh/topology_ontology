#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topology.py
------------------------------------------
Φόρτωση .MAT (Able-bodied ή Post-Stroke),
εξαγωγή L/R ή P/N κυματομορφών,
και υπολογισμός persistent homology (barcodes & H1 diagrams).

Χρήση:
  python topology.py --mat /path/to/MAT_normalizedData_*.mat --subject 1 --out output
"""

import argparse, os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) Φόρτωση MAT (συμβατό και με v7.3 HDF5)
# ------------------------------------------------------------
def load_mat_any(path):
    try:
        import mat73
        print("[INFO] Loading MAT (v7.3) with mat73...")
        return mat73.loadmat(path)
    except Exception:
        from scipy.io import loadmat
        print("[INFO] Loading MAT (pre-v7.3) with scipy.io...")
        return loadmat(path, simplify_cells=True)

# ------------------------------------------------------------
# 2) Βοηθητικά
# ------------------------------------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_get(d, *keys):
    """
    Πλοήγηση σε nested dicts/lists με ελαστικότητα.
    - Αν συναντήσει λίστα, θα προσπαθήσει να πάρει το πρώτο dict στοιχείο
      ή στοιχείο που περιέχει το επόμενο κλειδί.
    """
    cur = d
    for k in keys:
        if cur is None:
            return None

        # Αν είναι λίστα
        if isinstance(cur, list):
            if isinstance(k, int) and 0 <= k < len(cur):
                cur = cur[k]
                continue
            found = None
            for item in cur:
                if isinstance(item, dict) and (isinstance(k, str) and k in item):
                    found = item
                    break
            if found is not None:
                cur = found
            else:
                cur = cur[0] if cur else None

        # Αν είναι dict
        if isinstance(cur, dict):
            if isinstance(k, str):
                if k in cur:
                    cur = cur[k]
                else:
                    return None
            else:
                return None
        else:
            if k is not None:
                return None
    return cur

def detect_root(data):
    """
    Επιστρέφει dict που έχει κλειδί 'Sub' (λίστα ή dict).
    """
    if isinstance(data, dict) and 'Data' in data and isinstance(data['Data'], dict) and 'Sub' in data['Data']:
        return data['Data']
    if isinstance(data, dict) and 'Sub' in data:
        return data
    if isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, dict) and 'Sub' in v:
                return v
    raise KeyError("Δεν βρέθηκε Data.Sub στη δομή του MAT")

def _normalize_sub_char(sub_char):
    if sub_char is None:
        return None
    if isinstance(sub_char, dict):
        return sub_char
    if isinstance(sub_char, list):
        for item in sub_char:
            if isinstance(item, dict):
                return item
        return None
    return None

def is_stroke_subject(sub_char):
    sc = _normalize_sub_char(sub_char)
    if not isinstance(sc, dict):
        return False
    keys = set(map(str, sc.keys()))
    return len(keys & {'TPS','LesionLeft','FAC','POMA','TIS'}) > 0

def get_lesion_left(sub_char):
    sc = _normalize_sub_char(sub_char)
    if isinstance(sc, dict) and 'LesionLeft' in sc:
        return sc['LesionLeft']
    return None

# ------------------------------------------------------------
# 3) Εντοπισμός side blocks (robust)
# ------------------------------------------------------------
def get_side_blocks(sub, stroke=False):
    """
    Επιστρέφει dict με δεδομένα ανά πλευρά.
    Υγιείς: 'L'/'R'
    Stroke: 'P'/'N' (paretic/non-paretic)

    Υποστηρίζει πολλά aliases:
    - Able-bodied: Lsegmented_Ldata / Rsegmented_Rdata  ή  Lsegm_Ldata / Rsegm_Rdata
    - Stroke (εναλλακτικές του αρχείου που είδες):
        PsideSegm_PsideData / NsideSegm_NsideData (κύρια)
      και επίσης:
        Psegmented_Pdata / Nsegmented_Ndata
        Psegm_Pdata      / Nsegm_Ndata
    """
    if not stroke:
        cand_maps = [
            ('L','Lsegmented_Ldata'), ('R','Rsegmented_Rdata'),
            ('L','Lsegm_Ldata'),      ('R','Rsegm_Rdata'),
            # μερικά datasets έχουν LsideSegm_*:
            ('L','LsideSegm_LsideData'), ('R','RsideSegm_RsideData'),
        ]
        out = {'L': None, 'R': None}
    else:
        cand_maps = [
            # νέα ονόματα που βρήκες
            ('P','PsideSegm_PsideData'), ('N','NsideSegm_NsideData'),
            # κλασικά aliases
            ('P','Psegmented_Pdata'),    ('N','Nsegmented_Ndata'),
            ('P','Psegm_Pdata'),         ('N','Nsegm_Ndata'),
            # επιπλέον ασφαλιστικές δικλείδες
            ('P','PsideSegm_Pdata'),     ('N','NsideSegm_Ndata'),
        ]
        out = {'P': None, 'N': None}

    for side, key in cand_maps:
        if out.get(side) is None and isinstance(sub, dict) and key in sub:
            out[side] = sub[key]
    return out

# ------------------------------------------------------------
# 4) Εξαγωγή κυματομορφών (robust paths)
# ------------------------------------------------------------
def extract_waveform(block, kind, comp=None):
    """
    Επιστρέφει dict {kind: array}
    Δοκιμάζει πολλά paths: flat, 'Kinematic data', 'Kinetic data', 'EMG data'
    και axis keys σε πεζά/κεφαλαία.
    """
    out = {}
    if block is None:
        return out

    # Paths που θα δοκιμάσουμε
    paths = [
        [kind],                       # flat
        ['Kinematic data', kind],
        ['Kinetic data',   kind],
        ['EMG data',       kind],
    ]

    node = None
    for p in paths:
        node = safe_get(block, *p)
        if node is not None:
            break
    if node is None:
        return out

    # 3D components (angles/GRF) ή EMG (array)
    if kind in ['HipAngles','KneeAngles','AnkleAngles','GroundReactionForce']:
        axis = comp or 'x'
        for key_try in [axis, axis.upper(), axis.lower()]:
            if isinstance(node, dict) and key_try in node:
                out[kind] = np.asarray(node[key_try])
                return out
        # Αν δεν υπάρχουν άξονες (μερικές περιπτώσεις)
        if isinstance(node, (np.ndarray, list)):
            out[kind] = np.asarray(node)
            return out
    else:
        # EMG arrays (π.χ. GASnorm)
        out[kind] = np.asarray(node)
        return out

    return out

def mean_curve(arr):
    if arr is None or np.size(arr) == 0:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.shape[0] == 1001:
        return np.nanmean(arr, axis=1)
    if arr.shape[1] == 1001:
        return np.nanmean(arr, axis=0)
    ax = 1 if arr.shape[1] > 1 else 0
    return np.nanmean(arr, axis=ax)

# ------------------------------------------------------------
# 5) Persistent homology (Ripser)
# ------------------------------------------------------------
def takens_embedding(sig, m=8, tau=5):
    sig = np.asarray(sig, dtype=float)
    N = len(sig)
    L = N - (m - 1) * tau
    if L <= 0:
        raise ValueError("Σύντομο σήμα για το (m,tau)")
    X = np.stack([sig[i:i+L] for i in range(0, m*tau, tau)], axis=1)
    return X

def ripser_h1_diagram(X):
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms']) > 1 else np.empty((0, 2))

def save_pd_and_barcode(dgm, out_prefix, title="H1"):
    import matplotlib.pyplot as plt
    from persim import plot_diagrams, plot_barcodes
    # Diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Barcode
    plt.figure()
    plot_barcodes([dgm], title=f"{title} Barcode")
    plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------
# 6) Εξαγωγή σε CSV και summary
# ------------------------------------------------------------
def save_curve(curve, subject_id, side, varname, out_csv):
    t = np.linspace(0, 100, len(curve))
    df = pd.DataFrame({
        'subject': subject_id,
        'side': side,
        'variable': varname,
        'gc_percent': t,
        'value': curve
    })
    df.to_csv(out_csv, index=False)

def process_subject(sub, subj_index, outdir, stroke=False, lesion_left=None):
    sides = get_side_blocks(sub, stroke)
    results = []

    targets = [
        ('HipAngles', 'x'),
        ('KneeAngles', 'x'),
        ('AnkleAngles', 'x'),
        ('GroundReactionForce', 'z'),
        ('GASnorm', None),
    ]

    for side, block in sides.items():
        if block is None:
            continue
        for kind, comp in targets:
            waves = extract_waveform(block, kind, comp)
            if not waves:
                continue
            arr = list(waves.values())[0]
            curve = mean_curve(arr)
            if curve is None:
                continue

            varname = f"{kind}_{comp or 'n'}"
            out_prefix = os.path.join(outdir, f"sub{subj_index:03d}_{side}_{varname}")

            # καμπύλη σε CSV
            save_curve(curve, subj_index, side, varname, out_prefix + ".csv")

            # persistent homology
            try:
                X = takens_embedding(curve, m=8, tau=5)
                dgm1 = ripser_h1_diagram(X)
                save_pd_and_barcode(dgm1, out_prefix, f"Sub{subj_index:03d} {side} {varname}")
                total_persistence = float(np.nansum(np.clip(dgm1[:, 1] - dgm1[:, 0], 0, None))) if dgm1.size else 0.0
                results.append({
                    'subject': subj_index,
                    'side': side,
                    'variable': varname,
                    'h1_points': int(dgm1.shape[0]),
                    'h1_total_persistence': total_persistence,
                    'lesion_left': int(lesion_left) if lesion_left is not None else None
                })
            except Exception as e:
                results.append({
                    'subject': subj_index,
                    'side': side,
                    'variable': varname,
                    'error': str(e),
                    'lesion_left': int(lesion_left) if lesion_left is not None else None
                })

    if results:
        pd.DataFrame(results).to_csv(os.path.join(outdir, f"sub{subj_index:03d}_summary.csv"), index=False)

# ------------------------------------------------------------
# 7) Κύρια συνάρτηση
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True, help='Διαδρομή του MAT αρχείου')
    ap.add_argument('--subject', type=int, default=None, help='Αριθμός συμμετέχοντα (1..N)')
    ap.add_argument('--out', default='output', help='Φάκελος εξόδου')
    args = ap.parse_args()

    ensure_dir(args.out)
    data_raw = load_mat_any(args.mat)
    root = detect_root(data_raw)
    subs = root['Sub'] if 'Sub' in root else root.get('Sub')

    def looks_like_single_subject(d):
        if not isinstance(d, dict):
            return False
        keys = set(map(str, d.keys()))
        sentinels = {
            'sub_char', 'meas_char', 'events',
            'Lsegmented_Ldata', 'Rsegmented_Rdata',
            'Psegmented_Pdata', 'Nsegmented_Ndata',
            'Lsegmented_Bdata', 'Rsegmented_Bdata',
            # νέα που βρήκες:
            'PsideSegm_PsideData','PsideSegm_NsideData','PsideSegm_BsideData',
            'NsideSegm_PsideData','NsideSegm_NsideData','NsideSegm_BsideData'
        }
        return len(keys & sentinels) > 0

    # Κανονικοποίηση σε "λίστα από (index, subject_dict)"
    if isinstance(subs, list):
        all_subjects = list(enumerate(subs, start=1))
    elif isinstance(subs, dict):
        numeric_keys = [k for k in subs.keys() if str(k).isdigit()]
        if numeric_keys:
            idxs = sorted(int(k) for k in numeric_keys)
            all_subjects = [(i, subs[str(i)]) for i in idxs]
        else:
            if looks_like_single_subject(subs):
                all_subjects = [(1, subs)]
            else:
                candidates = []
                for k, v in subs.items():
                    if looks_like_single_subject(v):
                        try:
                            i = int(k)
                        except Exception:
                            continue
                        candidates.append((i, v))
                if not candidates:
                    raise ValueError("Δεν μπορώ να ερμηνεύσω το 'Sub' ως λίστα/σύνολο συμμετεχόντων.")
                all_subjects = sorted(candidates, key=lambda x: x[0])
    else:
        raise TypeError("Άγνωστος τύπος για Sub")

    # Φιλτράρισμα για --subject
    if args.subject is not None:
        all_subjects = [pair for pair in all_subjects if pair[0] == args.subject]

    for idx, sub in all_subjects:
        sub_char_raw = safe_get(sub, 'sub_char')
        stroke = is_stroke_subject(sub_char_raw)
        lesion_left = get_lesion_left(sub_char_raw) if stroke else None

        outdir = os.path.join(args.out, f"sub{idx:03d}")
        ensure_dir(outdir)
        print(f"[INFO] Processing subject {idx} | stroke={stroke} | lesion_left={lesion_left}")
        process_subject(sub, idx, outdir, stroke=stroke, lesion_left=lesion_left)

    print("[DONE] Αποτελέσματα στον φάκελο:", os.path.abspath(args.out))

# ------------------------------------------------------------
if __name__ == '__main__':
    main()