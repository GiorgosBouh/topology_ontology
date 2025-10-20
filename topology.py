#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topology.py
------------------------------------------
Φόρτωση .MAT (Able-bodied ή Post-Stroke),
εξαγωγή L/R ή P/N κυματομορφών,
φιλτράρισμα GRF με GoodLanding ή threshold,
και υπολογισμός persistent homology (barcodes & H1 diagrams).

Χρήση:
  python topology.py --mat /path/to/MAT_normalizedData_*.mat --subject 1 --out output
"""

import argparse, os, math, warnings
import numpy as np
import pandas as pd

# Καταστολή άσχετων warnings (π.χ. Mean of empty slice)
warnings.filterwarnings("ignore", message="Mean of empty slice")


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
    - Αν συναντήσει λίστα, προσπαθεί να βρει στοιχείο-dict που έχει το επόμενο κλειδί,
      αλλιώς παίρνει το πρώτο στοιχείο.
    """
    cur = d
    for k in keys:
        if cur is None:
            return None

        if isinstance(cur, list):
            if isinstance(k, int) and 0 <= k < len(cur):
                cur = cur[k]; continue
            found = None
            for it in cur:
                if isinstance(it, dict) and isinstance(k, str) and k in it:
                    found = it; break
            cur = found if found is not None else (cur[0] if cur else None)

        if isinstance(cur, dict):
            if isinstance(k, str):
                if k in cur: cur = cur[k]
                else: return None
            else:
                return None
        else:
            if k is not None:
                return None
    return cur

def detect_root(data):
    """Εντόπισε αντικείμενο που έχει κλειδί 'Sub'."""
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
    if sub_char is None: return None
    if isinstance(sub_char, dict): return sub_char
    if isinstance(sub_char, list):
        for it in sub_char:
            if isinstance(it, dict): return it
    return None

def is_stroke_subject(sub_char):
    sc = _normalize_sub_char(sub_char)
    if not isinstance(sc, dict): return False
    keys = set(map(str, sc.keys()))
    return len(keys & {'TPS','LesionLeft','FAC','POMA','TIS'}) > 0

def get_lesion_left(sub_char):
    sc = _normalize_sub_char(sub_char)
    if isinstance(sc, dict) and 'LesionLeft' in sc:
        return sc['LesionLeft']
    return None


# ------------------------------------------------------------
# 3) Side blocks (με aliases & “δύο καρτέλες” P/N)
# ------------------------------------------------------------
def get_side_blocks(sub, stroke=False):
    """
    Υγιείς: 'L'/'R'
    Stroke: 'P'/'N'
    Επιλογή των σωστών tabs:
      - P από PsideSegm_PsideData
      - N από NsideSegm_NsideData
    Αν είναι λίστα, παίρνουμε το πρώτο dict στοιχείο.
    """
    def first_dict(x):
        if isinstance(x, list):
            for it in x:
                if isinstance(it, dict):
                    return it
            return x[0] if x else None
        return x

    if not stroke:
        out = {
            'L': first_dict(sub.get('Lsegmented_Ldata') or sub.get('Lsegm_Ldata') or sub.get('LsideSegm_LsideData')),
            'R': first_dict(sub.get('Rsegmented_Rdata') or sub.get('Rsegm_Rdata') or sub.get('RsideSegm_RsideData')),
        }
        return out
    else:
        Pblk = first_dict(sub.get('PsideSegm_PsideData')) or first_dict(sub.get('Psegmented_Pdata') or sub.get('Psegm_Pdata'))
        Nblk = first_dict(sub.get('NsideSegm_NsideData')) or first_dict(sub.get('Nsegmented_Ndata') or sub.get('Nsegm_Ndata'))
        return {'P': Pblk, 'N': Nblk}


# ------------------------------------------------------------
# 4) Μετατροπή οποιασδήποτε δομής σε 2D numeric πίνακα
# ------------------------------------------------------------
def to_numeric_matrix(x):
    """
    Μετατρέπει:
      - λίστα από (1001,) διανύσματα -> (n_strides, 1001)
      - numpy array -> όπως είναι (1D -> (1,L))
      - dict αξόνων -> None (χειρίζεται αλλού)
    """
    if x is None: return None
    if isinstance(x, dict): return None

    if isinstance(x, np.ndarray):
        if x.size == 0 or x.ndim == 0: return None
        if x.ndim == 1: return x.reshape(1, -1)
        return x

    if isinstance(x, list):
        rows = []
        for it in x:
            if it is None: continue
            arr = np.asarray(it)
            if arr.ndim == 0: continue
            if arr.ndim > 1: arr = arr.reshape(-1)
            rows.append(arr)
        if not rows: return None
        try:
            return np.stack(rows, axis=0)
        except Exception:
            L = min(len(r) for r in rows)
            return np.stack([r[:L] for r in rows], axis=0)

    return None


# ------------------------------------------------------------
# 5) Εξαγωγή κυματομορφών (flat / Kinematic / Kinetic / EMG)
# ------------------------------------------------------------
def extract_waveform(block, kind, comp=None):
    """
    Επιστρέφει dict {kind: array2d}
    Δοκιμάζει paths: flat / 'Kinematic data' / 'Kinetic data' / 'EMG data'
    και άξονες x/X/y/Y/z/Z όπου χρειάζεται.
    """
    if block is None:
        return {}

    # Αν block είναι λίστα, πάρε το πρώτο dict
    if isinstance(block, list):
        block = next((it for it in block if isinstance(it, dict)), block[0] if block else None)
        if block is None: return {}

    paths = [
        [kind],
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
        return {}

    if isinstance(node, list):
        node = next((it for it in node if isinstance(it, dict)), node[0] if node else None)
        if node is None:
            return {}

    # Angles & GRF (3D) ή EMG (array)
    if kind in ['HipAngles','KneeAngles','AnkleAngles','GroundReactionForce']:
        axis = comp or 'x'
        if isinstance(node, dict):
            for key_try in [axis, axis.upper(), axis.lower()]:
                if key_try in node:
                    arr = to_numeric_matrix(node[key_try])
                    if arr is not None:
                        return {kind: arr}
        arr = to_numeric_matrix(node)
        return {kind: arr} if arr is not None else {}
    else:
        arr = to_numeric_matrix(node)
        return {kind: arr} if arr is not None else {}


# ------------------------------------------------------------
# 6) Stride φίλτρα & μέση καμπύλη
# ------------------------------------------------------------
def drop_all_nan_strides(A):
    """Κράτα μόνο strides που έχουν τουλάχιστον ένα finite sample.
       Δουλεύει και για (1001, n_strides) και για (n_strides, 1001)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        return A, 0
    if A.shape[0] == 1001:
        valid = np.isfinite(A).any(axis=0)  # strides = columns
        return A[:, valid], int(valid.sum())
    elif A.shape[1] == 1001:
        valid = np.isfinite(A).any(axis=1)  # strides = rows
        return A[valid, :], int(valid.sum())
    else:
        if np.isfinite(A).any():
            return A, 1
        return A, 0

def valid_grf_mask_from_z(grf_z_2d, min_peak_bw=0.1, min_samples_over=5):
    """
    grf_z_2d: (1001, n_strides) ή (n_strides, 1001)
    Επιστρέφει μάσκα (n_strides,) με True όπου ο stride έχει
    τουλάχιστον 'min_samples_over' δείγματα > min_peak_bw (BW-normalized).
    """
    A = np.asarray(grf_z_2d, dtype=float)
    if A.ndim != 2:
        return None
    if A.shape[0] == 1001:
        over = A > min_peak_bw
        return (np.sum(over, axis=0) >= min_samples_over)
    elif A.shape[1] == 1001:
        over = A > min_peak_bw
        return (np.sum(over, axis=1) >= min_samples_over)
    return None

def mean_curve(arr2d):
    """
    Επιστρέφει 1D καμπύλη ή None αν δεν υπάρχουν έγκυρα (finite) δείγματα.
    Δέχεται (n_strides, 1001) ή (1001, n_strides) ή (1,1001) ή 1D.
    """
    if arr2d is None:
        return None
    A = np.asarray(arr2d, dtype=float)
    if A.ndim == 0:
        return None
    if not np.isfinite(A).any():
        return None
    if A.ndim == 1:
        return A

    if A.shape[0] == 1001:
        row_mask = np.isfinite(A).any(axis=1)
        if not row_mask.any():
            return None
        return np.nanmean(A, axis=1)
    if A.shape[1] == 1001:
        col_mask = np.isfinite(A).any(axis=0)
        if not col_mask.any():
            return None
        return np.nanmean(A, axis=0)

    ax = 1 if (A.ndim >= 2 and A.shape[1] > 1) else 0
    return np.nanmean(A, axis=ax)


# ------------------------------------------------------------
# 7) Persistence homology (Ripser) & plots
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
    # --- Persistence Diagram (πάντα διαθέσιμο) ---
    from persim import plot_diagrams
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix + "_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- Barcode: δοκίμασε persim.plot_barcodes, αλλιώς custom ---
    try:
        from persim import plot_barcodes
        plt.figure()
        plot_barcodes([dgm], title=f"{title} Barcode")
        plt.savefig(out_prefix + "_H1barcode.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        # custom barcode renderer
        import numpy as np
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


# ------------------------------------------------------------
# 8) GoodLanding από events (αν υπάρχει)
# ------------------------------------------------------------
def get_goodlanding_mask(sub, side_key):
    """
    Προσπαθεί να βρει μάσκα GoodLanding για το αντίστοιχο tab.
    side_key: 'P' ή 'N' (post-stroke) ή 'L'/'R' (healthy)
    Επιστρέφει np.array shape (n_strides,) από True/False ή None αν δεν βρεθεί.
    """
    ev = sub.get('events')
    if not isinstance(ev, dict):
        return None

    candidates = []
    if side_key == 'P':
        candidates = ['PsideSegm_GoodLanding', 'P_GoodLanding', 'GoodLanding_P']
    elif side_key == 'N':
        candidates = ['NsideSegm_GoodLanding', 'N_GoodLanding', 'GoodLanding_N']
    elif side_key == 'L':
        candidates = ['L_GoodLanding', 'GoodLanding_L']
    elif side_key == 'R':
        candidates = ['R_GoodLanding', 'GoodLanding_R']

    for ck in candidates:
        gl = ev.get(ck)
        if gl is not None:
            arr = np.asarray(gl).astype(float).reshape(-1)
            return arr == 1
    return None


# ------------------------------------------------------------
# 9) Εξαγωγή σε CSV και summary
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

        any_written = False

        for kind, comp in targets:
            waves = extract_waveform(block, kind, comp)
            if not waves:
                continue
            arr2d = list(waves.values())[0]

            # --- ειδικός χειρισμός για GRF: φίλτραρισμα ανά stride ---
            if kind == 'GroundReactionForce':
                # 1) Αν υπάρχει GoodLanding για το tab, φιλτράρισε
                gl_mask = get_goodlanding_mask(sub, side)
                if gl_mask is not None:
                    A = np.asarray(arr2d, dtype=float)
                    if A.ndim == 2:
                        if A.shape[0] == 1001 and gl_mask.size == A.shape[1]:
                            arr2d = A[:, gl_mask]
                        elif A.shape[1] == 1001 and gl_mask.size == A.shape[0]:
                            arr2d = A[gl_mask, :]
                else:
                    # 2) αλλιώς, threshold στον GRF_z
                    mask = valid_grf_mask_from_z(arr2d, min_peak_bw=0.1, min_samples_over=5)
                    if mask is not None:
                        A = np.asarray(arr2d, dtype=float)
                        if A.ndim == 2:
                            if A.shape[0] == 1001 and mask.size == A.shape[1]:
                                arr2d = A[:, mask]
                            elif A.shape[1] == 1001 and mask.size == A.shape[0]:
                                arr2d = A[mask, :]

            # ➊ Πέτα strides που είναι όλοι-NaN
            arr2d_filt, n_valid = drop_all_nan_strides(arr2d)
            if n_valid == 0 or not np.isfinite(arr2d_filt).any():
                results.append({
                    'subject': subj_index,
                    'side': side,
                    'variable': f"{kind}_{comp or 'n'}",
                    'error': 'empty_or_all_nan',
                    'lesion_left': int(lesion_left) if lesion_left is not None else None
                })
                continue

            # ➋ μέση καμπύλη
            curve = mean_curve(arr2d_filt)
            if curve is None or (isinstance(curve, np.ndarray) and not np.isfinite(curve).any()):
                results.append({
                    'subject': subj_index,
                    'side': side,
                    'variable': f"{kind}_{comp or 'n'}",
                    'error': 'empty_or_all_nan',
                    'lesion_left': int(lesion_left) if lesion_left is not None else None
                })
                continue

            varname = f"{kind}_{comp or 'n'}"
            out_prefix = os.path.join(outdir, f"sub{subj_index:03d}_{side}_{varname}")

            # CSV
            save_curve(curve, subj_index, side, varname, out_prefix + ".csv")
            any_written = True

            # PH
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

        # Debug αν δεν γράψαμε τίποτα
        if not any_written:
            with open(os.path.join(outdir, f"sub{subj_index:03d}_{side}_DEBUG.txt"), "w") as f:
                if isinstance(block, dict):
                    f.write("BLOCK KEYS:\n" + "\n".join(map(str, block.keys())) + "\n")
                elif isinstance(block, list):
                    f.write(f"BLOCK is list with {len(block)} items.\n")
                    for it in block:
                        if isinstance(it, dict):
                            f.write("First dict item keys:\n" + "\n".join(map(str, it.keys())) + "\n")
                            break
                else:
                    f.write(f"BLOCK TYPE: {type(block).__name__}\n")

    if results:
        pd.DataFrame(results).to_csv(os.path.join(outdir, f"sub{subj_index:03d}_summary.csv"), index=False)


# ------------------------------------------------------------
# 10) Κύρια συνάρτηση
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True, help='Διαδρομή του MAT αρχείου')
    ap.add_argument('--subject', type=int, default=None, help='Αριθμός συμμετέχοντα (1..N)')
    ap.add_argument('--out', default='output', help='Φάκελος εξόδου')
    args = ap.parse_args()

    ensure_dir(args.out)
    data_raw = load_mat_any(args.mat)

    # Βρες 'Sub'
    if isinstance(data_raw, dict) and 'Sub' in data_raw:
        root = data_raw
    else:
        root = detect_root(data_raw)
    subs = root['Sub'] if 'Sub' in root else root.get('Sub')

    def looks_like_single_subject(d):
        if not isinstance(d, dict): return False
        keys = set(map(str, d.keys()))
        sentinels = {
            'sub_char','meas_char','events',
            'Lsegmented_Ldata','Rsegmented_Rdata',
            'Psegmented_Pdata','Nsegmented_Ndata',
            'Lsegmented_Bdata','Rsegmented_Bdata',
            'PsideSegm_PsideData','PsideSegm_NsideData','PsideSegm_BsideData',
            'NsideSegm_PsideData','NsideSegm_NsideData','NsideSegm_BsideData'
        }
        return len(keys & sentinels) > 0

    # Κανονικοποίηση σε λίστα (idx, subject_dict)
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
                        try: i = int(k)
                        except Exception: continue
                        candidates.append((i, v))
                if not candidates:
                    raise ValueError("Δεν μπορώ να ερμηνεύσω το 'Sub' ως λίστα/σύνολο συμμετεχόντων.")
                all_subjects = sorted(candidates, key=lambda x: x[0])
    else:
        raise TypeError("Άγνωστος τύπος για Sub")

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