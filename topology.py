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
# 1. Φόρτωση MAT (συμβατό και με v7.3 HDF5)
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
# 2. Βοηθητικά
# ------------------------------------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_get(d, *keys):
    cur = d
    for k in keys:
        if cur is None: return None
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def detect_root(data):
    """Βρίσκει πού είναι το Data.Sub μέσα στο struct"""
    if 'Data' in data and isinstance(data['Data'], dict) and 'Sub' in data['Data']:
        return data['Data']
    if 'Sub' in data:
        return data
    for k,v in data.items():
        if isinstance(v, dict) and 'Sub' in v:
            return v
    raise KeyError("Δεν βρέθηκε Data.Sub στη δομή του MAT")

def is_stroke_subject(sub_char):
    if sub_char is None: return False
    keys = set(sub_char.keys())
    return any(k in keys for k in ['TPS','LesionLeft','FAC','POMA','TIS'])

def get_side_blocks(sub, stroke=False):
    """Επιστρέφει dict με δεδομένα ανά πλευρά"""
    if not stroke:
        return {
            'L': safe_get(sub,'Lsegmented_Ldata'),
            'R': safe_get(sub,'Rsegmented_Rdata')
        }
    else:
        return {
            'P': safe_get(sub,'Psegmented_Pdata'),
            'N': safe_get(sub,'Nsegmented_Ndata')
        }

# ------------------------------------------------------------
# 3. Εξαγωγή κυματομορφών
# ------------------------------------------------------------
def extract_waveform(block, kind, comp=None):
    out = {}
    if block is None: return out
    # Κινηματικά (γωνίες)
    if kind in ['HipAngles','KneeAngles','AnkleAngles']:
        node = safe_get(block, kind)
        if node is not None and comp in node:
            out[kind] = np.asarray(node[comp])
        return out
    # Δυνάμεις εδάφους
    if kind == 'GroundReactionForce':
        node = safe_get(block, kind)
        if node is not None and comp in node:
            out[kind] = np.asarray(node[comp])
        return out
    # EMG
    if kind in ['GASnorm','RFnorm','VLnorm','BFnorm','STnorm','TAnorm','ERSnorm']:
        node = safe_get(block, kind)
        if node is not None:
            out[kind] = np.asarray(node)
        return out
    return out

def mean_curve(arr):
    if arr is None or arr.size==0: return None
    arr = np.asarray(arr)
    if arr.ndim==1: return arr
    if arr.shape[0]==1001: return np.nanmean(arr,axis=1)
    else: return np.nanmean(arr,axis=0)

# ------------------------------------------------------------
# 4. Persistent homology (Ripser)
# ------------------------------------------------------------
def takens_embedding(sig, m=8, tau=5):
    sig = np.asarray(sig).astype(float)
    N = len(sig)
    L = N - (m - 1)*tau
    if L <= 0: raise ValueError("Σύντομο σήμα για το (m,tau)")
    X = np.stack([sig[i:i+L] for i in range(0,m*tau,tau)],axis=1)
    return X

def ripser_h1_diagram(X):
    from ripser import ripser
    res = ripser(X, maxdim=1)
    return res['dgms'][1] if len(res['dgms'])>1 else np.empty((0,2))

def save_pd_and_barcode(dgm, out_prefix, title="H1"):
    import matplotlib.pyplot as plt
    from persim import plot_diagrams, plot_barcodes
    # Diagram
    plt.figure()
    plot_diagrams([dgm], show=False, title=f"{title} Persistence Diagram")
    plt.savefig(out_prefix+"_H1diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Barcode
    plt.figure()
    plot_barcodes([dgm], title=f"{title} Barcode")
    plt.savefig(out_prefix+"_H1barcode.png", dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------
# 5. Εξαγωγή σε CSV και summary
# ------------------------------------------------------------
def save_curve(curve, subject_id, side, varname, out_csv):
    t = np.linspace(0,100,len(curve))
    df = pd.DataFrame({'subject':subject_id,'side':side,'variable':varname,'gc_percent':t,'value':curve})
    df.to_csv(out_csv,index=False)

def process_subject(sub, subj_index, outdir, stroke=False, lesion_left=None):
    sides = get_side_blocks(sub, stroke)
    results = []
    targets = [
        ('HipAngles','x'),
        ('KneeAngles','x'),
        ('AnkleAngles','x'),
        ('GroundReactionForce','z'),
        ('GASnorm',None)
    ]
    for side, block in sides.items():
        if block is None: continue
        for kind, comp in targets:
            waves = extract_waveform(block, kind, comp)
            if not waves: continue
            arr = list(waves.values())[0]
            curve = mean_curve(arr)
            if curve is None: continue
            varname = f"{kind}_{comp or 'n'}"
            out_prefix = os.path.join(outdir,f"sub{subj_index:03d}_{side}_{varname}")
            save_curve(curve, subj_index, side, varname, out_prefix+".csv")
            # persistent homology
            try:
                X = takens_embedding(curve)
                dgm1 = ripser_h1_diagram(X)
                save_pd_and_barcode(dgm1, out_prefix, f"Sub{subj_index:03d} {side} {varname}")
                total_persistence = float(np.nansum(np.clip(dgm1[:,1]-dgm1[:,0],0,None))) if dgm1.size else 0.0
                results.append({
                    'subject':subj_index,'side':side,'variable':varname,
                    'h1_points':int(dgm1.shape[0]),
                    'h1_total_persistence':total_persistence,
                    'lesion_left':int(lesion_left) if lesion_left is not None else None
                })
            except Exception as e:
                results.append({'subject':subj_index,'side':side,'variable':varname,'error':str(e)})
    if results:
        pd.DataFrame(results).to_csv(os.path.join(outdir,f"sub{subj_index:03d}_summary.csv"),index=False)

# ------------------------------------------------------------
# 6. Κύρια συνάρτηση
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
    subs = root['Sub']
    if isinstance(subs, dict):
        ids = sorted([int(k) for k in subs.keys()])
        iterable = [(i, subs[str(i)]) for i in ids]
    else:
        iterable = list(enumerate(subs, start=1))

    if args.subject is not None:
        iterable = [x for x in iterable if x[0]==args.subject]

    for idx, sub in iterable:
        sub_char = safe_get(sub,'sub_char')
        stroke = is_stroke_subject(sub_char)
        lesion_left = safe_get(sub_char,'LesionLeft') if stroke else None
        outdir = os.path.join(args.out,f"sub{idx:03d}")
        ensure_dir(outdir)
        print(f"[INFO] Processing subject {idx} | stroke={stroke} | lesion_left={lesion_left}")
        process_subject(sub, idx, outdir, stroke=stroke, lesion_left=lesion_left)

    print("[DONE] Αποτελέσματα στον φάκελο:", os.path.abspath(args.out))

# ------------------------------------------------------------
if __name__ == '__main__':
    main()
