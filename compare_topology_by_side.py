#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_topology_by_side.py
------------------------------------------
Side-aware σύγκριση τοπολογικών H1 metrics.

Τι κάνει
- Φορτώνει 2 summaries (post-stroke & healthy) από το xlsx_to_ph.py.
- Δεν ενοποιεί τα sides: κρατά P/N (post) και L/R (healthy).
- Για ΚΑΘΕ variable:
    * POST: P vs N (paired όπου υπάρχει ίδιο subject με P & N)
    * HEALTHY: L vs R (paired όπου υπάρχει ίδιο subject με L & R)
- Επιπλέον CROSS-GROUP ανά side:
    * post P vs healthy L, post P vs healthy R,
      post N vs healthy L, post N vs healthy R (unpaired tests).
- Metrics:
    - means/SD/N ανά side
    - paired t-test, Wilcoxon (για εντός-ομάδας)
    - Welch t-test, Mann–Whitney (για μεταξύ-ομάδων)
    - Effect sizes: Cohen’s dz (paired), Cohen’s d (unpaired), Cliff’s delta
- Plots: boxplot + violin για κάθε σύγκριση.
- Αποτέλεσμα: side_stats.csv (εντός-ομάδων) + cross_side_stats.csv (μεταξύ-ομάδων) + PNGs.

Χρήση:
  python compare_topology_by_side.py \
      --post    ~/topology/out_post_xlsx_col/all_subjects_summary.csv \
      --healthy ~/topology/out_healthy_xlsx_col/all_subjects_summary.csv \
      --out     ~/topology/compare_by_side
"""

import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------ helpers ------------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def normalize_variable_name(v):
    """Αφαιρεί pside_/nside_ prefixes ώστε να ταιριάξει με healthy ονόματα."""
    v = str(v).strip()
    v_low = v.lower()
    if v_low.startswith('pside_'):
        return v[len('pside_'):]
    if v_low.startswith('nside_'):
        return v[len('nside_'):]
    return v

def read_summary(path, group_label):
    df = pd.read_csv(path)
    df['group'] = group_label
    if 'h1_total_persistence' not in df.columns:
        raise ValueError(f"{path}: λείπει η στήλη h1_total_persistence")
    df['h1_total_persistence'] = pd.to_numeric(df['h1_total_persistence'], errors='coerce')
    df = df[df['h1_total_persistence'].notna()].copy()
    # normalize subject id (για pairing)
    df['subject_norm'] = df['subject'].astype(str).str.extract(r'(\d+)')
    df['subject_norm'] = df['subject_norm'].fillna(df['subject'].astype(str))
    df['subject_norm'] = df['subject_norm'].apply(
        lambda s: f"sub{int(s):03d}" if str(s).isdigit() else re.sub(r'[^A-Za-z0-9]+','', str(s))
    )
    # normalized variable (για cross-group match)
    df['variable_base'] = df['variable'].apply(normalize_variable_name)
    return df

def cohens_dz_paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    if d.size < 2: return np.nan
    sd = np.std(d, ddof=1)
    if sd == 0 or not np.isfinite(sd): return np.nan
    return float(np.mean(d) / sd)

def cohens_d_unpaired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    denom = (na + nb - 2)
    if denom <= 0: return np.nan
    s = np.sqrt(((na-1)*va + (nb-1)*vb) / denom)
    if s == 0 or not np.isfinite(s): return np.nan
    return float((np.mean(a) - np.mean(b)) / s)

def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    n_conc = 0; n_disc = 0
    for x in a:
        n_conc += np.sum(x > b)
        n_disc += np.sum(x < b)
    return float((n_conc - n_disc) / (len(a)*len(b)))

def simple_boxplot(two_arrays, labels, title, out_png):
    plt.figure()
    data = [np.asarray(a, float) for a in two_arrays]
    plt.boxplot(data, tick_labels=labels, showfliers=True)  # Matplotlib 3.9+: tick_labels
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def violin_plot(two_arrays, labels, title, out_png):
    plt.figure()
    data = [np.asarray(a, float) for a in two_arrays]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks([1,2], labels)
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def paired_arrays(df, side_a, side_b):
    piv = df.pivot_table(index='subject_norm', columns='side', values='h1_total_persistence', aggfunc='mean')
    if side_a not in piv.columns or side_b not in piv.columns:
        return np.array([]), np.array([])
    sub = piv.dropna(subset=[side_a, side_b])
    return sub[side_a].to_numpy(float), sub[side_b].to_numpy(float)

def side_stats_block(df, side_a, side_b, label_a, label_b, out_dir, var, group_tag):
    # περιγραφικά ανά side (ανεξάρτητα)
    def desc_for_side(side_code):
        vals = df.loc[df['side']==side_code, 'h1_total_persistence'].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        return float(np.nanmean(vals)) if vals.size else np.nan, \
               float(np.nanstd(vals, ddof=1)) if vals.size>1 else np.nan, \
               int(vals.size), vals

    meanA, sdA, nA, vals_a = desc_for_side(side_a)
    meanB, sdB, nB, vals_b = desc_for_side(side_b)

    # paired tests
    x, y = paired_arrays(df, side_a, side_b)
    if x.size >= 2:
        t_stat, p_t = stats.ttest_rel(x, y, alternative='two-sided')
        try:
            w_stat, p_w = stats.wilcoxon(x, y, alternative='two-sided', zero_method='wilcox')
        except Exception:
            w_stat, p_w = (np.nan, np.nan)
        dz = cohens_dz_paired(x, y)
        cd = cliffs_delta(x, y)  # σε μη-paired εκδοχή, εδώ είναι επί των ζευγών – αποδεκτό ως ένδειξη
    else:
        t_stat = p_t = w_stat = p_w = dz = cd = np.nan

    # plots
    safe_var = re.sub(r'[^A-Za-z0-9_]+','', str(var))
    base = f"{group_tag}_{safe_var}_{label_a}_vs_{label_b}"
    box_png    = os.path.join(out_dir, f"box_{base}.png")
    violin_png = os.path.join(out_dir, f"violin_{base}.png")
    simple_boxplot([vals_a, vals_b], [label_a, label_b], title=f"{group_tag} — {var}: {label_a} vs {label_b}", out_png=box_png)
    violin_plot([vals_a, vals_b], [label_a, label_b], title=f"{group_tag} — {var}: {label_a} vs {label_b}", out_png=violin_png)

    return {
        'group': group_tag,
        'variable': var,
        'sideA_label': label_a, 'sideB_label': label_b,
        'mean_A': meanA, 'sd_A': sdA, 'n_A': int(nA),
        'mean_B': meanB, 'sd_B': sdB, 'n_B': int(nB),
        'paired_N': int(x.size),
        'paired_t_stat': float(t_stat) if np.isfinite(t_stat) else np.nan,
        'paired_t_pvalue': float(p_t) if np.isfinite(p_t) else np.nan,
        'wilcoxon_stat': float(w_stat) if np.isfinite(w_stat) else np.nan,
        'wilcoxon_pvalue': float(p_w) if np.isfinite(p_w) else np.nan,
        'cohens_dz': float(dz) if np.isfinite(dz) else np.nan,
        'cliffs_delta_on_pairs': float(cd) if np.isfinite(cd) else np.nan,
        'boxplot_png': os.path.basename(box_png),
        'violin_png':  os.path.basename(violin_png),
    }

def cross_group_block(df_post, df_hlt, var_base, post_side, hlt_side, out_dir):
    """Μεταξύ-ομάδων (unpaired) για συγκεκριμένο variable_base & sides."""
    sub_post = df_post[(df_post['variable_base']==var_base) & (df_post['side']==post_side)]
    sub_hlt  = df_hlt [(df_hlt ['variable_base']==var_base) & (df_hlt ['side']==hlt_side)]
    a = sub_post['h1_total_persistence'].astype(float).to_numpy()
    b = sub_hlt ['h1_total_persistence'].astype(float).to_numpy()
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    meanA, sdA, nA = (np.nan, np.nan, 0) if a.size==0 else (float(np.nanmean(a)), float(np.nanstd(a, ddof=1)) if a.size>1 else np.nan, int(a.size))
    meanB, sdB, nB = (np.nan, np.nan, 0) if b.size==0 else (float(np.nanmean(b)), float(np.nanstd(b, ddof=1)) if b.size>1 else np.nan, int(b.size))
    if a.size>=2 and b.size>=2:
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
        try:
            u_stat, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
        except Exception:
            u_stat, p_u = (np.nan, np.nan)
        d  = cohens_d_unpaired(a, b)
        cd = cliffs_delta(a, b)
    else:
        t_stat = p_t = u_stat = p_u = d = cd = np.nan

    # plots
    safe_var = re.sub(r'[^A-Za-z0-9_]+','', str(var_base))
    base = f"cross_{safe_var}_post{post_side}_vs_healthy{hlt_side}"
    box_png    = os.path.join(out_dir, f"box_{base}.png")
    violin_png = os.path.join(out_dir, f"violin_{base}.png")
    simple_boxplot([a, b], [f"post-{post_side}", f"healthy-{hlt_side}"],
                   title=f"CROSS {var_base}: post-{post_side} vs healthy-{hlt_side}", out_png=box_png)
    violin_plot([a, b], [f"post-{post_side}", f"healthy-{hlt_side}"],
                title=f"CROSS {var_base}: post-{post_side} vs healthy-{hlt_side}", out_png=violin_png)

    return {
        'variable_base': var_base,
        'post_side': post_side, 'healthy_side': hlt_side,
        'mean_post': meanA, 'sd_post': sdA, 'n_post': int(nA),
        'mean_healthy': meanB, 'sd_healthy': sdB, 'n_healthy': int(nB),
        'welch_t_stat': float(t_stat) if np.isfinite(t_stat) else np.nan,
        'welch_t_pvalue': float(p_t) if np.isfinite(p_t) else np.nan,
        'mannwhitney_u': float(u_stat) if np.isfinite(u_stat) else np.nan,
        'mannwhitney_pvalue': float(p_u) if np.isfinite(p_u) else np.nan,
        'cohens_d': float(d) if np.isfinite(d) else np.nan,
        'cliffs_delta': float(cd) if np.isfinite(cd) else np.nan,
        'boxplot_png': os.path.basename(box_png),
        'violin_png':  os.path.basename(violin_png),
    }

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--post', required=True, help='CSV: all_subjects_summary (post-stroke)')
    ap.add_argument('--healthy', required=True, help='CSV: all_subjects_summary (healthy)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου')
    args = ap.parse_args()

    ensure_dir(args.out)

    df_post    = read_summary(args.post,    'post')
    df_healthy = read_summary(args.healthy, 'healthy')

    # ---------- ΕΝΤΟΣ-ΟΜΑΔΑΣ ----------
    rows_within = []

    # POST: P vs N
    post_vars = sorted(df_post['variable'].unique())
    for var in post_vars:
        sub = df_post[df_post['variable'] == var].copy()
        if sub.empty or 'side' not in sub.columns:
            continue
        sub = sub[sub['side'].isin(['P','N'])]
        if sub.empty:
            continue
        rows_within.append(
            side_stats_block(
                df=sub, side_a='P', side_b='N',
                label_a='P (paretic)', label_b='N (non-paretic)',
                out_dir=args.out, var=var, group_tag='post'
            )
        )

    # HEALTHY: L vs R
    healthy_vars = sorted(df_healthy['variable'].unique())
    for var in healthy_vars:
        sub = df_healthy[df_healthy['variable'] == var].copy()
        if sub.empty or 'side' not in sub.columns:
            continue
        sub = sub[sub['side'].isin(['L','R'])]
        if sub.empty:
            continue
        rows_within.append(
            side_stats_block(
                df=sub, side_a='L', side_b='R',
                label_a='L', label_b='R',
                out_dir=args.out, var=var, group_tag='healthy'
            )
        )

    pd.DataFrame(rows_within).to_csv(os.path.join(args.out, "side_stats.csv"), index=False)

    # ---------- ΜΕΤΑΞΥ-ΟΜΑΔΩΝ (CROSS) ----------
    rows_cross = []
    # κοινές base-μεταβλητές (μετά το strip pside_/nside_)
    common_bases = sorted(set(df_post['variable_base']).intersection(set(df_healthy['variable_base'])))
    for vb in common_bases:
        for ps in ['P','N']:
            for hs in ['L','R']:
                rows_cross.append(
                    cross_group_block(df_post, df_healthy, vb, ps, hs, args.out)
                )

    pd.DataFrame(rows_cross).to_csv(os.path.join(args.out, "cross_side_stats.csv"), index=False)

    print(f"[DONE] Wrote: {os.path.join(args.out, 'side_stats.csv')}")
    print(f"[DONE] Wrote: {os.path.join(args.out, 'cross_side_stats.csv')}")
    print(f"[DONE] Plots in: {os.path.abspath(args.out)}")

if __name__ == '__main__':
    main()