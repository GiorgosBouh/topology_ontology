#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_topology.py (patched)
------------------------------------------
Σύγκριση post-stroke vs healthy με βάση τα H1 metrics από τα summaries.

Βελτιώσεις:
  1) Ασφαλές dtype cast (astype(float).to_numpy()).
  2) Αποθήκευση raw τιμών ανά variable (CSV) για εύκολο replot/stats.
  3) Επιπλέον violin plots πέρα από boxplots.
  4) Έλεγχος πολλαπλών συγκρίσεων (Benjamini–Hochberg FDR) για Welch-t & Mann–Whitney.
  5) ΕΝΟΠΟΙΗΣΗ ονομάτων μεταβλητών (strip pside_/nside_ κ.λπ.) ώστε να ταιριάζουν ομάδες.

Χρήση:
  python compare_topology.py \
      --post    ~/topology/out_post_xlsx_col/all_subjects_summary.csv \
      --healthy ~/topology/out_healthy_xlsx_col/all_subjects_summary.csv \
      --out     ~/topology/compare_post_vs_healthy
"""

import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests  # (4)

# ------------------------------ helpers ------------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_summary(path, group_label):
    df = pd.read_csv(path)
    df['group'] = group_label
    # περιμένουμε: subject, side, variable, h1_points, h1_total_persistence (ή error)
    return df

# --- ομογενοποίηση subject id για ομαδοποίηση ανά άτομο ---
def sanitize_subject_label(s):
    s = str(s)
    m = re.search(r'(\d+)', s)
    if m:
        return f"sub{int(m.group(1)):03d}"
    return re.sub(r'[^A-Za-z0-9]+','', s)

# --- ΕΝΟΠΟΙΗΣΗ ονομάτων μεταβλητών ανά group ---
def normalize_variable_name(v, group):
    """
    Επιστρέφει εναρμονισμένο όνομα μεταβλητής:
    - Post-stroke: αφαιρεί prefixes πλευράς (pside_/nside_) για να ταιριάζει με healthy.
    - Τα υπόλοιπα κρατιούνται ως έχουν (π.χ. ankleangles_x, pelvisangles, gasnorm, κ.λπ.).
    """
    v0 = str(v).strip().lower()
    if group == 'post':
        v0 = re.sub(r'^(pside_|nside_)', '', v0)  # strip side prefixes
    # μπορούμε εδώ να περάσουμε και μικρούς χαρτογράφους αν χρειαστεί:
    # mappings = {'hipanglesx':'hipangles_x', ...}
    # v0 = mappings.get(v0, v0)
    return v0

def normalize_df_variables(df, group_label):
    df = df.copy()
    if 'variable' in df.columns:
        df['variable'] = df['variable'].apply(lambda x: normalize_variable_name(x, group_label))
    if 'subject' in df.columns:
        df['subject'] = df['subject'].astype(str)
    return df

def aggregate_per_subject(df):
    """
    Για κάθε subject & variable, μέσος όρος στα sides,
    ώστε κάθε subject να συνεισφέρει μία τιμή ανά variable.
    """
    df = df.copy()
    if 'h1_total_persistence' not in df.columns:
        return df.iloc[0:0]
    df['h1_total_persistence'] = pd.to_numeric(df['h1_total_persistence'], errors='coerce')
    df = df[df['h1_total_persistence'].notna()].copy()
    df['subject_norm'] = df['subject'].map(sanitize_subject_label)
    grp = df.groupby(['group','subject_norm','variable'], dropna=False, as_index=False)['h1_total_persistence'].mean()
    return grp

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    denom = (na + nb - 2)
    if denom <= 0: return np.nan
    s = np.sqrt(((na-1)*va + (nb-1)*vb) / denom)
    if s == 0 or not np.isfinite(s): return np.nan
    return (np.mean(a) - np.mean(b)) / s

def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    n_conc = 0; n_disc = 0
    for x in a:
        n_conc += np.sum(x > b)
        n_disc += np.sum(x < b)
    return float((n_conc - n_disc) / (len(a)*len(b)))

def welch_ttest(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

def mannwhitney(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    try:
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        return float(u), float(p)
    except Exception:
        return np.nan, np.nan

def simple_boxplot(a, b, labels, title, out_png):
    plt.figure()
    data = [np.asarray(a, float), np.asarray(b, float)]
    # Matplotlib 3.9+: χρησιμοποιούμε tick_labels αντί labels
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def violin_plot(a, b, title, out_png):
    plt.figure()
    data = [np.asarray(a, float), np.asarray(b, float)]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks([1,2], ['post','healthy'])
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--post', required=True, help='CSV: all_subjects_summary (post-stroke)')
    ap.add_argument('--healthy', required=True, help='CSV: all_subjects_summary (healthy)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου για plots & group_stats.csv')
    args = ap.parse_args()

    ensure_dir(args.out)

    # διαβάζουμε & ενοποιούμε ονόματα μεταβλητών
    df_post    = normalize_df_variables(read_summary(args.post,    'post'),    'post')
    df_healthy = normalize_df_variables(read_summary(args.healthy, 'healthy'), 'healthy')

    # Aggregate per subject (mean across sides)
    ag_post    = aggregate_per_subject(df_post)
    ag_healthy = aggregate_per_subject(df_healthy)

    if ag_post.empty or ag_healthy.empty:
        print("[ERROR] Άδειο aggregation—έλεγξε τα input CSVs.")
        return

    # Κοινές μεταβλητές
    common_vars = sorted(set(ag_post['variable']).intersection(set(ag_healthy['variable'])))
    if not common_vars:
        print("[ERROR] Δεν βρέθηκαν κοινές 'variable' ανάμεσα στις ομάδες.")
        return

    rows = []
    pvals_t = []
    pvals_u = []

    for var in common_vars:
        a = ag_post.loc[ag_post['variable']==var, 'h1_total_persistence'].astype(float).to_numpy()
        b = ag_healthy.loc[ag_healthy['variable']==var, 'h1_total_persistence'].astype(float).to_numpy()

        # περιγραφικά
        mean_post, std_post, n_post = np.nanmean(a), np.nanstd(a, ddof=1), np.sum(np.isfinite(a))
        mean_hlt,  std_hlt,  n_hlt  = np.nanmean(b), np.nanstd(b, ddof=1), np.sum(np.isfinite(b))

        # tests & effect sizes
        t_stat, p_t = welch_ttest(a, b)
        u_stat, p_u = mannwhitney(a, b)
        d  = cohens_d(a, b)
        cd = cliffs_delta(a, b)

        # raw dumps (2)
        safe_var = re.sub(r'[^A-Za-z0-9_]+','', str(var))
        pd.DataFrame({'group':'post', 'value':a}).to_csv(os.path.join(args.out, f'raw_{safe_var}_post.csv'), index=False)
        pd.DataFrame({'group':'healthy','value':b}).to_csv(os.path.join(args.out, f'raw_{safe_var}_healthy.csv'), index=False)

        # plots (3)
        box_png    = os.path.join(args.out, f"box_{safe_var}.png")
        violin_png = os.path.join(args.out, f"violin_{safe_var}.png")
        simple_boxplot(a, b, labels=['post','healthy'], title=f"{var} — H1 total persistence", out_png=box_png)
        violin_plot(a, b, title=f"{var} — H1 total persistence", out_png=violin_png)

        rows.append({
            'variable': var,
            'mean_post': mean_post, 'std_post': std_post, 'n_post': int(n_post),
            'mean_healthy': mean_hlt, 'std_healthy': std_hlt, 'n_healthy': int(n_hlt),
            'welch_t_stat': t_stat, 'welch_t_pvalue': p_t,
            'mannwhitney_u': u_stat, 'mannwhitney_pvalue': p_u,
            'cohens_d': d, 'cliffs_delta': cd,
            'boxplot_png': os.path.basename(box_png),
            'violin_png':  os.path.basename(violin_png),
        })
        pvals_t.append(p_t)
        pvals_u.append(p_u)

    # (4) FDR (Benjamini–Hochberg) για p-values
    try:
        rej_t, q_t, _, _ = multipletests(pvals_t, method='fdr_bh')
        rej_u, q_u, _, _ = multipletests(pvals_u, method='fdr_bh')
        for i in range(len(rows)):
            rows[i]['welch_t_qvalue'] = float(q_t[i])
            rows[i]['mannwhitney_qvalue'] = float(q_u[i])
            rows[i]['welch_t_reject_fdr05'] = bool(rej_t[i])
            rows[i]['mannwhitney_reject_fdr05'] = bool(rej_u[i])
    except Exception:
        pass  # αν κάτι πάει στραβά, συνεχίζουμε χωρίς q-values

    out_csv = os.path.join(args.out, "group_stats.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Plots in: {os.path.abspath(args.out)}")

if __name__ == '__main__':
    main()