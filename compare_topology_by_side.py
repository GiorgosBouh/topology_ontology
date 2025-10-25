#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_topology_by_side.py
------------------------------------------
Side-aware σύγκριση post-stroke (P/N) έναντι healthy (B) για H1 total persistence.

Είσοδοι:
  --post    : CSV all_subjects_summary από post-stroke export
  --healthy : CSV all_subjects_summary από healthy export
  --out     : φάκελος εξόδου

Παράγει:
  - group_stats_by_side.csv (όλες οι συγκρίσεις ανά variable_base)
  - raw_*.csv (raw τιμές ανά σύγκριση)
  - box_*.png, violin_*.png (plots ανά σύγκριση)
"""

import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ------------------------------ helpers ------------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sanitize_subject_label(s):
    s = str(s)
    m = re.search(r'(\d+)', s)
    if m:
        return f"sub{int(m.group(1)):03d}"
    return re.sub(r'[^A-Za-z0-9]+','', s)

def normalize_variable_base(varname: str) -> str:
    """
    Ενώνει ονοματοδοσίες:
      post:  pside_gasnorm / nside_gasnorm / AnkleAngles_x / ...
      healthy: gasnorm / AnkleAngles_x / ...
    Επιστρέφει κοινή "βάση" μεταβλητής για συγκρίσεις.
    """
    v = str(varname).strip().lower()
    # αφαίρεσε prefixes pside_/nside_/bside_ αν υπάρχουν
    v = re.sub(r'^(pside_|nside_|bside_)', '', v)
    return v

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

def boxplot_two(a, b, labels, title, out_png):
    if len(a) == 0 or len(b) == 0:
        return
    plt.figure()
    data = [np.asarray(a, float), np.asarray(b, float)]
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def violin_two(a, b, labels, title, out_png):
    if len(a) == 0 or len(b) == 0:
        return
    plt.figure()
    data = [np.asarray(a, float), np.asarray(b, float)]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks([1,2], labels)
    plt.ylabel("H1 total persistence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def aggregate_post(df_post):
    """
    Επιστρέφει δύο πίνακες:
      post_P: per-subject ανά variable_base (μόνο side=P)
      post_N: per-subject ανά variable_base (μόνο side=N)
    """
    d = df_post.copy()
    d['variable_base'] = d['variable'].map(normalize_variable_base)
    d['h1_total_persistence'] = pd.to_numeric(d['h1_total_persistence'], errors='coerce')
    d = d[d['h1_total_persistence'].notna()]
    d['subject_norm'] = d['subject'].map(sanitize_subject_label)

    post_P = d[d['side'].astype(str).str.upper()=='P'].groupby(
        ['subject_norm','variable_base'], as_index=False
    )['h1_total_persistence'].mean()

    post_N = d[d['side'].astype(str).str.upper()=='N'].groupby(
        ['subject_norm','variable_base'], as_index=False
    )['h1_total_persistence'].mean()

    return post_P, post_N

def aggregate_healthy(df_healthy):
    """
    Healthy: οι τιμές είναι ήδη B (ή ανά subject). Παίρνουμε per-subject
    mean ανά variable_base.
    """
    d = df_healthy.copy()
    d['variable_base'] = d['variable'].map(normalize_variable_base)
    d['h1_total_persistence'] = pd.to_numeric(d['h1_total_persistence'], errors='coerce')
    d = d[d['h1_total_persistence'].notna()]
    d['subject_norm'] = d['subject'].map(sanitize_subject_label)

    healthy_B = d.groupby(
        ['subject_norm','variable_base'], as_index=False
    )['h1_total_persistence'].mean()

    return healthy_B

def collect_values(agg_df, var):
    """Επιστρέφει numpy array τιμών για συγκεκριμένο variable_base."""
    sub = agg_df[agg_df['variable_base']==var]['h1_total_persistence'].astype(float).to_numpy()
    # ρίξε NaN
    sub = sub[np.isfinite(sub)]
    return sub

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--post', required=True, help='CSV: all_subjects_summary (post-stroke)')
    ap.add_argument('--healthy', required=True, help='CSV: all_subjects_summary (healthy)')
    ap.add_argument('--out', required=True, help='Φάκελος εξόδου')
    args = ap.parse_args()

    ensure_dir(args.out)

    # Φόρτωση
    post = pd.read_csv(args.post)
    healthy = pd.read_csv(args.healthy)

    # Aggregations
    post_P, post_N = aggregate_post(post)
    healthy_B      = aggregate_healthy(healthy)

    # Κοινές μεταβλητές (επί της βάσης)
    vars_post = set(post_P['variable_base']).union(set(post_N['variable_base']))
    vars_hlt  = set(healthy_B['variable_base'])
    common_vars = sorted(vars_post.intersection(vars_hlt))
    if not common_vars:
        print("[ERROR] Δεν βρέθηκαν κοινές μεταβλητές.")
        return

    rows = []
    p_all = []  # για FDR
    which_test = []  # 't' ή 'u' για να αποθηκεύσουμε ξεχωριστά

    def one_comparison(values_a, values_b, label_a, label_b, var, tag):
        """Τρέχει όλα τα stats & plots για ένα ζεύγος groups."""
        a = np.asarray(values_a, float); a = a[np.isfinite(a)]
        b = np.asarray(values_b, float); b = b[np.isfinite(b)]
        n_a, n_b = len(a), len(b)

        # περιγραφικά
        mean_a = float(np.nanmean(a)) if n_a else np.nan
        std_a  = float(np.nanstd(a, ddof=1)) if n_a>1 else np.nan
        mean_b = float(np.nanmean(b)) if n_b else np.nan
        std_b  = float(np.nanstd(b, ddof=1)) if n_b>1 else np.nan

        # tests
        t_stat, p_t = welch_ttest(a, b)
        u_stat, p_u = mannwhitney(a, b)
        d  = cohens_d(a, b)
        cd = cliffs_delta(a, b)

        # raw dumps
        safe_var = re.sub(r'[^A-Za-z0-9_]+','', str(var))
        comp_id  = f"{tag}_{safe_var}_{label_a}_vs_{label_b}"
        pd.DataFrame({'group':label_a,'value':a}).to_csv(os.path.join(args.out, f'raw_{comp_id}_{label_a}.csv'), index=False)
        pd.DataFrame({'group':label_b,'value':b}).to_csv(os.path.join(args.out, f'raw_{comp_id}_{label_b}.csv'), index=False)

        # plots (παράλειψη αν λείπει κάποιο group)
        box_png   = os.path.join(args.out, f"box_{comp_id}.png")
        violin_png= os.path.join(args.out, f"violin_{comp_id}.png")
        boxplot_two(a, b, [label_a, label_b], title=f"{tag} — {var}: {label_a} vs {label_b}", out_png=box_png)
        violin_two(a, b, [label_a, label_b], title=f"{tag} — {var}: {label_a} vs {label_b}", out_png=violin_png)

        # καταγραφή
        rows.append({
            'variable_base': var,
            'comparison': f"{tag}: {label_a} vs {label_b}",
            'n_'+label_a: int(n_a), 'mean_'+label_a: mean_a, 'std_'+label_a: std_a,
            'n_'+label_b: int(n_b), 'mean_'+label_b: mean_b, 'std_'+label_b: std_b,
            'welch_t_stat': t_stat, 'welch_t_pvalue': p_t,
            'mannwhitney_u': u_stat, 'mannwhitney_pvalue': p_u,
            'cohens_d': d, 'cliffs_delta': cd,
            'boxplot_png': os.path.basename(box_png) if n_a and n_b else '',
            'violin_png':  os.path.basename(violin_png) if n_a and n_b else ''
        })
        p_all.append(p_t); which_test.append('t')
        p_all.append(p_u); which_test.append('u')

    # Τρέξε συγκρίσεις για κάθε κοινή μεταβλητή
    for var in common_vars:
        vals_P = collect_values(post_P, var)
        vals_N = collect_values(post_N, var)
        vals_B = collect_values(healthy_B, var)

        # 1) P vs N (μόνο αν υπάρχουν και τα δύο)
        if len(vals_P) > 0 and len(vals_N) > 0:
            one_comparison(vals_P, vals_N, 'P', 'N', var, tag='post')

        # 2) P vs B
        if len(vals_P) > 0 and len(vals_B) > 0:
            one_comparison(vals_P, vals_B, 'P', 'B', var, tag='post_vs_healthy')

        # 3) N vs B
        if len(vals_N) > 0 and len(vals_B) > 0:
            one_comparison(vals_N, vals_B, 'N', 'B', var, tag='post_vs_healthy')

    # FDR (ξεχωριστά labels για t/u)
    if p_all:
        reject, qvals, _, _ = multipletests(p_all, method='fdr_bh')
        # γράψε τα q-values πίσω κατά σειρά
        qi = 0
        for r in rows:
            # γράφουμε q για t & u ξεχωριστά
            if np.isfinite(r.get('welch_t_pvalue', np.nan)):
                r['welch_t_qvalue'] = float(qvals[qi]); r['welch_t_reject_fdr05'] = bool(reject[qi]); qi += 1
            else:
                r['welch_t_qvalue'] = np.nan; r['welch_t_reject_fdr05'] = False
            if np.isfinite(r.get('mannwhitney_pvalue', np.nan)):
                r['mannwhitney_qvalue'] = float(qvals[qi]); r['mannwhitney_reject_fdr05'] = bool(reject[qi]); qi += 1
            else:
                r['mannwhitney_qvalue'] = np.nan; r['mannwhitney_reject_fdr05'] = False

    out_csv = os.path.join(args.out, "group_stats_by_side.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Plots/raw in: {os.path.abspath(args.out)}")

if __name__ == '__main__':
    main()