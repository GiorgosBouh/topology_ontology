#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_topology.py
------------------------------------------
Σύγκριση post-stroke vs healthy με βάση τα H1 metrics από τα summaries.

Είσοδοι:
  - Δύο αρχεία CSV (all_subjects_summary.csv) από τα runs του xlsx_to_ph.py
    * το ένα για post-stroke
    * το άλλο για healthy

Τι κάνει:
  1) Φορτώνει τα δύο summaries και βάζει group labels.
  2) Καθαρίζει/κρατά μόνο έγκυρα H1 (χωρίς NaN).
  3) Κάνει aggregation PER SUBJECT (μέσος όρος στα sides, για να μην διπλομετράμε).
  4) Για κάθε "variable":
       - υπολογίζει μέσους όρους/διασπορά ανά group
       - κάνει t-test (Welch) και Mann–Whitney U
       - υπολογίζει effect sizes (Cohen's d, Cliff's delta)
       - σώζει boxplot PNG
  5) Γράφει συγκεντρωτικό CSV: group_stats.csv

Χρήση:
  python compare_topology.py \
      --post ~/topology/out_post_xlsx/all_subjects_summary.csv \
      --healthy ~/topology/out_healthy_xlsx/all_subjects_summary.csv \
      --out ~/topology/compare_post_vs_healthy
"""

import argparse, os, math, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------ helpers ------------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_summary(path, group_label):
    df = pd.read_csv(path)
    df['group'] = group_label
    # τυποποίηση ονομάτων για σιγουριά
    # περιμένουμε: subject, side, variable, h1_points, h1_total_persistence (ή error)
    for col in ['subject','side','variable','h1_points','h1_total_persistence','error','group']:
        if col not in df.columns:
            pass  # ορισμένα πεδία (π.χ. error) μπορεί να λείπουν
    return df

def sanitize_subject_label(s):
    s = str(s)
    # αν είναι "post_sub001" ή "healthy_sub034" κράτα μόνο τον αριθμό για ομαλοποίηση,
    # αλλιώς κράτα το αρχικό id
    m = re.search(r'(\d+)', s)
    if m:
        return f"sub{int(m.group(1)):03d}"
    return re.sub(r'[^A-Za-z0-9]+','', s)

def aggregate_per_subject(df):
    """
    Για κάθε subject & variable, πάρε ΜΕΣΟ ΟΡΟ στα sides,
    ώστε κάθε subject να συνεισφέρει μία τιμή ανά variable.
    """
    # φιλτράρουμε μη διαθέσιμα H1
    df = df.copy()
    if 'h1_total_persistence' not in df.columns:
        return df.iloc[0:0]
    df = df[pd.to_numeric(df['h1_total_persistence'], errors='coerce').notna()].copy()
    df['h1_total_persistence'] = df['h1_total_persistence'].astype(float)
    # κανονικοποίησε subject ids για ματσάρισμα
    df['subject_norm'] = df['subject'].map(sanitize_subject_label)
    grp = df.groupby(['group','subject_norm','variable'], dropna=False, as_index=False)['h1_total_persistence'].mean()
    return grp

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    # pooled std (unbiased)
    s = np.sqrt(((na-1)*va + (nb-1)*vb) / (na + nb - 2)) if (na+nb-2)>0 else np.nan
    if s == 0 or not np.isfinite(s): return np.nan
    return (np.mean(a) - np.mean(b)) / s

def cliffs_delta(a, b):
    """
    Cliff's delta: effect size για μη-παραμετρική σύγκριση.
    τιμές ~ [-1, 1], 0=καμία επίδραση
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    # O(N log N) approx. (απλό O(N*M) είναι ΟΚ για μικρά)
    # εδώ για απλότητα, O(N*M)
    n_concordant = 0
    n_discordant = 0
    for x in a:
        n_concordant += np.sum(x > b)
        n_discordant += np.sum(x < b)
    n_ties = len(a)*len(b) - n_concordant - n_discordant  # not used
    delta = (n_concordant - n_discordant) / (len(a)*len(b))
    return float(delta)

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
    # use two-sided, auto continuity
    try:
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        return float(u), float(p)
    except Exception:
        return np.nan, np.nan

def simple_boxplot(a, b, labels, title, out_png):
    plt.figure()
    data = [np.asarray(a, float), np.asarray(b, float)]
    plt.boxplot(data, labels=labels, showfliers=True)
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

    df_post   = read_summary(args.post,   'post')
    df_healthy= read_summary(args.healthy,'healthy')

    # Aggregate per subject (mean across sides)
    ag_post    = aggregate_per_subject(df_post)
    ag_healthy = aggregate_per_subject(df_healthy)

    if ag_post.empty or ag_healthy.empty:
        print("[ERROR] Άδειο aggregation—έλεγξε τα input CSVs.")
        return

    # κοινές μεταβλητές για δίκαιη σύγκριση
    common_vars = sorted(set(ag_post['variable']).intersection(set(ag_healthy['variable'])))
    if not common_vars:
        print("[ERROR] Δεν βρέθηκαν κοινές 'variable' ανάμεσα στις ομάδες.")
        return

    rows = []
    for var in common_vars:
        a = ag_post.loc[ag_post['variable']==var, 'h1_total_persistence'].to_numpy(float)
        b = ag_healthy.loc[ag_healthy['variable']==var, 'h1_total_persistence'].to_numpy(float)

        # περιγραφικά
        mean_post, std_post, n_post = np.nanmean(a), np.nanstd(a, ddof=1), np.sum(np.isfinite(a))
        mean_hlt,  std_hlt,  n_hlt  = np.nanmean(b), np.nanstd(b, ddof=1), np.sum(np.isfinite(b))

        # tests & effect sizes
        t_stat, p_t = welch_ttest(a, b)
        u_stat, p_u = mannwhitney(a, b)
        d = cohens_d(a, b)
        cd = cliffs_delta(a, b)

        # plot
        safe_var = re.sub(r'[^A-Za-z0-9_]+','', str(var))
        out_png = os.path.join(args.out, f"box_{safe_var}.png")
        simple_boxplot(a, b, labels=['post','healthy'], title=f"{var} — H1 total persistence", out_png=out_png)

        rows.append({
            'variable': var,
            'mean_post': mean_post, 'std_post': std_post, 'n_post': int(n_post),
            'mean_healthy': mean_hlt, 'std_healthy': std_hlt, 'n_healthy': int(n_hlt),
            'welch_t_stat': t_stat, 'welch_t_pvalue': p_t,
            'mannwhitney_u': u_stat, 'mannwhitney_pvalue': p_u,
            'cohens_d': d, 'cliffs_delta': cd,
            'boxplot_png': os.path.basename(out_png),
        })

    out_csv = os.path.join(args.out, "group_stats.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Plots in: {os.path.abspath(args.out)}")

if __name__ == '__main__':
    main()
    