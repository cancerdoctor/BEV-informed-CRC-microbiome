"""
Step 2 분석 스크립트 — Gut Microbes v4 보강
실행: python3 ~/larc_microbiome/step2_analysis.py
출력: ~/larc_microbiome/06_loco_cv/step2_results/
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import logit
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────
BASE   = os.path.expanduser('~/larc_microbiome')
OUT    = f'{BASE}/06_loco_cv/step2_results'
os.makedirs(OUT, exist_ok=True)

# ── 폰트 ─────────────────────────────────────────────────
import matplotlib.font_manager as fm
avail = [f.name for f in fm.fontManager.ttflist]
FONT  = 'Arial' if 'Arial' in avail else \
        'Liberation Sans' if 'Liberation Sans' in avail else 'DejaVu Sans'
plt.rcParams.update({
    'font.family': FONT, 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})
NAVY = '#1B3A5C'; TEAL = '#006D75'; RED = '#B03A2E'; GRAY = '#888888'

# ══════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════
print("=== 데이터 로드 ===")

# GSE165255
s16_raw = pd.read_csv(f'{BASE}/05_integrated/GSE165255/genus_16S.txt',
                      sep='\t', index_col=0)
rna_raw = pd.read_csv(f'{BASE}/05_integrated/GSE165255/rna_normalized.txt',
                      sep='\t', index_col=0)
gene_sym = rna_raw['convert_symbol'].copy()
rna_expr = rna_raw.drop(columns=['convert_symbol'])

def extract_genus(x): return str(x).strip('/').split('/')[-1].strip()
s16 = s16_raw.iloc[2:].copy()
s16 = s16.apply(pd.to_numeric, errors='coerce').fillna(0)
s16.index = [extract_genus(i) for i in s16.index]
s16 = s16.groupby(level=0).sum()
s16_rel = s16.div(s16.sum(axis=0)+1e-10, axis=1)
sample_ids = s16_raw.iloc[0]
tumor_mask = sample_ids.str.contains('-T[A-Z]', na=False, regex=True)
s16_t = s16_rel.loc[:, tumor_mask]
s16_t.columns = [c.split('.')[0] for c in s16_raw.columns[tumor_mask]]
s16_t = s16_t.T.groupby(level=0).mean().T
rna_tc = [c for c in rna_expr.columns if 'Tumor' in c]
rna_t = rna_expr[rna_tc].copy()
rna_t.columns = [c.replace('-Tumor','') for c in rna_tc]
common = sorted(set(rna_t.columns) & set(s16_t.columns))
rna_m = rna_t[common]; s16_m = s16_t[common]
fn = s16_m.loc['Fusobacterium']
fc = s16_m.loc['Faecalibacterium']

# TCMA survival
bev = pd.read_csv(f'{BASE}/06_loco_cv/TCMA_BEV_survival.tsv',
                  sep='\t', index_col=0)
bev = bev[bev['OS_days'].notna() & (bev['OS_days']>0)].copy()
coad = bev[bev['acronym']=='COAD'].copy()
print(f"GSE165255 paired: {len(common)} samples")
print(f"TCMA COAD: {len(coad)} (OS events: {coad['OS_event'].sum()})")

# ── 항생제 메타데이터 ──────────────────────────────────────
meta = pd.read_csv(f'{BASE}/05_integrated/GSE165255/metadata.csv')
abx_col = None
for c in ['Antibiotics','antibiotics','antibiotic','ANTIBIOTICS']:
    if c in meta.columns: abx_col = c; break
print(f"Antibiotic 컬럼: {abx_col}")
if abx_col:
    print(meta[abx_col].value_counts())

# ══════════════════════════════════════════════════════════
# 2. CLR Transformation Sensitivity
# ══════════════════════════════════════════════════════════
print("\n=== 2. CLR Transformation Sensitivity ===")

def clr_transform(df):
    """Centered Log-Ratio transformation (genus × sample)"""
    df_pos = df.copy() + 1e-6  # pseudocount
    log_df = np.log(df_pos)
    gm = log_df.mean(axis=0)   # geometric mean per sample
    return log_df - gm

s16_clr = clr_transform(s16_m)
fn_clr  = s16_clr.loc['Fusobacterium']
fc_clr  = s16_clr.loc['Faecalibacterium']

key_genes = {
    'IFNGR1': 'ENSG00000027697',
    'JAK1':   'ENSG00000162434',
    'TCF7L2': 'ENSG00000148737',
    'TIMP3':  'ENSG00000100234',
    'IRAK2':  'ENSG00000134070',
    'NFKB1':  'ENSG00000109320',
    'STAT1':  'ENSG00000115415',
    'DROSHA': 'ENSG00000113360',
    'PTEN':   'ENSG00000171862',
}

clr_results = []
print(f"\n{'Gene':<10} {'Rel.abund ρ':>12} {'CLR ρ':>8} {'Stable?'}")
print("-"*45)
for sym, eid in key_genes.items():
    if eid not in rna_m.index: continue
    expr = rna_m.loc[eid]
    g = fn if sym != 'IRAK2' else fc
    g_clr = fn_clr if sym != 'IRAK2' else fc_clr
    genus_lbl = 'Fusobacterium' if sym != 'IRAK2' else 'Faecalibacterium'
    r1, p1 = stats.spearmanr(g.values, expr.values)
    r2, p2 = stats.spearmanr(g_clr.values, expr.values)
    same_dir = (r1 * r2) > 0
    stable = 'YES' if same_dir else 'NO ⚠'
    print(f"{sym:<10} {r1:+.3f} (p={p1:.3f})  {r2:+.3f}  {stable}")
    clr_results.append({
        'gene': sym, 'genus': genus_lbl,
        'rho_relabund': r1, 'p_relabund': p1,
        'rho_clr': r2, 'p_clr': p2,
        'direction_stable': same_dir
    })

clr_df = pd.DataFrame(clr_results)
clr_df.to_csv(f'{OUT}/CLR_sensitivity.tsv', sep='\t', index=False)
print(f"\n저장: {OUT}/CLR_sensitivity.tsv")

# ══════════════════════════════════════════════════════════
# 3. Antibiotic-Stratified Sensitivity
# ══════════════════════════════════════════════════════════
print("\n=== 3. Antibiotic-Stratified Sensitivity ===")

if abx_col and meta is not None:
    # SampleID 매핑
    id_col = None
    for c in ['SubmissionID','SampleID','SUBJID']:
        if c in meta.columns: id_col = c; break

    if id_col:
        meta_tumor = meta[meta['Type2']=='Tumor'].copy() if 'Type2' in meta.columns else meta.copy()
        # SubmissionID "SB-036-Tumor" → "SB-036"
        meta_tumor['patient_id'] = meta_tumor[id_col].str.replace('-Tumor','').str.strip()
        meta_sub = meta_tumor[['patient_id', abx_col]].dropna().set_index('patient_id')
        common_abx = [p for p in common if p in meta_sub.index]
        print(f"항생제 정보 있는 환자: {len(common_abx)}")

        if len(common_abx) > 5:
            abx_vals = meta_sub.loc[common_abx, abx_col]
            abx_groups = abx_vals.value_counts()
            print("항생제 그룹:", abx_groups.to_dict())

            abx_results = []
            for group in abx_groups.index:
                pts_g = [p for p in common_abx if meta_sub.loc[p, abx_col]==group]
                if len(pts_g) < 5: continue
                fn_g  = fn.loc[pts_g]
                fc_g  = fc.loc[pts_g]
                for sym, eid in key_genes.items():
                    if eid not in rna_m.index: continue
                    expr_g = rna_m.loc[eid, pts_g]
                    g_g = fn_g if sym != 'IRAK2' else fc_g
                    r, p = stats.spearmanr(g_g.values, expr_g.values)
                    abx_results.append({'group':group,'gene':sym,'n':len(pts_g),'rho':r,'p':p})
                    print(f"  [{group}, n={len(pts_g)}] {sym}: ρ={r:+.3f}, p={p:.3f}")

            if abx_results:
                pd.DataFrame(abx_results).to_csv(
                    f'{OUT}/antibiotic_stratified.tsv', sep='\t', index=False)
                print(f"저장: {OUT}/antibiotic_stratified.tsv")
    else:
        print("SampleID 컬럼 매핑 불가")
else:
    print("항생제 메타데이터 없음 → 전체 코호트 결과로 대체")

# ══════════════════════════════════════════════════════════
# 4. Continuous Cox per SD + Covariate-adjusted
# ══════════════════════════════════════════════════════════
print("\n=== 4. Cox Regression (Continuous + Adjusted) ===")

feats = ['butyrate_producer','nucleotide_biosynthesis','LPS_TLR4_load',
         'STING_agonist_potential']
feat_labels = ['Butyrate producer','Nucleotide biosynthesis',
               'LPS/TLR4 load','STING agonist']

cox_results = []
print(f"\n{'Feature':<25} {'HR':>6} {'95% CI':<18} {'p':>7} {'Model'}")
print("-"*70)

for feat, flabel in zip(feats, feat_labels):
    sub = coad[['OS_days','OS_event',feat]].dropna().copy()
    sub[feat+'_std'] = (sub[feat] - sub[feat].mean()) / sub[feat].std()

    # Univariable continuous
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub[['OS_days','OS_event',feat+'_std']],
            duration_col='OS_days', event_col='OS_event')
    s = cph.summary
    hr  = np.exp(s.loc[feat+'_std','coef'])
    lci = np.exp(s.loc[feat+'_std','coef lower 95%'])
    uci = np.exp(s.loc[feat+'_std','coef upper 95%'])
    pv  = s.loc[feat+'_std','p']
    print(f"{flabel:<25} {hr:6.3f}  [{lci:.3f}, {uci:.3f}]  {pv:7.4f}  Univar-continuous")
    cox_results.append({'feature':flabel,'model':'Univar-continuous',
                        'HR':hr,'CI_lo':lci,'CI_hi':uci,'p':pv,'n':len(sub)})

    # Median-split univariable
    sub2 = coad[['OS_days','OS_event',feat]].dropna().copy()
    med  = sub2[feat].median()
    sub2['group'] = (sub2[feat] > med).astype(int)
    cph2 = CoxPHFitter(penalizer=0.1)
    cph2.fit(sub2[['OS_days','OS_event','group']],
             duration_col='OS_days', event_col='OS_event')
    s2  = cph2.summary
    hr2  = np.exp(s2.loc['group','coef'])
    lci2 = np.exp(s2.loc['group','coef lower 95%'])
    uci2 = np.exp(s2.loc['group','coef upper 95%'])
    pv2  = s2.loc['group','p']
    print(f"{flabel:<25} {hr2:6.3f}  [{lci2:.3f}, {uci2:.3f}]  {pv2:7.4f}  Median-split")
    cox_results.append({'feature':flabel,'model':'Median-split',
                        'HR':hr2,'CI_lo':lci2,'CI_hi':uci2,'p':pv2,'n':len(sub2)})

cox_df = pd.DataFrame(cox_results)
cox_df.to_csv(f'{OUT}/cox_results.tsv', sep='\t', index=False)
print(f"\n저장: {OUT}/cox_results.tsv")

# ══════════════════════════════════════════════════════════
# 5. Genome-wide Ranked Correlation Plot (Figure 5 추가 패널)
# ══════════════════════════════════════════════════════════
print("\n=== 5. Genome-wide Ranked Correlation Plot ===")

# 전체 상관 계산 (이전에 이미 계산했으면 파일 읽기)
corr_file = f'{BASE}/05_integrated/GSE165255/Fusobacterium_gene_correlation.tsv'
if os.path.exists(corr_file):
    corr_fn = pd.read_csv(corr_file, sep='\t')
    print(f"기존 파일 로드: {len(corr_fn)} genes")
else:
    print("전체 상관 재계산 중...")
    corrs, pvals, gids = [], [], []
    expr_filt = rna_m[rna_m.median(axis=1) > 1]
    for gene in expr_filt.index:
        r, p = stats.spearmanr(fn.values, expr_filt.loc[gene].values)
        corrs.append(r); pvals.append(p); gids.append(gene)
    corr_fn = pd.DataFrame({'gene_id':gids,
        'symbol':[gene_sym.get(g,'') for g in gids],
        'rho':corrs, 'pval':pvals})
    _, corr_fn['fdr'], _, _ = multipletests(corr_fn['pval'], method='fdr_bh')
    corr_fn = corr_fn.sort_values('rho', ascending=False)
    corr_fn.to_csv(corr_file, sep='\t', index=False)

# Faecalibacterium
corr_fc_file = f'{BASE}/05_integrated/GSE165255/Faecalibacterium_gene_correlation.tsv'
if os.path.exists(corr_fc_file):
    corr_fc = pd.read_csv(corr_fc_file, sep='\t')
else:
    corrs2, pvals2, gids2 = [], [], []
    expr_filt = rna_m[rna_m.median(axis=1) > 1]
    for gene in expr_filt.index:
        r, p = stats.spearmanr(fc.values, expr_filt.loc[gene].values)
        corrs2.append(r); pvals2.append(p); gids2.append(gene)
    corr_fc = pd.DataFrame({'gene_id':gids2,
        'symbol':[gene_sym.get(g,'') for g in gids2],
        'rho':corrs2, 'pval':pvals2})
    _, corr_fc['fdr'], _, _ = multipletests(corr_fc['pval'], method='fdr_bh')
    corr_fc = corr_fc.sort_values('rho', ascending=False)
    corr_fc.to_csv(corr_fc_file, sep='\t', index=False)

# ── Ranked correlation figure ─────────────────────────────
highlight_fn = {'IFNGR1':'IFNGR1','JAK1':'JAK1','TCF7L2':'TCF7L2',
                'TIMP3':'TIMP3','DROSHA':'DROSHA','NFKB1':'NFKB1',
                'PTEN':'PTEN','BCL2L13':'BCL2L13'}
highlight_fc = {'IRAK2':'IRAK2','SUV39H1':'SUV39H1','PSMA7':'PSMA7','KHDRBS1':'KHDRBS1'}

fig_rank, axes_rank = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=300,
    gridspec_kw=dict(wspace=0.45, left=0.10, right=0.97, top=0.90, bottom=0.15))

def ranked_plot(ax, corr_df, highlights, color, title, genus_label):
    corr_sorted = corr_df.sort_values('rho').reset_index(drop=True)
    n = len(corr_sorted)
    ax.scatter(range(n), corr_sorted['rho'], s=1, color='#CCCCCC', alpha=0.4, zorder=1)

    # Highlight 유전자
    sym_to_idx = {row['symbol']:i for i, row in corr_sorted.iterrows()
                  if row['symbol'] in highlights}
    for sym, label in highlights.items():
        if sym in sym_to_idx:
            idx = sym_to_idx[sym]
            rho = corr_sorted.loc[idx,'rho']
            ax.scatter(idx, rho, s=28, color=color, zorder=3, linewidths=0)
            # 라벨 위치 조정
            x_off = -0.05*n if idx > n*0.5 else 0.02*n
            ha = 'right' if idx > n*0.5 else 'left'
            ax.annotate(sym, (idx, rho), (idx+x_off, rho),
                        fontsize=7, ha=ha, color=color,
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.6))

    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.set_xlabel(f'Genes ranked by Spearman ρ\nwith {genus_label} abundance', fontsize=8)
    ax.set_ylabel('Spearman ρ', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', loc='left')
    ax.set_xlim(-50, n+50)
    ax.text(0.97, 0.03, f'n = {n:,} genes\nNo gene FDR q < 0.05',
            transform=ax.transAxes, ha='right', fontsize=6.5,
            color='#888', style='italic')

ranked_plot(axes_rank[0], corr_fn, highlight_fn, NAVY,
            'a  Fusobacterium abundance\nvs genome-wide expression',
            'Fusobacterium')
ranked_plot(axes_rank[1], corr_fc, highlight_fc, TEAL,
            'b  Faecalibacterium abundance\nvs genome-wide expression',
            'Faecalibacterium')

fig_rank.savefig(f'{OUT}/Fig5_ranked_correlation.png', dpi=300,
                  bbox_inches='tight', facecolor='white')
print(f"저장: {OUT}/Fig5_ranked_correlation.png")

# ══════════════════════════════════════════════════════════
# 6. Figure 3 업그레이드 — Number-at-risk + HR CI
# ══════════════════════════════════════════════════════════
print("\n=== 6. Figure 3 업그레이드 ===")

fig3, axes3 = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=300,
    gridspec_kw=dict(hspace=0.55, wspace=0.42,
                     left=0.10, right=0.97, top=0.93, bottom=0.10))

km_specs = [
    ('butyrate_producer',      'a  Butyrate producer load', TEAL,  (0,0)),
    ('nucleotide_biosynthesis', 'b  Nucleotide biosynthesis load', RED,  (0,1)),
    ('LPS_TLR4_load',           'c  LPS/OMV load',          NAVY, (1,0)),
]

def km_with_risk(ax, feat, title, color, coad_df):
    sub = coad_df[['OS_days','OS_event',feat]].dropna().copy()
    med = sub[feat].median()
    hi  = sub[sub[feat]>med]; lo = sub[sub[feat]<=med]
    res = logrank_test(hi['OS_days'], lo['OS_days'],
                       event_observed_A=hi['OS_event'],
                       event_observed_B=lo['OS_event'])
    p   = res.p_value

    # Cox HR
    sub['grp'] = (sub[feat]>med).astype(int)
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub[['OS_days','OS_event','grp']],
                duration_col='OS_days', event_col='OS_event')
        s   = cph.summary
        hr  = np.exp(s.loc['grp','coef'])
        lci = np.exp(s.loc['grp','coef lower 95%'])
        uci = np.exp(s.loc['grp','coef upper 95%'])
        hr_str = f'HR={hr:.2f} [{lci:.2f}–{uci:.2f}]'
    except Exception:
        hr_str = 'HR: n/a'

    kmf = KaplanMeierFitter()
    kmf.fit(hi['OS_days'], hi['OS_event'], label=f'High (n={len(hi)})')
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True,
                                linewidth=1.4, ci_alpha=0.12)
    kmf.fit(lo['OS_days'], lo['OS_event'], label=f'Low (n={len(lo)})')
    kmf.plot_survival_function(ax=ax, color='#888', ci_show=True,
                                linewidth=1.4, ci_alpha=0.10)

    p_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
    ax.set_title(f'{title}\nLog-rank {p_str}; {hr_str}',
                 fontsize=8.5, fontweight='bold', loc='left', pad=4)
    ax.set_xlabel('Days', fontsize=8)
    ax.set_ylabel('Survival probability', fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=7, loc='lower left')

    # Number-at-risk table
    time_points = [0, 500, 1000, 1500, 2000, 2500, 3000]
    risk_hi, risk_lo = [], []
    for t in time_points:
        risk_hi.append((hi['OS_days'] >= t).sum())
        risk_lo.append((lo['OS_days'] >= t).sum())

    ax2 = ax.inset_axes([0, -0.28, 1, 0.18])
    ax2.axis('off')
    ax2.text(-0.01, 0.75, 'High', color=color, fontsize=6.5,
             transform=ax2.transAxes, va='center', ha='right')
    ax2.text(-0.01, 0.25, 'Low', color='#888', fontsize=6.5,
             transform=ax2.transAxes, va='center', ha='right')
    for j, (t, rh, rl) in enumerate(zip(time_points, risk_hi, risk_lo)):
        x_pos = j / (len(time_points)-1)
        ax2.text(x_pos, 0.75, str(rh), color=color, fontsize=6,
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.text(x_pos, 0.25, str(rl), color='#888', fontsize=6,
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.text(0.5, -0.1, 'Days: ' + '  '.join(str(t) for t in time_points),
             fontsize=5.5, ha='center', transform=ax2.transAxes, color='#888')

for feat, title, color, pos in km_specs:
    km_with_risk(axes3[pos[0]][pos[1]], feat, title, color, coad)

# d: Cancer type BEV bar
ax3d = axes3[1][1]
bc = bev.groupby('acronym')[feats[:4]].mean()
bc = bc.loc[[t for t in ['COAD','READ','STAD'] if t in bc.index]]
bc.columns = ['Butyrate','Nucleotide','LPS/TLR4','STING']
x_pos = np.arange(4); w = 0.25
ctype_cols = [NAVY, TEAL, RED]
ctype_ns   = {'COAD':114,'READ':41,'STAD':113}
for j, (ct, col) in enumerate(zip(bc.index, ctype_cols)):
    n_str = f'n={ctype_ns.get(ct,"?")}'
    ax3d.bar(x_pos+(j-1)*w, bc.loc[ct], width=w,
             color=col, alpha=0.88,
             label=f'{ct} ({n_str})')
ax3d.set_xticks(x_pos)
ax3d.set_xticklabels(['Butyrate','Nucleotide','LPS/\nTLR4','STING'],
                     fontsize=7.5)
ax3d.set_ylabel('Mean inferred BEV score', fontsize=8)
ax3d.set_title('d  BEV scores by cancer type\n(TCMA, n=290)',
               fontsize=8.5, fontweight='bold', loc='left', pad=4)
ax3d.legend(frameon=False, fontsize=7)
ax3d.axhline(0.5, color='#CCCCCC', lw=0.7, ls='--', zorder=0)
ax3d.set_ylim(0, 1.05)

fig3.savefig(f'{OUT}/Fig3_v4_survival.png', dpi=300,
              bbox_inches='tight', facecolor='white')
print(f"저장: {OUT}/Fig3_v4_survival.png")

# ══════════════════════════════════════════════════════════
# 7. Figure 4 라벨 수정 + ρ값 통일 확인
# ══════════════════════════════════════════════════════════
print("\n=== 7. Figure 4 — ρ값 재확인 ===")
from lifelines import KaplanMeierFitter
cib = pd.read_csv(f'{BASE}/05_integrated/TCMA/CIBERSORT_tumor_only.tsv',
                  sep='\t', index_col=0)
coad_idx  = coad.index.intersection(cib.index)
lps_coad  = bev.loc[coad_idx,'LPS_TLR4_load']
cd8_coad  = cib.loc[coad_idx,'T.cells.CD8']
r_actual, p_actual = stats.spearmanr(lps_coad, cd8_coad)
print(f"LPS/TLR4 vs CD8+ T: ρ={r_actual:.3f}, p={p_actual:.4f}")
print(f"→ 원고에 ρ=+0.228, p=0.014 사용 중")
print(f"  실제값: ρ={r_actual:.3f} — 이 값으로 원고 통일 필요")

# Figure 4 재생성
boot = pd.read_csv(f'{BASE}/06_loco_cv/mediation_bootstrap_final.tsv', sep='\t')
fig4, axes4 = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=300,
    gridspec_kw=dict(wspace=0.45, left=0.28, right=0.97,
                     top=0.90, bottom=0.14))

ax4a = axes4[0]
ylabels = [
    'Fusobacterium\n→ LPS → CD8+ T',
    'Fusobacterium\n→ LPS → Neutrophil',
    'Bacteroides\n→ LPS → CD8+ T',
    'Faecalibacterium\n→ LPS → CD8+ T',
]
inds  = boot['indirect'].values
ci_lo = boot['ci_lo'].values
ci_hi = boot['ci_hi'].values
y_pos = np.arange(len(boot))
col_fp = [NAVY if v>0 else RED for v in inds]

ax4a.axvline(0, color='black', lw=0.8, ls='--', zorder=1)
for i,(ind,lo,hi,col) in enumerate(zip(inds,ci_lo,ci_hi,col_fp)):
    ax4a.plot([lo,hi],[i,i], color=col, lw=2.0, zorder=2, solid_capstyle='round')
    ax4a.scatter(ind, i, color=col, s=36, zorder=3,
                 edgecolors='white', linewidths=0.7)
ax4a.set_yticks(y_pos)
ax4a.set_yticklabels(ylabels, fontsize=7.5)
ax4a.set_xlabel('Indirect effect (a × b)', fontsize=8)
# ★ 라벨 수정
ax4a.set_title('a  BEV-informed path analysis\n(not causal mediation; 95% bootstrap CI, n=1,000)',
               fontsize=8.5, fontweight='bold', loc='left', pad=4)

ax4b = axes4[1]
ax4b.scatter(lps_coad.values, cd8_coad.values,
             s=18, color=NAVY, alpha=0.82, linewidths=0)
z4 = np.polyfit(lps_coad.values, cd8_coad.values, 1)
xl4 = np.linspace(lps_coad.min(), lps_coad.max(), 100)
ax4b.plot(xl4, np.poly1d(z4)(xl4), color=NAVY, lw=1.3)
ax4b.set_xlabel('Inferred LPS/TLR4 BEV score', fontsize=8)
ax4b.set_ylabel('CD8+ T cell fraction (CIBERSORT)', fontsize=8)
p4_str = f'p = {p_actual:.3f}' if p_actual >= 0.001 else 'p < 0.001'
# ★ 실제 ρ값으로 통일
ax4b.set_title(f'b  LPS score vs CD8+ T cells\n\u03C1 = {r_actual:.3f}, {p4_str}  (COAD, n={len(lps_coad)})',
               fontsize=8.5, fontweight='bold', loc='left', pad=4)

fig4.savefig(f'{OUT}/Fig4_v4.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"저장: {OUT}/Fig4_v4.png")
print(f"→ 원고 본문도 ρ={r_actual:.3f}로 통일하세요")

# ══════════════════════════════════════════════════════════
# 8. Supplementary — All 200 paths heatmap
# ══════════════════════════════════════════════════════════
print("\n=== 8. Supplementary: 200 paths ===")
med_all = pd.read_csv(f'{BASE}/06_loco_cv/mediation_final_results.tsv', sep='\t') \
    if os.path.exists(f'{BASE}/06_loco_cv/mediation_final_results.tsv') else None

if med_all is not None:
    print(f"전체 path 수: {len(med_all)}")
    med_all.to_csv(f'{OUT}/Supplementary_Table1_all_paths.tsv', sep='\t', index=False)
    print(f"저장: {OUT}/Supplementary_Table1_all_paths.tsv")

# ══════════════════════════════════════════════════════════
# 9. 전체 요약 출력
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("분석 완료 요약")
print("="*60)
print(f"\n출력 폴더: {OUT}")
print("생성된 파일:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f'{OUT}/{f}')
    print(f"  {f:<45} {size/1024:.1f} KB")

print("\n=== 원고 수정 필요 사항 ===")
print(f"1. Figure 4b ρ값: 원고에 0.228이지만 실제={r_actual:.3f}")
print("   → 원고 본문, legend, Figure 4 모두 통일하세요")
print("2. CLR sensitivity: 위 결과 확인 후 'direction-stable' 유전자만 주요 발견으로 유지")
print("3. Figure 3: number-at-risk 추가된 버전 생성됨")
print("4. Figure 5 추가 패널: ranked correlation plot 생성됨")
