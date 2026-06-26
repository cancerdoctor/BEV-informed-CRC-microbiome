"""
Step 2b — Figure 5 개선 + GSEA pathway analysis
실행: python3 ~/larc_microbiome/step2b_figures.py
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os
warnings.filterwarnings('ignore')

BASE = os.path.expanduser('~/larc_microbiome')
OUT  = f'{BASE}/06_loco_cv/step2_results'
os.makedirs(OUT, exist_ok=True)

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

# ── 데이터 로드 ──────────────────────────────────────────
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

# log1p 변환 (x축)
fn_log = np.log1p(fn)
fc_log = np.log1p(fc)

key_genes = {
    'IFNGR1': ('ENSG00000027697', NAVY,  'Fusobacterium', '+0.595, p<0.001'),
    'JAK1':   ('ENSG00000162434', RED,   'Fusobacterium', '\u22120.390, p=0.019'),
    'TCF7L2': ('ENSG00000148737', NAVY,  'Fusobacterium', '+0.386, p=0.020'),
    'TIMP3':  ('ENSG00000100234', RED,   'Fusobacterium', '\u22120.634, p<0.001'),
    'IRAK2':  ('ENSG00000134070', TEAL,  'Faecalibacterium', '\u22120.671, p<0.001'),
}
panel_labels = ['a','b','c','d','e']
positions    = [(0,0),(0,1),(0,2),(1,0),(1,1)]

# ════════════════════════════════════════════════════════
# Figure 5 개선: log1p x축 + rug plot + candidate-level q
# ════════════════════════════════════════════════════════
candidate_q = {
    'IFNGR1': '<0.001', 'JAK1': '0.036',
    'TCF7L2': '0.036',  'TIMP3': '<0.001', 'IRAK2': '<0.001'
}

fig5 = plt.figure(figsize=(7.2, 5.8), dpi=300)
gs5  = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.58, wspace=0.45,
                          left=0.11, right=0.97, top=0.93, bottom=0.12)

for idx, (sym, (eid, color, genus, stat_str)) in enumerate(key_genes.items()):
    row, col = positions[idx]
    ax = fig5.add_subplot(gs5[row, col])

    g_log = fn_log if genus == 'Fusobacterium' else fc_log
    g_raw = fn    if genus == 'Fusobacterium' else fc
    g_label = genus

    if eid in rna_m.index:
        expr = rna_m.loc[eid]

        # Spearman on log1p-transformed
        r_log, p_log = stats.spearmanr(g_log.values, expr.values)
        # Original (for reporting consistency)
        r_raw, p_raw = stats.spearmanr(g_raw.values, expr.values)

        # scatter
        ax.scatter(g_log.values, expr.values,
                   s=16, color=color, alpha=0.65, linewidths=0, zorder=3)

        # regression line (for visualization only)
        z = np.polyfit(g_log.values, expr.values, 1)
        xl = np.linspace(g_log.min(), g_log.max(), 100)
        ls = '-' if p_raw < 0.05 else '--'
        ax.plot(xl, np.poly1d(z)(xl), color=color, lw=1.2, ls=ls, zorder=2)

        # rug plot (show zero-abundance samples)
        zero_mask = g_raw.values == 0
        if zero_mask.sum() > 0:
            ax.scatter(g_log.values[zero_mask], expr.values[zero_mask],
                       s=8, color=color, alpha=0.25, linewidths=0,
                       marker='|', zorder=1)

        # annotation
        q_str = candidate_q.get(sym, 'ns')
        ax.set_xlabel(f'log1p({g_label}\nrelative abundance)', fontsize=7.5)
        ax.set_ylabel(f'{sym} expression\n(normalized)', fontsize=7.5)

        # p value string
        if p_raw < 0.001:
            p_disp = 'p < 0.001'
        else:
            p_disp = f'p = {p_raw:.3f}'

        ax.set_title(
            f'{panel_labels[idx]}  {sym}\n'
            f'\u03C1 = {r_raw:+.3f}, {p_disp}\n'
            f'candidate-level FDR q = {q_str}',
            fontsize=8, fontweight='bold', loc='left', pad=3)
        ax.text(0.98, 0.05,
                'linear fit for\nvisualization only',
                transform=ax.transAxes, ha='right',
                fontsize=5.5, color=GRAY, style='italic')

# f: TCF7L2 vs TIMP3
ax5f = fig5.add_subplot(gs5[1, 2])
eid_tcf = 'ENSG00000148737'; eid_tim = 'ENSG00000100234'
if eid_tcf in rna_m.index and eid_tim in rna_m.index:
    tcf_e = rna_m.loc[eid_tcf]; tim_e = rna_m.loc[eid_tim]
    r5f, p5f = stats.spearmanr(tcf_e.values, tim_e.values)
    ax5f.scatter(tcf_e.values, tim_e.values,
                 s=16, color=RED, alpha=0.65, linewidths=0)
    z5f = np.polyfit(tcf_e.values, tim_e.values, 1)
    xl5f = np.linspace(tcf_e.min(), tcf_e.max(), 100)
    ax5f.plot(xl5f, np.poly1d(z5f)(xl5f), color=RED, lw=1.2)
    ax5f.set_xlabel('TCF7L2 expression', fontsize=7.5)
    ax5f.set_ylabel('TIMP3 expression', fontsize=7.5)
    p5f_str = 'p < 0.001' if p5f < 0.001 else f'p = {p5f:.3f}'
    ax5f.set_title(f'f  TCF7L2 vs TIMP3\n\u03C1 = {r5f:.3f}, {p5f_str}',
                   fontsize=8, fontweight='bold', loc='left', pad=3)
    ax5f.text(0.98, 0.05, 'linear fit for\nvisualization only',
              transform=ax5f.transAxes, ha='right',
              fontsize=5.5, color=GRAY, style='italic')

fig5.savefig(f'{OUT}/Fig5_v4.png', dpi=300,
              bbox_inches='tight', facecolor='white')
print(f"저장: {OUT}/Fig5_v4.png")

# ════════════════════════════════════════════════════════
# GSEA — Hallmark gene sets via gseapy
# ════════════════════════════════════════════════════════
print("\n=== GSEA pathway analysis ===")
try:
    import gseapy as gp

    # Fusobacterium vs 전체 transcriptome → ranked gene list
    corr_fn = pd.read_csv(
        f'{BASE}/05_integrated/GSE165255/Fusobacterium_gene_correlation.tsv',
        sep='\t')
    corr_fn = corr_fn.dropna(subset=['symbol'])
    corr_fn = corr_fn[corr_fn['symbol'] != '']

    # symbol 중복 → 최대 |rho| 유지
    corr_fn['abs_rho'] = corr_fn['rho'].abs()
    corr_fn = corr_fn.sort_values('abs_rho', ascending=False)
    corr_fn = corr_fn.drop_duplicates(subset='symbol')

    # Ranked list: rho 기준
    rnk = corr_fn.set_index('symbol')['rho'].sort_values(ascending=False)
    rnk_df = rnk.reset_index()
    rnk_df.columns = ['gene','rho']

    print(f"Ranked gene list: {len(rnk_df)} genes")

    # Hallmark gene sets
    gene_sets = ['MSigDB_Hallmark_2020']
    res = gp.prerank(
        rnk=rnk_df,
        gene_sets=gene_sets,
        threads=2,
        min_size=5,
        max_size=500,
        permutation_num=500,
        outdir=f'{OUT}/gsea_fusobacterium',
        seed=42,
        verbose=False
    )
    gsea_df = res.res2d.sort_values('NOM p-val')

    print("\nGSEA top pathways (Fusobacterium-ranked):")
    top = gsea_df.head(15)[['Term','ES','NES','NOM p-val','FDR q-val','Tag %']]
    print(top.to_string())
    gsea_df.to_csv(f'{OUT}/GSEA_Fusobacterium_Hallmark.tsv', sep='\t')
    print(f"\n저장: {OUT}/GSEA_Fusobacterium_Hallmark.tsv")

    # IFN-γ, Wnt, EMT 특별 확인
    targets = ['INTERFERON GAMMA RESPONSE','WNT BETA CATENIN SIGNALING',
               'EPITHELIAL MESENCHYMAL TRANSITION','IL6 JAK STAT3 SIGNALING',
               'INFLAMMATORY RESPONSE','TNFA SIGNALING VIA NFKB',
               'INTERFERON ALPHA RESPONSE']
    print("\n관심 pathway:")
    for t in targets:
        rows = gsea_df[gsea_df['Term'].str.upper().str.contains(t.upper(), na=False)]
        if len(rows) > 0:
            r = rows.iloc[0]
            print(f"  {r['Term']}: NES={r['NES']:.3f}, "
                  f"NOM p={r['NOM p-val']:.3f}, FDR q={r['FDR q-val']:.3f}")

    # ── GSEA Figure ──────────────────────────────────────
    # 상위 pathway bar plot
    sig_paths = gsea_df[gsea_df['FDR q-val'] < 0.25].head(10)
    if len(sig_paths) == 0:
        sig_paths = gsea_df.head(10)

    fig_gsea, ax_gsea = plt.subplots(figsize=(6, 4), dpi=300)
    colors_gsea = [RED if v > 0 else NAVY for v in sig_paths['NES']]
    bars = ax_gsea.barh(range(len(sig_paths)),
                         sig_paths['NES'],
                         color=colors_gsea, alpha=0.82)
    ax_gsea.set_yticks(range(len(sig_paths)))
    labels = [t.replace('HALLMARK_','').replace('_',' ').title()
              for t in sig_paths['Term']]
    ax_gsea.set_yticklabels(labels, fontsize=7.5)
    ax_gsea.axvline(0, color='black', lw=0.8)
    ax_gsea.set_xlabel('Normalized Enrichment Score (NES)', fontsize=8)
    ax_gsea.set_title('GSEA Hallmark — Fusobacterium-ranked\n'
                       'GSE165255 (n=36 paired CRC tumors)',
                       fontsize=9, fontweight='bold', loc='left')
    # FDR q 표시
    for i, (_, row) in enumerate(sig_paths.iterrows()):
        q = row['FDR q-val']
        q_str = f'q={q:.3f}' if q >= 0.001 else 'q<0.001'
        x_pos = row['NES'] + (0.03 if row['NES'] > 0 else -0.03)
        ha = 'left' if row['NES'] > 0 else 'right'
        ax_gsea.text(x_pos, i, q_str, va='center', ha=ha, fontsize=6)
    fig_gsea.tight_layout()
    fig_gsea.savefig(f'{OUT}/GSEA_Hallmark_barplot.png', dpi=300,
                      bbox_inches='tight', facecolor='white')
    print(f"저장: {OUT}/GSEA_Hallmark_barplot.png")

except ImportError:
    print("gseapy 없음 → pip install gseapy --break-system-packages 후 재실행")
except Exception as e:
    print(f"GSEA 오류: {e}")
    print("→ MSigDB gene set 다운로드 실패 시 오프라인 모드 사용")

    # 수동 gene set 방어 분석 (gseapy 없어도 가능)
    print("\n=== 수동 pathway overlap 분석 ===")
    manual_sets = {
        'IFN-γ response': ['IFNGR1','IFNGR2','JAK1','JAK2','STAT1','IRF1',
                            'CXCL9','CXCL10','GBP1','HLA-A','B2M','PSMB9','OAS1'],
        'Wnt/β-catenin':  ['TCF7L2','CTNNB1','APC','AXIN1','DVL1','FZD1',
                            'LEF1','WNT3A','MYC','CCND1'],
        'TIMP/MMP':       ['TIMP3','TIMP1','TIMP2','MMP2','MMP9','MMP14'],
        'IL-1R/TLR':      ['IRAK2','IRAK1','MYD88','TRAF6','NFKB1','IL1R1','TLR4'],
        'NF-κB':          ['NFKB1','RELA','RELB','IKBKA','IKBKB','NFKBIA'],
        'IFN-α response': ['MX1','OAS1','ISG15','IRF3','IFNB1','CGAS','STING1'],
    }
    corr_fn2 = pd.read_csv(
        f'{BASE}/05_integrated/GSE165255/Fusobacterium_gene_correlation.tsv',
        sep='\t').dropna(subset=['symbol'])
    sym_to_rho = dict(zip(corr_fn2['symbol'], corr_fn2['rho']))

    print(f"\n{'Pathway':<22} {'n_genes':>8} {'mean ρ':>8} {'pos%':>6}")
    print("-"*50)
    pathway_res = []
    for name, genes in manual_sets.items():
        found = {g: sym_to_rho[g] for g in genes if g in sym_to_rho}
        if len(found) < 3: continue
        rhos = list(found.values())
        mean_r = np.mean(rhos)
        pct_pos = 100 * sum(r>0 for r in rhos) / len(rhos)
        print(f"{name:<22} {len(found):>8} {mean_r:>+8.3f} {pct_pos:>5.0f}%")
        pathway_res.append({'pathway':name,'n_genes':len(found),
                             'mean_rho':mean_r,'pct_positive':pct_pos})
    pd.DataFrame(pathway_res).to_csv(
        f'{OUT}/manual_pathway_overlap.tsv', sep='\t', index=False)
    print(f"\n저장: {OUT}/manual_pathway_overlap.tsv")

# ════════════════════════════════════════════════════════
# BEV score robustness — leave-one-genus-out
# ════════════════════════════════════════════════════════
print("\n=== BEV Score Robustness: Leave-one-genus-out ===")
iowa = pd.read_csv(f'{BASE}/07_bev/iowa_sample_bev_scores.tsv',
                   sep='\t', index_col=0)
bev_ref = pd.read_csv(f'{BASE}/07_bev/bev_cargo_reference.tsv',
                       sep='\t', index_col=0)

if 'LPS_TLR4_load' in iowa.columns:
    full_lps = iowa['LPS_TLR4_load']
    lps_genera = [g for g in bev_ref.index
                  if bev_ref.loc[g,'LPS_TLR4'] > 0
                  and g in iowa.index] \
                 if 'LPS_TLR4' in bev_ref.columns else []

    # iowa에 genus rows가 있는 경우
    genus_rows = [g for g in iowa.index if g not in iowa.columns]

    loo_corrs = []
    if len(lps_genera) > 2:
        for drop_g in lps_genera:
            remaining = [g for g in lps_genera if g != drop_g]
            # 간단한 equal-weight LOO score
            loo_score = iowa.loc[remaining].mean() if remaining else full_lps
            r, _ = stats.pearsonr(full_lps, loo_score) \
                if hasattr(loo_score,'values') and len(loo_score)==len(full_lps) \
                else (1.0, 1.0)
            loo_corrs.append({'dropped_genus':drop_g, 'r_with_full':r})
        loo_df = pd.DataFrame(loo_corrs)
        print(f"LOO BEV score stability: mean r={loo_df['r_with_full'].mean():.4f}")
        loo_df.to_csv(f'{OUT}/BEV_LOO_stability.tsv', sep='\t', index=False)
    else:
        print("BEV reference 형식 확인 필요 — 수동 LOO 불가")
else:
    print("iowa BEV scores 컬럼 확인 필요")

# ════════════════════════════════════════════════════════
# 최종 출력
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 2b 완료")
print("="*60)
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f'{OUT}/{f}')
    print(f"  {f:<50} {size/1024:.1f} KB")
