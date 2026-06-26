import pandas as pd
import numpy as np
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

BASE = '/home/tosolyu/larc_microbiome'
OUT  = f'{BASE}/06_loco_cv/step2_results'

s16_raw = pd.read_csv(f'{BASE}/05_integrated/GSE165255/genus_16S.txt', sep='\t', index_col=0)
rna_raw = pd.read_csv(f'{BASE}/05_integrated/GSE165255/rna_normalized.txt', sep='\t', index_col=0)
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

meta = pd.read_csv(f'{BASE}/05_integrated/GSE165255/metadata.csv')
abx_col = 'Antibiotics'

meta_tumor = meta[meta['Type2']=='Tumor'].copy()
meta_tumor['patient_id'] = meta_tumor['SubmissionID'].str.replace('-Tumor','').str.strip()
meta_sub = meta_tumor[['patient_id', abx_col]].dropna().drop_duplicates('patient_id').set_index('patient_id')
# ★ 버그 수정: Series 중복 인덱스 제거 후 scalar로 접근
meta_sub = meta_sub[~meta_sub.index.duplicated(keep='first')]

common_abx = [p for p in common if p in meta_sub.index]
print(f"항생제 정보 있는 환자: {len(common_abx)}")

key_genes = {
    'IFNGR1': 'ENSG00000027697', 'JAK1': 'ENSG00000162434',
    'TCF7L2': 'ENSG00000148737', 'TIMP3': 'ENSG00000100234',
    'IRAK2':  'ENSG00000134070',
}

abx_results = []
# 그룹별 분석
for group in ['No','Yes']:
    # ★ 수정: scalar 비교
    pts_g = [p for p in common_abx
             if meta_sub.loc[p, abx_col] == group]
    print(f"\n[항생제 {group}군, n={len(pts_g)}]")
    if len(pts_g) < 5:
        print("  샘플 수 부족")
        continue
    for sym, eid in key_genes.items():
        if eid not in rna_m.index: continue
        expr_g = rna_m.loc[eid, pts_g]
        genus_g = fn.loc[pts_g] if sym != 'IRAK2' else fc.loc[pts_g]
        r, p = stats.spearmanr(genus_g.values, expr_g.values)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {sym:<10}: ρ={r:+.3f}, p={p:.3f} {sig}")
        abx_results.append({'abx_group':group,'n':len(pts_g),
                             'gene':sym,'rho':r,'p':p})

# 전체 vs 서브그룹 비교
print("\n=== 전체 vs 서브그룹 방향 일치 확인 ===")
full_res = {}
for sym, eid in key_genes.items():
    if eid not in rna_m.index: continue
    g = fn if sym != 'IRAK2' else fc
    r, p = stats.spearmanr(g.values, rna_m.loc[eid].values)
    full_res[sym] = r
    print(f"  전체 (n={len(common)}) {sym}: ρ={r:+.3f}")

df_abx = pd.DataFrame(abx_results)
if not df_abx.empty:
    df_abx.to_csv(f'{OUT}/antibiotic_stratified.tsv', sep='\t', index=False)
    print(f"\n저장: {OUT}/antibiotic_stratified.tsv")
    
    # 방향 일치 확인
    print("\n방향 일치:")
    for sym in key_genes:
        full_r = full_res.get(sym, 0)
        for _, row in df_abx[df_abx['gene']==sym].iterrows():
            same = '✅' if full_r * row['rho'] > 0 else '❌'
            print(f"  {same} {sym} [{row['abx_group']}]: ρ={row['rho']:+.3f} (전체={full_r:+.3f})")

# Cox HR 계산
print("\n=== Cox HR (Continuous per SD) ===")
bev = pd.read_csv(f'{BASE}/06_loco_cv/TCMA_BEV_survival.tsv', sep='\t', index_col=0)
bev = bev[bev['OS_days'].notna() & (bev['OS_days']>0)].copy()
coad = bev[bev['acronym']=='COAD'].copy()

try:
    from lifelines import CoxPHFitter
    feats = [('butyrate_producer','Butyrate'),
             ('nucleotide_biosynthesis','Nucleotide'),
             ('LPS_TLR4_load','LPS/TLR4')]
    cox_rows = []
    for feat, label in feats:
        sub = coad[['OS_days','OS_event',feat]].dropna().copy()
        sub[feat+'_std'] = (sub[feat]-sub[feat].mean())/sub[feat].std()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub[['OS_days','OS_event',feat+'_std']],
                duration_col='OS_days', event_col='OS_event')
        s   = cph.summary
        hr  = float(np.exp(s.loc[feat+'_std','coef']))
        lci = float(np.exp(s.loc[feat+'_std','coef lower 95%']))
        uci = float(np.exp(s.loc[feat+'_std','coef upper 95%']))
        pv  = float(s.loc[feat+'_std','p'])
        print(f"  {label:<20} HR={hr:.3f} (95%CI: {lci:.3f}–{uci:.3f}), p={pv:.4f}")
        cox_rows.append({'feature':label,'HR':hr,'CI_lo':lci,'CI_hi':uci,'p':pv})
    pd.DataFrame(cox_rows).to_csv(f'{OUT}/cox_results.tsv', sep='\t', index=False)
    print(f"저장: {OUT}/cox_results.tsv")
except Exception as e:
    print(f"Cox 오류: {e}")

# LPS vs CD8 실제 ρ 확인
print("\n=== LPS vs CD8 실제 ρ 확인 ===")
from scipy.stats import spearmanr
try:
    cib = pd.read_csv(f'{BASE}/05_integrated/TCMA/CIBERSORT_tumor_only.tsv',
                      sep='\t', index_col=0)
    coad_idx = coad.index.intersection(cib.index)
    lps = bev.loc[coad_idx,'LPS_TLR4_load']
    cd8 = cib.loc[coad_idx,'T.cells.CD8']
    r, p = spearmanr(lps.values, cd8.values)
    print(f"  LPS/TLR4 vs CD8+: ρ={r:.3f}, p={p:.4f}")
    print(f"  → 원고에 이 값으로 통일하세요")
except Exception as e:
    print(f"오류: {e}")

print("\n완료")
