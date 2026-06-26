# BEV-informed CRC Microbiome Analysis

This repository contains processed data tables, curated reference tables, and analysis code supporting:

> Yu T, Kim J-G, Yoon Y, Choi CW, Jeon W. **BEV-informed intratumoral microbiome profiling nominates candidate host transcriptional programs associated with Fusobacterium abundance in colorectal cancer: a multi-cohort in-silico study.** *Gut Microbes* (submitted).

## Overview

This study integrates three independent public datasets to characterize inferred bacterial extracellular vesicle (BEV) cargo profiles in colorectal cancer and their associations with immune microenvironment, survival, and host transcriptional programs.

**All findings in this study are hypothesis-generating.** BEV cargo scores are computationally inferred from genus-level microbiome abundance and do not represent direct measurements of bacterial extracellular vesicles. No transcriptomic finding survived genome-wide FDR correction; candidate genes were prioritized using a focused candidate-level FDR analysis and validated across multiple sensitivity analyses (see Supplementary Tables).

## Repository structure
## Data sources

| Cohort | Accession | Source |
|---|---|---|
| Iowa rectal cancer | PRJNA1029660 | NCBI SRA |
| TCMA pan-cancer | N/A | Duke Research Data Repository (Dohlman et al., 2022) |
| GSE165255 | GSE165255 | NCBI GEO (Li et al., 2021) |

Raw sequencing and expression data are **not redistributed** here; they remain available from the original public repositories listed above. This repository contains only derived/processed tables and analysis code.

## Supplementary Tables

| Table | Description |
|---|---|
| Table 1 | All 200 BEV-informed path analysis combinations |
| Table 2 | Sensitivity analysis summary (LOO, CLR transformation, antibiotic stratification) |
| Table 3 | Curated 24-genus BEV cargo reference table |
| Table 4 | Full GSEA Hallmark pathway results (50 pathways; 1,000 permutations) |
| Table 5 | Partial correlation analysis (CD8A-adjusted) |
| Table 6 | TIMP3 sensitivity analysis (epithelial/stromal markers) |
| Table 7 | Cohort characteristics |
| Table 8 | Cox regression results (univariable, continuous per SD) |

## Supplementary Data

| File | Description |
|---|---|
| Data 1 | Genome-wide Spearman correlation matrix (Fusobacterium and Faecalibacterium vs 13,297 genes) |

## Code

| Script | Purpose |
|---|---|
| `01_qiime2_16S_processing.sh` | 16S rRNA amplicon processing (QIIME2/DADA2) |
| `02_BEV_cargo_inference.py` | BEV cargo score computation from genus abundance |
| `03_survival_analysis.py` | Kaplan-Meier and Cox regression (TCMA COAD) |
| `04_path_analysis.py` | Bootstrap path analysis (Baron-Kenny framework) |
| `05_transcriptome_integration.py` | Genome-wide Spearman correlation (GSE165255) |
| `06_sensitivity_analyses.py` | CLR transformation, LOO, antibiotic stratification |
| `06b_antibiotic_fix.py` | Antibiotic stratification bug fix and Cox HR calculation |
| `07_GSEA_pathway_analysis.py` | Hallmark pathway enrichment (gseapy) |
| `08_figures.py` | Figure generation (matplotlib) |

## Environment

See `environment.yml` for the full conda environment specification.

```bash
conda env create -f environment.yml
conda activate bev-crc-microbiome
```

QIIME2 2024.10 requires a separate environment for 16S processing (see `environment.yml` for details).

## Important methodological notes

- BEV cargo inference scores are abundance-derived functional annotations, **not** direct measurements of bacterial extracellular vesicles.
- The path analysis mediator is computationally derived from the same genus-level abundance used as the exposure variable; results represent statistical decomposition, not causal mediation.
- No gene survived genome-wide FDR correction in the transcriptomic analysis (n=36). Candidate-level FDR values are reported for prioritization only.
- Genus-level 16S data cannot confirm species identity (e.g., *Fusobacterium nucleatum*) or strain-specific virulence factors (e.g., *fadA*).
- Median-split Kaplan-Meier survival findings did not replicate in continuous Cox regression for two of three BEV features, suggesting possible dichotomization artifact (Royston & Sauerbrei, *Stat Med* 2006).

## License

Code: MIT License. Processed data tables: CC-BY 4.0.

## Contact

Tosol Yu, MD, PhD — Department of Radiation Oncology, Dongnam Institute of Radiological & Medical Sciences (DIRAMS), Busan, Republic of Korea.
tosolyu@naver.com

## Citation

If you use this code or these processed tables, please cite the manuscript above (full citation to be updated upon publication) and this repository (Zenodo DOI: [to be added upon archiving]).
