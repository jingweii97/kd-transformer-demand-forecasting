# Walkthrough: Phase 1 M5 Timeline Analysis & Evaluation Protocol Selection

All intermediate and redundant candidate search files have been removed. The **[`id-ood-analysis`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis)** directory now contains exclusively the **4 core deliverables** required for this phase.

No forecasting models were trained, and zero forecasting accuracy metrics (WRMSSE, MAE, MASE) or distribution distance metrics (KS, Wasserstein) were used to select evaluation dates.

---

## Retained Core Deliverables in `id-ood-analysis/`

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **28-Day Timeline Summary** | [`m5_28day_timeline_summary.csv`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis/m5_28day_timeline_summary.csv) | Summary of 69 complete 28-day blocks reporting event days, demand statistics, zero-sales ratios, active SKU ratios, and price dynamics. |
| **Temporal Profile Visualization** | [`m5_temporal_profile.png`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis/m5_temporal_profile.png) | 4-panel publication-quality chronological plot illustrating event-day counts, mean demand, zero-sales/active SKU ratios, and price-change frequency. |
| **Candidate Protocols** | [`candidate_protocols.csv`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis/candidate_protocols.csv) | Structured comparison of 3 candidate chronological protocols for supervisor review. |
| **Analysis Summary Report** | [`protocol_analysis_summary.md`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis/protocol_analysis_summary.md) | Comprehensive report addressing all 6 core research questions. |

---

## Chronological Temporal Profile Visualization

Below is the generated 4-panel chronological visualization:

![M5 Temporal Profile across 69 Non-Overlapping 28-Day Blocks](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis/m5_temporal_profile.png)

---

## Summary of Candidate Protocols in `id-ood-analysis/`

```
Protocol 1 (Recommended Primary Single-Cutoff):
Training (d1–d1829, 1829d) ──> Validation (d1830–d1857, 28d) ──> ID Ref (d1858–d1885, 28d) ──> Ext-Gap OOD (d1886–d1913, 28d) ──> Event OOD (d1914–d1941, 28d)
```

1. **Protocol 1 (Recommended Single-Cutoff)**:
   - $T_{train\_end} = 1829$ (Jan 31, 2016).
   - Validation: `d_1830`–`d_1857` (5 event days).
   - ID Reference: `d_1858`–`d_1885` (3 event days).
   - Extended-Gap OOD: `d_1886`–`d_1913` (0 event days - clean event-free baseline).
   - Event-Intensive OOD: `d_1914`–`d_1941` (4 event days).
2. **Protocol 2 (Standard Kaggle Cutoff)**:
   - $T_{train\_end} = 1857$ (Feb 28, 2016).
   - Single combined OOD target at `d_1914`–`d_1941`.
3. **Protocol 3 (Secondary Rolling-Origin)**:
   - Dual fold ($T_{train\_end} = 730$ for 6-event stress-test `d_731`–`d_758`; $T_{train\_end} = 1829$ for post-training targets).

---

## Verification

The 4 core deliverables are cleanly organized in [`id-ood-analysis/`](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/id-ood-analysis), and all redundant intermediate candidate search files have been removed.
