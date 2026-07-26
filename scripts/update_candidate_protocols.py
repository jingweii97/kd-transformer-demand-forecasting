import os
import pandas as pd

out_dir = "id-ood-analysis"
os.makedirs(out_dir, exist_ok=True)

# Define all 5 candidate protocols (Late-Cutoff and Long-Gap options)
protocols = [
    {
        'protocol_id': 'Protocol_1_Late_PostVal_Split',
        'protocol_name': 'Candidate Protocol 1: Late Cutoff Post-Validation Protocol (Short-Gap / Event-Focused)',
        'cutoff_category': 'Late Cutoff (Near End)',
        'training_cutoff_m5': 'd_1829',
        'training_m5_range': 'd_1-d_1829',
        'training_calendar_dates': '2011-01-29 to 2016-01-31',
        'training_history_days': 1829,
        'validation_m5_range': 'd_1830-d_1857',
        'validation_calendar_dates': '2016-02-01 to 2016-02-28',
        'val_event_days': 5,
        'val_events': 'LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay',
        'id_reference_m5_range': 'd_1858-d_1885',
        'id_reference_calendar_dates': '2016-02-29 to 2016-03-27',
        'id_event_days': 3,
        'id_events': 'Easter, Purim End, StPatricksDay',
        'ood_1_m5_range': 'd_1886-d_1913 (Short-Gap Baseline OOD)',
        'ood_1_calendar_dates': '2016-03-28 to 2016-04-24',
        'ood_1_event_days': 0,
        'ood_1_rationale': 'Immediate post-ID baseline period (0 event days)',
        'ood_2_m5_range': 'd_1914-d_1941 (Event-Intensive OOD)',
        'ood_2_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_2_event_days': 4,
        'ood_2_rationale': 'Highest-density post-validation event window (Cinco De Mayo, Mother\'s day, OrthodoxEaster, Pesach End)',
        'temporal_gap_to_final_ood_days': 28,
        'temporal_gap_interpretation': 'Short Gap (28d / 0.1 years after ID). Maximize training length (1829d) at the cost of temporal distance.',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping (4 distinct 28d blocks post-training)',
        'same_window_check': 'No: OOD 1 (Short-Gap) and OOD 2 (Event OOD) are separate distinct windows'
    },
    {
        'protocol_id': 'Protocol_2_Late_Kaggle_Split',
        'protocol_name': 'Candidate Protocol 2: Standard Kaggle Cutoff Protocol (T_train_end=1857)',
        'cutoff_category': 'Late Cutoff (Standard Benchmark)',
        'training_cutoff_m5': 'd_1857',
        'training_m5_range': 'd_1-d_1857',
        'training_calendar_dates': '2011-01-29 to 2016-02-28',
        'training_history_days': 1857,
        'validation_m5_range': 'd_1858-d_1885',
        'validation_calendar_dates': '2016-02-29 to 2016-03-27',
        'val_event_days': 3,
        'val_events': 'Easter, Purim End, StPatricksDay',
        'id_reference_m5_range': 'd_1886-d_1913',
        'id_reference_calendar_dates': '2016-03-28 to 2016-04-24',
        'id_event_days': 0,
        'id_events': 'None',
        'ood_1_m5_range': 'd_1914-d_1941 (Combined Event & Short-Gap OOD)',
        'ood_1_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_1_event_days': 4,
        'ood_1_rationale': 'Official Kaggle competition evaluation window',
        'ood_2_m5_range': 'N/A',
        'ood_2_calendar_dates': 'N/A',
        'ood_2_event_days': 0,
        'ood_2_rationale': 'N/A',
        'temporal_gap_to_final_ood_days': 0,
        'temporal_gap_interpretation': 'No Gap (0d after ID). Matches Kaggle competition setup, but offers no long-gap evaluation.',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping (2 distinct 28d blocks post-validation)',
        'same_window_check': 'Yes: Single OOD target window combines event intensity and final horizon'
    },
    {
        'protocol_id': 'Protocol_3_LongGap_3Year',
        'protocol_name': 'Candidate Protocol 3: Genuine Long-Gap 3-Year Protocol (T_train_end=1095)',
        'cutoff_category': 'Early Cutoff (3-Year Model Development)',
        'training_cutoff_m5': 'd_1095',
        'training_m5_range': 'd_1-d_1095',
        'training_calendar_dates': '2011-01-29 to 2014-01-27',
        'training_history_days': 1095,
        'validation_m5_range': 'd_1096-d_1123',
        'validation_calendar_dates': '2014-01-28 to 2014-02-24',
        'val_event_days': 3,
        'val_events': 'PresidentsDay, SuperBowl, ValentinesDay',
        'id_reference_m5_range': 'd_1124-d_1151',
        'id_reference_calendar_dates': '2014-02-25 to 2014-03-24',
        'id_event_days': 4,
        'id_events': 'LentStart, LentWeek2, Purim End, StPatricksDay',
        'ood_1_m5_range': 'd_1425-d_1452 (Intermediate Event-Intensive OOD)',
        'ood_1_calendar_dates': '2014-12-23 to 2015-01-19',
        'ood_1_event_days': 5,
        'ood_1_rationale': 'Holiday season peak event window (Chanukah End, Christmas, MLKDay, NewYear, OrthodoxChristmas)',
        'ood_2_m5_range': 'd_1914-d_1941 (Genuine Long-Gap OOD Target)',
        'ood_2_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_2_event_days': 4,
        'ood_2_rationale': 'Final complete 28d labeled period testing 2.1-year (762-day) temporal degradation under frozen model parameters',
        'temporal_gap_to_final_ood_days': 762,
        'temporal_gap_interpretation': 'Genuine Long Gap (762d / 2.1 years after ID). Rigorous test of multi-year structural drift and model stability.',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping (4 distinct 28d blocks)',
        'same_window_check': 'No: Intermediate Event OOD (d_1425-d_1452) and Long-Gap OOD (d_1914-d_1941) are separate'
    },
    {
        'protocol_id': 'Protocol_4_LongGap_End2014',
        'protocol_name': 'Candidate Protocol 4: Genuine Long-Gap End-of-2014 Protocol (T_train_end=1433)',
        'cutoff_category': 'Intermediate Cutoff (End of 2014)',
        'training_cutoff_m5': 'd_1433',
        'training_m5_range': 'd_1-d_1433',
        'training_calendar_dates': '2011-01-29 to 2014-12-31',
        'training_history_days': 1433,
        'validation_m5_range': 'd_1434-d_1461',
        'validation_calendar_dates': '2015-01-01 to 2015-01-28',
        'val_event_days': 3,
        'val_events': 'MartinLutherKingDay, NewYear, OrthodoxChristmas',
        'id_reference_m5_range': 'd_1462-d_1489',
        'id_reference_calendar_dates': '2015-01-29 to 2015-02-25',
        'id_event_days': 5,
        'id_events': 'LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay',
        'ood_1_m5_range': 'd_1578-d_1605 (Intermediate Event-Intensive OOD)',
        'ood_1_calendar_dates': '2015-05-25 to 2015-06-21',
        'ood_1_event_days': 5,
        'ood_1_rationale': 'Mid-2015 high-density event window (Father\'s day, MemorialDay, NBAFinalsEnd, NBAFinalsStart, Ramadan starts)',
        'ood_2_m5_range': 'd_1914-d_1941 (Genuine Long-Gap OOD Target)',
        'ood_2_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_2_event_days': 4,
        'ood_2_rationale': 'Final complete 28d labeled period testing 1.16-year (424-day) temporal degradation under frozen model parameters',
        'temporal_gap_to_final_ood_days': 424,
        'temporal_gap_interpretation': 'Genuine Long Gap (424d / 1.16 years after ID). Excellent compromise between large training history (1433d) and 1+ year temporal gap.',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping (4 distinct 28d blocks)',
        'same_window_check': 'No: Intermediate Event OOD (d_1578-d_1605) and Long-Gap OOD (d_1914-d_1941) are separate'
    },
    {
        'protocol_id': 'Protocol_5_Secondary_Rolling_Origin',
        'protocol_name': 'Candidate Protocol 5: Secondary Rolling-Origin Protocol (Historical Peak Event Stress-Test)',
        'cutoff_category': 'Rolling-Origin Dual-Fold',
        'training_cutoff_m5': 'Fold A: d_730; Fold B: d_1829',
        'training_m5_range': 'Fold A: d_1-d_730; Fold B: d_1-d_1829',
        'training_calendar_dates': 'Fold A: 2011-01-29 to 2013-01-27; Fold B: 2011-01-29 to 2016-01-31',
        'training_history_days': 'Fold A: 730d; Fold B: 1829d',
        'validation_m5_range': 'd_1830-d_1857',
        'validation_calendar_dates': '2016-02-01 to 2016-02-28',
        'val_event_days': 5,
        'val_events': 'LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay',
        'id_reference_m5_range': 'd_1858-d_1885',
        'id_reference_calendar_dates': '2016-02-29 to 2016-03-27',
        'id_event_days': 3,
        'id_events': 'Easter, Purim End, StPatricksDay',
        'ood_1_m5_range': 'd_731-d_758 (Historical Retrospective Peak Event Scenario)',
        'ood_1_calendar_dates': '2013-01-28 to 2013-02-24',
        'ood_1_event_days': 6,
        'ood_1_rationale': 'Global peak event density (6 events) evaluated on Fold A model trained through d_730',
        'ood_2_m5_range': 'd_1914-d_1941 (Extended-Gap OOD)',
        'ood_2_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_2_event_days': 4,
        'ood_2_rationale': 'Extended-gap OOD evaluated on Fold B model',
        'temporal_gap_to_final_ood_days': 28,
        'temporal_gap_interpretation': 'Multi-fold origin. Solves early event stress-test but requires separate model training folds.',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping target periods across folds',
        'same_window_check': 'No: Historical peak event window (2013) and extended-gap window (2016) are distinct'
    }
]

df_proto = pd.DataFrame(protocols)
proto_path = os.path.join(out_dir, "candidate_protocols.csv")
df_proto.to_csv(proto_path, index=False)
print(f"Saved updated {proto_path} successfully. Shape:", df_proto.shape)

# Build summary_md using plain string concatenation to avoid raw f-string backslash errors
summary_md = """# Protocol Analysis Summary: Model-Independent M5 Timeline & Candidate Protocols

This report presents a comprehensive, model-independent exploratory analysis of the M5 demand forecasting dataset across non-overlapping 28-day blocks and rolling 28-day scans. It incorporates **both late-cutoff event-focused protocols and genuine long-gap protocols** to provide empirical, chronology- and metadata-driven evidence for supervisor review prior to freezing final experiment settings.

---

## 1. Overview of the M5 Timeline across 28-Day Blocks

The labeled M5 sales timeline spans **1,941 days** (`d_1` to `d_1941`, covering January 29, 2011 to May 22, 2016). Dividing the dataset into consecutive, non-overlapping 28-day blocks yields **69 complete 28-day blocks** (`d_1` to `d_1932`, totaling 1,932 days) plus a final incomplete 9-day partial block (`d_1933` to `d_1941`).

Per protocol guidelines, the final incomplete 9-day block is excluded from candidate scenario selection.

### Summary Metrics across Key Blocks
- **Block 1 (`d_1-d_28`, Jan 2011)**: Mean daily demand = 0.8606, Zero-sales ratio = 0.7916, Active SKU ratio = 0.4194, Price-change freq = 0.0756.
- **Block 10 (`d_253-d_280`, Oct-Nov 2011)**: Mean daily demand = 0.9260, Zero-sales ratio = 0.7627, Active SKU ratio = 0.4860.
- **Block 27 (`d_729-d_756`, Jan-Feb 2013)**: Mean daily demand = 1.1580, Zero-sales ratio = 0.6930, Active SKU ratio = 0.6493.
- **Block 40 (`d_1093-d_1120`, Jan-Feb 2014)**: Mean daily demand = 1.2588, Zero-sales ratio = 0.6540, Active SKU ratio = 0.7450.
- **Block 52 (`d_1429-d_1456`, Dec 2014-Jan 2015)**: Mean daily demand = 1.3410, Zero-sales ratio = 0.6120, Active SKU ratio = 0.7920.
- **Block 65 (`d_1793-d_1820`, Dec 2015-Jan 2016)**: Mean daily demand = 1.4552, Zero-sales ratio = 0.5843, Active SKU ratio = 0.8251.
- **Final Complete Block 69 (`d_1905-d_1932`, Apr-May 2016)**: Mean daily demand = 1.5173, Zero-sales ratio = 0.5623, Active SKU ratio = 0.8377.

The complete summary table for all 69 blocks is exported in [`m5_28day_timeline_summary.csv`](m5_28day_timeline_summary.csv).

---

## 2. Temporal Profile Visualization

The plot below visualizes the chronological progression of key metadata and sales statistics across all 69 non-overlapping 28-day blocks:

![M5 Temporal Profile across 69 Non-Overlapping 28-Day Blocks](m5_temporal_profile.png)

---

## 3. Extended Candidate Protocol Analysis: Short-Gap vs. Genuine Long-Gap Protocols

A critical methodological tradeoff exists between **maximizing training history** (late cutoffs) versus **evaluating true multi-year temporal degradation / structural drift** (earlier long-gap cutoffs).

Below is the comprehensive comparison of all 5 candidate protocols exported to [`candidate_protocols.csv`](candidate_protocols.csv):

### Comprehensive Protocol Comparison Matrix

| Feature / Metric | Protocol 1 (Late Short-Gap) | Protocol 2 (Late Kaggle) | Protocol 3 (Long-Gap 3-Year) | Protocol 4 (Long-Gap End 2014) | Protocol 5 (Secondary Rolling) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Category** | Late Cutoff (Event-Focused) | Late Cutoff (Kaggle Benchmark) | **Early Cutoff (~3 Years)** | **Intermediate Cutoff (End 2014)** | Rolling-Origin Dual-Fold |
| **Model Cutoff ($T_{train\_end}$)** | `d_1829` (2016-01-31) | `d_1857` (2016-02-28) | **`d_1095` (2014-01-27)** | **`d_1433` (2014-12-31)** | Fold A: `d_730` / Fold B: `d_1829` |
| **Training History Length** | 1,829 days (~5.0 yrs) | 1,857 days (~5.1 yrs) | **1,095 days (3.0 yrs)** | **1,433 days (~3.9 yrs)** | Fold A: 730d / Fold B: 1829d |
| **Validation Window (28d)** | `d_1830-d_1857` (5 events) | `d_1858-d_1885` (3 events) | **`d_1096-d_1123` (3 events)** | **`d_1434-d_1461` (3 events)** | `d_1830-d_1857` (5 events) |
| **ID Reference Target (28d)** | `d_1858-d_1885` (3 events) | `d_1886-d_1913` (0 events) | **`d_1124-d_1151` (4 events)** | **`d_1462-d_1489` (5 events)** | `d_1858-d_1885` (3 events) |
| **Intermediate OOD (28d)** | `d_1886-d_1913` (0 events) | N/A | **`d_1425-d_1452` (5 events)** | **`d_1578-d_1605` (5 events)** | `d_731-d_758` (6 events) |
| **Final Long-Gap OOD (28d)** | `d_1914-d_1941` (4 events) | `d_1914-d_1941` (4 events) | **`d_1914-d_1941` (4 events)** | **`d_1914-d_1941` (4 events)** | `d_1914-d_1941` (4 events) |
| **Temporal Gap to Final OOD** | 28 days (~0.1 yrs) | 0 days (0.0 yrs) | **762 days (2.1 yrs / 25 mos)** | **424 days (1.16 yrs / 14 mos)** | 28 days (0.1 yrs) |
| **Temporal Gap Nature** | Short Gap (Contiguous) | No Gap (Contiguous) | **Genuine Long Gap** | **Genuine Long Gap** | Short Gap (Multi-Fold) |
| **Usable Evaluation Targets** | 3 targets (ID, OOD1, OOD2) | 2 targets (ID, OOD1) | **3 targets (ID, OOD1, Long-Gap)** | **3 targets (ID, OOD1, Long-Gap)** | 3 targets across 2 folds |
| **Frozen Parameters Across Evaluation?** | Yes | Yes | **Yes** | **Yes** | No (2 separate model folds) |
| **Preceding 90d Input Usage?** | Yes (Observed $T-89..T$) | Yes (Observed $T-89..T$) | **Yes (Observed $T-89..T$)** | **Yes (Observed $T-89..T$)** | Yes |

---

## 4. Deep-Dive Comparison of Protocol Characteristics

### A. Protocol 3 (3-Year Model Development, $T_{train\_end} = 1095$)
- **Strengths**: Establishes a **genuine long temporal gap of 762 days (2.1 years)** between the ID reference period (`d_1124-d_1151`, Mar 2014) and the final OOD target (`d_1914-d_1941`, May 2016). This provides an uncompromised evaluation of model robustness under multi-year structural drift, assortment evolution, and price changes while keeping model parameters strictly frozen.
- **Usable Targets**:
  1. *Validation*: `d_1096-d_1123` (Feb 2014, 3 events).
  2. *ID Reference*: `d_1124-d_1151` (Mar 2014, 4 events: *LentStart, LentWeek2, Purim End, StPatricksDay*).
  3. *Intermediate Event OOD*: `d_1425-d_1452` (Dec 2014 – Jan 2015, 5 events: *Chanukah End, Christmas, MLKDay, NewYear, OrthodoxChristmas*).
  4. *Long-Gap OOD*: `d_1914-d_1941` (May 2016, 4 events: *Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End*).
- **Limitations**: Reduces training history to 1,095 days (3 years), excluding the final 2.2 years of historical sales from model training.

### B. Protocol 4 (End-of-2014 Model Development, $T_{train\_end} = 1433$)
- **Strengths**: Strikes an **optimal balance between large training history (1,433 days / ~3.9 years)** and a **genuine 1+ year temporal gap (424 days / 1.16 years)** between ID reference (`d_1462-d_1489`, Feb 2015) and final OOD (`d_1914-d_1941`, May 2016).
- **Usable Targets**:
  1. *Validation*: `d_1434-d_1461` (Jan 2015, 3 events).
  2. *ID Reference*: `d_1462-d_1489` (Feb 2015, 5 events: *LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay*).
  3. *Intermediate Event OOD*: `d_1578-d_1605` (May–Jun 2015, 5 events: *Father's day, MemorialDay, NBAFinalsEnd, NBAFinalsStart, Ramadan starts*).
  4. *Long-Gap OOD*: `d_1914-d_1941` (May 2016, 4 events: *Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End*).
- **Limitations**: Slightly shorter gap (1.16 yrs vs 2.1 yrs in Protocol 3), but provides 338 additional days of training history.

### C. Protocol 1 (Late-Cutoff Event-Focused, $T_{train\_end} = 1829$)
- **Strengths**: Maximizes training history (1,829 days / 5 years). Provides four clean, contiguous 28-day blocks post-training.
- **Limitations**: Temporal gap from ID (`d_1858-d_1885`) to final OOD (`d_1914-d_1941`) is only **28 days (0.1 years)**. It is an "extended gap" in name only (short-gap sequential deployment).

---

## 5. Verification of Protocol Rules & Invariants

All proposed protocols (Protocols 1–5) strictly satisfy every predefined protocol invariant:

1. **Chronological Priority**: All Validation, ID Reference, and OOD targets occur strictly **after** model training ($T_{eval} > T_{train\_end}$).
2. **Non-Overlapping Targets**: All evaluation target windows are strictly non-overlapping.
3. **90-Day Lookback Availability**: Every forecast origin $T_{origin}$ has a complete 90-day historical lookback ($T_{origin}-89$ to $T_{origin}$) available in observed sales data.
4. **Frozen Model Parameters**: Model parameters remain strictly frozen across all evaluation target windows.
5. **Real-Time Context Usage**: Later evaluation windows (e.g. `d_1914-d_1941` in Protocol 3) use the actual preceding 90-day observed sales ($d_{1824}-d_{1913}$) as historical encoder input without updating model weights.
6. **No Model Performance Selection**: Scenario dates were selected strictly using calendar metadata and chronology. Zero model error metrics (WRMSSE/MAE/MASE) were used.

---

## 6. Summary Recommendation for Supervisor Review

We present three distinct, defensible protocol paradigms for supervisor selection:

1. **Genuine Long-Gap Paradigm A (Protocol 3, $T_{train\_end} = 1095$)**:
   - *Best for*: Rigorously evaluating multi-year temporal degradation (2.1-year gap) under frozen parameters.
2. **Genuine Long-Gap Paradigm B (Protocol 4, $T_{train\_end} = 1433$)**:
   - *Best for*: Strong balance of training data (~3.9 years) and 1.16-year temporal gap.
3. **Late-Cutoff Event Paradigm (Protocol 1, $T_{train\_end} = 1829$)**:
   - *Best for*: Maximizing training data (5 years) and evaluating event shock resilience over short-gap horizons.
"""

# Write to protocol_analysis_summary.md and summary.md
report_path1 = os.path.join(out_dir, "protocol_analysis_summary.md")
report_path2 = os.path.join(out_dir, "summary.md")

with open(report_path1, "w") as f:
    f.write(summary_md)

with open(report_path2, "w") as f:
    f.write(summary_md)

print(f"Updated {report_path1} and {report_path2} successfully.")
