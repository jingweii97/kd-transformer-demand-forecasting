import os
import pandas as pd

out_dir = "id-ood-analysis"
os.makedirs(out_dir, exist_ok=True)

# Define updated candidate protocols focusing on the simplified 1-OOD protocol (Training -> Val -> ID -> Long-Gap Event OOD)
protocols = [
    {
        'protocol_id': 'Candidate_1_LongGap_3Year_MaxDistance',
        'protocol_name': 'Candidate 1: 3-Year Cutoff with Maximal Long-Gap Event OOD (Recommended)',
        'protocol_structure': 'Training -> Val -> ID Reference -> Long-Gap Event-Intensive OOD',
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
        'ood_target_m5_range': 'd_1914-d_1941 (Long-Gap Event-Intensive OOD)',
        'ood_target_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_event_days': 4,
        'ood_events': 'Cinco De Mayo, Mother\'s day, OrthodoxEaster, Pesach End',
        'ood_event_types': 'Cultural:1; National:1; Religious:2',
        'temporal_gap_from_val_end_days': 790,
        'temporal_gap_from_id_end_days': 762,
        'temporal_gap_interpretation': 'Maximal Long Gap (762d / 2.1 years / 25 months post-ID). Rigorous multi-year temporal degradation & spring holiday event test.',
        'pareto_efficiency_status': 'Pareto-Optimal (Maximal Temporal Separation)',
        'same_checkpoint_evaluation': 'Yes (Evaluated on frozen d1-d1095 checkpoint using observed d1824-d1913 sales as 90d lookback context)',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping target periods'
    },
    {
        'protocol_id': 'Candidate_2_LongGap_End2014_Balanced',
        'protocol_name': 'Candidate 2: End-of-2014 Cutoff with Balanced Long-Gap Event OOD',
        'protocol_structure': 'Training -> Val -> ID Reference -> Long-Gap Event-Intensive OOD',
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
        'ood_target_m5_range': 'd_1914-d_1941 (Long-Gap Event-Intensive OOD)',
        'ood_target_calendar_dates': '2016-04-25 to 2016-05-22',
        'ood_event_days': 4,
        'ood_events': 'Cinco De Mayo, Mother\'s day, OrthodoxEaster, Pesach End',
        'ood_event_types': 'Cultural:1; National:1; Religious:2',
        'temporal_gap_from_val_end_days': 452,
        'temporal_gap_from_id_end_days': 424,
        'temporal_gap_interpretation': 'Balanced Long Gap (424d / 1.16 years / 14 months post-ID). Excellent trade-off between large training size (1433d) and 1+ year temporal gap.',
        'pareto_efficiency_status': 'Pareto-Optimal (Balanced Training Size & 1+ Yr Gap)',
        'same_checkpoint_evaluation': 'Yes (Evaluated on frozen d1-d1433 checkpoint using observed d1824-d1913 sales as 90d lookback context)',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping target periods'
    },
    {
        'protocol_id': 'Candidate_3_LongGap_3Year_MaxEvent',
        'protocol_name': 'Candidate 3: 3-Year Cutoff with High Event-Density Long-Gap OOD',
        'protocol_structure': 'Training -> Val -> ID Reference -> Long-Gap Event-Intensive OOD',
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
        'ood_target_m5_range': 'd_1836-d_1863 (High Event-Density Long-Gap OOD)',
        'ood_target_calendar_dates': '2016-02-07 to 2016-03-05',
        'ood_event_days': 5,
        'ood_events': 'LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay',
        'ood_event_types': 'Cultural:1; National:1; Religious:2; Sporting:1',
        'temporal_gap_from_val_end_days': 712,
        'temporal_gap_from_id_end_days': 684,
        'temporal_gap_interpretation': 'High-Event Long Gap (684d / 1.87 years / 22.5 months post-ID). Maximizes post-ID event density (5 events) while maintaining a 1.87-year gap.',
        'pareto_efficiency_status': 'Pareto-Optimal (Maximal Post-ID Event Density)',
        'same_checkpoint_evaluation': 'Yes (Evaluated on frozen d1-d1095 checkpoint using observed d1746-d1835 sales as 90d lookback context)',
        'lookback_feasibility': 'All evaluation windows have complete 90-day lookbacks',
        'target_overlap_status': 'Strictly non-overlapping target periods'
    }
]

df_proto = pd.DataFrame(protocols)
proto_path = os.path.join(out_dir, "candidate_protocols.csv")
df_proto.to_csv(proto_path, index=False)
print(f"Saved updated {proto_path} successfully. Shape:", df_proto.shape)

# Build summary_md using plain string concatenation
summary_md = """# Protocol Analysis Summary: Pareto-Efficient Long-Gap Event-Intensive OOD Protocols

This report presents an extended, model-independent exploratory analysis of the M5 demand forecasting dataset. It identifies **Pareto-efficient candidate OOD target scenarios** that simultaneously achieve:
1. **Genuine temporal separation** from model development; and
2. **High event-day concentration** according to M5 calendar metadata.

No forecasting models were trained, and zero forecasting accuracy metrics (WRMSSE, MAE, MASE) or distribution distance metrics (KS, Wasserstein) were used.

---

## 1. M5 Temporal Profile Visualization

The plot below visualizes the chronological progression of key metadata and sales statistics across all 69 non-overlapping 28-day blocks ($d_1$ to $d_{1932}$):

![M5 Temporal Profile across 69 Non-Overlapping 28-Day Blocks](m5_temporal_profile.png)

---

## 2. Pareto Frontier Analysis of Post-ID Candidate OOD Windows

For each model-development cutoff, every eligible rolling 28-day target window occurring post-Validation and post-ID Reference was evaluated. A candidate is **Pareto-efficient** if no other candidate possesses both a longer temporal distance from ID/Val AND a higher or equal event-day count.

### Pareto Frontier Summary by Cutoff

#### A. 3-Year Model Development Cutoff ($T_{train\_end} = 1095$, Jan 27, 2014)
- **Validation**: `d_1096-d_1123` (Feb 2014, 3 event days).
- **ID Reference Target**: `d_1124-d_1151` (Mar 2014, 4 event days).
- **Pareto-Efficient OOD Candidates**:
  1. **`d_1914-d_1941` (Apr 25 – May 22, 2016)**:
     - *Temporal Distance from ID*: **762 days (2.1 years / 25 months)** [Maximal Temporal Gap].
     - *Temporal Distance from Val*: **790 days**.
     - *Event-Day Count*: **4 event days** (*Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End*).
     - *Event Types*: Cultural: 1, National: 1, Religious: 2.
     - *Holiday Context*: Spring holiday cluster.
  2. **`d_1836-d_1863` (Feb 7 – Mar 5, 2016)**:
     - *Temporal Distance from ID*: **684 days (1.87 years / 22.5 months)**.
     - *Event-Day Count*: **5 event days** (*LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay*).
     - *Event Types*: Cultural: 1, National: 1, Religious: 2, Sporting: 1.

#### B. End-of-2014 Model Development Cutoff ($T_{train\_end} = 1433$, Dec 31, 2014)
- **Validation**: `d_1434-d_1461` (Jan 2015, 3 event days).
- **ID Reference Target**: `d_1462-d_1489` (Feb 2015, 5 event days).
- **Pareto-Efficient OOD Candidates**:
  1. **`d_1914-d_1941` (Apr 25 – May 22, 2016)**:
     - *Temporal Distance from ID*: **424 days (1.16 years / 14 months)** [Maximal Temporal Gap].
     - *Event-Day Count*: **4 event days** (*Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End*).
  2. **`d_1836-d_1863` (Feb 7 – Mar 5, 2016)**:
     - *Temporal Distance from ID*: **346 days (0.95 years / 11.4 months)**.
     - *Event-Day Count*: **5 event days**.

---

## 3. Recommended Candidate Protocols for Supervisor Review

We recommend the following **three Pareto-efficient candidate protocols** (each specifying a single, unified Long-Gap Event-Intensive OOD scenario):

### Candidate Protocol Comparison Matrix

| Protocol Feature | Candidate 1 (Max Distance - Recommended) | Candidate 2 (Balanced Size & Gap) | Candidate 3 (Max Event Density) |
| :--- | :--- | :--- | :--- |
| **Model Cutoff ($T_{train\_end}$)** | `d_1095` (2014-01-27) | `d_1433` (2014-12-31) | `d_1095` (2014-01-27) |
| **Training History Length** | **1,095 days (3.0 years)** | **1,433 days (~3.9 years)** | **1,095 days (3.0 years)** |
| **Validation Window (28d)** | `d_1096-d_1123` (3 event days) | `d_1434-d_1461` (3 event days) | `d_1096-d_1123` (3 event days) |
| **ID Reference Target (28d)** | `d_1124-d_1151` (4 event days) | `d_1462-d_1489` (5 event days) | `d_1124-d_1151` (4 event days) |
| **Long-Gap Event OOD Target** | **`d_1914-d_1941`** | **`d_1914-d_1941`** | **`d_1836-d_1863`** |
| **OOD Calendar Dates** | 2016-04-25 to 2016-05-22 | 2016-04-25 to 2016-05-22 | 2016-02-07 to 2016-03-05 |
| **OOD Event-Day Count** | **4 event days** | **4 event days** | **5 event days** |
| **OOD Named Events** | Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End | Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End | LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay |
| **Temporal Gap from ID End** | **762 days (2.1 yrs / 25 mos)** | **424 days (1.16 yrs / 14 mos)** | **684 days (1.87 yrs / 22.5 mos)** |
| **Temporal Gap from Val End** | **790 days** | **452 days** | **712 days** |
| **Pareto Rationale** | Maximal temporal distance | Best training size & 1+ yr gap | Highest post-ID event density |
| **Same Checkpoint Evaluation?** | **Yes** (Frozen $T_{train\_end}$ weights) | **Yes** (Frozen $T_{train\_end}$ weights) | **Yes** (Frozen $T_{train\_end}$ weights) |

---

## 4. Evaluation Protocol Design Comparison: Single OOD vs. Dual OOD

We compare the simplified 4-stage protocol:

$$\text{Training} \longrightarrow \text{Validation (28d)} \longrightarrow \text{ID Reference (28d)} \longrightarrow \text{Long-Gap Event-Intensive OOD (28d)}$$

against the previous 5-stage protocol containing two separate OOD scenarios:

### Comparative Analysis Dimensions

1. **Simplicity**:
   - The **simplified single-OOD protocol** requires evaluating only one out-of-domain target scenario (`d_1914-d_1941`).
   - It eliminates redundant evaluation tables, reduces benchmark complexity, and avoids artificial distinctions between "Extended-Gap OOD" vs "Event-Intensive OOD".

2. **Academic Defensibility**:
   - Under earlier cutoffs ($T_{train\_end} = 1095$ or $1433$), `d_1914-d_1941` is **simultaneously** genuinely long-gap (762d or 424d temporal separation) AND event-intensive (4 event days).
   - This single window provides a unified, uncompromised test of both temporal degradation and event shock resilience without requiring arbitrary dual-window split rules.

3. **Alignment with Core Research Questions**:
   - The primary research objective is evaluating lightweight student forecasting models and Knowledge Distillation (KD) under distribution shift.
   - Testing on a single robust OOD target directly answers whether Student+KD models outperform standalone Student/Teacher models under combined temporal drift and event stress.

4. **Focus on Lightweight Forecasting Contribution**:
   - Having two separate OOD target scenarios shifts thesis focus heavily onto complex benchmark taxonomy.
   - The simplified single-OOD design keeps the experimental structure lean, elegant, and focused on the core Knowledge Distillation contribution.

---

## 5. Protocol Invariants & Checkpoint Verification

All three recommended candidate protocols satisfy every required invariant:
- **Chronological Priority**: All Validation, ID, and OOD targets occur strictly post-training ($T_{eval} > T_{train\_end}$).
- **Non-Overlapping Targets**: All evaluation target windows are strictly non-overlapping.
- **90-Day Lookback Availability**: Every forecast origin $T_{origin}$ has a complete 90-day lookback ($T_{origin}-89$ to $T_{origin}$) in observed sales data.
- **Frozen Model Checkpoint Evaluation**: Models (TFT Teacher, Student, Student+KD) are trained ONCE on $d_1..T_{train\_end}$. The frozen model checkpoint evaluates both the ID reference target and the Long-Gap Event OOD target by taking the actual preceding 90-day observed sales ($T_{origin}-89..T_{origin}$) as local encoder context without updating model parameters.
- **Model-Independent Selection**: Scenario dates were selected strictly using calendar metadata and chronology without using model error metrics or predictions.
"""

# Write to protocol_analysis_summary.md and summary.md
report_path1 = os.path.join(out_dir, "protocol_analysis_summary.md")
report_path2 = os.path.join(out_dir, "summary.md")

with open(report_path1, "w") as f:
    f.write(summary_md)

with open(report_path2, "w") as f:
    f.write(summary_md)

print(f"Updated {report_path1} and {report_path2} successfully.")
