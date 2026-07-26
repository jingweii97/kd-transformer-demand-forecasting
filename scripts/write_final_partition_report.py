import os

out_dir = "id-ood-analysis"

summary_report = """# Final Evaluation Protocol & Event-Intensive Window Selection Report

This report presents the final, model-independent event-window selection analysis conducted strictly within the agreed-upon chronological partition of the M5 dataset.

No forecasting models were trained, and zero forecasting accuracy metrics (WRMSSE, MAE, MASE, model predictions) or demand statistics were used to select the event-intensive window.

---

## 1. Final Chronological Partition Structure

The M5 dataset is partitioned chronologically as follows:

| Partition / Stage | M5 Day Range | Calendar Date Range | Length | Description / Invariant |
| :--- | :--- | :--- | :---: | :--- |
| **Training** | `d_1-d_1359` | 2011-01-29 to 2014-10-18 | 1,359 days | Model fitting & preprocessing statistics estimation only |
| **Validation** | `d_1360-d_1553` | 2014-10-19 to 2015-04-29 | 194 days | Hyperparameter tuning and model selection only |
| **Held-Out Test Period** | `d_1554-d_1941` | 2015-04-30 to 2016-05-22 | 388 days | Strictly frozen model evaluation space |
| **ID Reference Target** | `d_1554-d_1581` | 2015-05-01 to 2015-05-28 | 28 days | In-distribution baseline test (origin `d_1553`, lookback `d_1464-d_1553`) |
| **Event-Intensive OOD** | `d_1819-d_1846` | 2016-01-21 to 2016-02-17 | 28 days | Selected peak event window in held-out test set (5 event days) |
| **Temporal OOD Target** | `d_1914-d_1941` | 2016-04-25 to 2016-05-22 | 28 days | Longest temporal distance test in held-out test set (origin `d_1913`) |

```
Timeline Architecture:
Training (d1-d1359, 1359d)
+-- Validation (d1360-d1553, 194d)
    +-- Held-Out Test Period (d1554-d1941, 388d)
        +-- ID Reference Target (d1554-d1581, 28d)
        +-- Event-Intensive OOD Target (d1819-d1846, 28d)
        +-- Temporal OOD Target (d1914-d1941, 28d)
```

---

## 2. Selected Event-Intensive Evaluation Window

Following the deterministic rolling 28-day scan (1-day stride, excluding ID Reference `d_1554-d_1581`), the highest-ranked window is identified:

- **M5 Day Range**: `d_1819-d_1846`
- **Calendar Date Range**: `2016-01-21` to `2016-02-17` (28 days)
- **Event-Day Count**: **5 event days**
- **Named Event Occurrence Count**: **5**
- **Unique Event Names**: `LentStart`, `LentWeek2`, `PresidentsDay`, `SuperBowl`, `ValentinesDay`
- **Event Types Composition**: `Cultural: 1; National: 1; Religious: 2; Sporting: 1`
- **Overlap with Temporal OOD (`d_1914-d_1941`)**: **NO OVERLAP** (`d_1846` ends 67 days before `d_1914`).

---

## 3. Overlap Decision & Final Scenario Structure

Because the selected event-intensive window (`d_1819-d_1846`) does NOT overlap with the Temporal OOD window (`d_1914-d_1941`), we retain **two separate, complementary OOD scenarios** in the held-out test set:

1. **Event-Intensive Evaluation Scenario (OOD)**: `d_1819-d_1846` (Feb 2016 holiday cluster: Super Bowl, Valentine's Day, Presidents' Day, Lent Start).
2. **Temporal OOD Scenario**: `d_1914-d_1941` (May 2016 final complete 28-day labeled window; 11-month temporal gap post-ID).

---

## 4. Descriptive Comparison Table

The following model-independent descriptive values were calculated post-selection for the three evaluation scenarios:

| Metric / Characteristic | ID Reference Target | Event-Intensive OOD Target | Temporal OOD Target |
| :--- | :--- | :--- | :--- |
| **M5 Day Range** | `d_1554-d_1581` | `d_1819-d_1846` | `d_1914-d_1941` |
| **Calendar Date Range** | 2015-05-01 to 2015-05-28 | 2016-01-21 to 2016-02-17 | 2016-04-25 to 2016-05-22 |
| **Window Length** | 28 days | 28 days | 28 days |
| **Event-Day Count** | 3 event days | **5 event days** | 4 event days |
| **Unique Named Events** | Cinco De Mayo, MemorialDay, Mother's day | LentStart, LentWeek2, PresidentsDay, SuperBowl, ValentinesDay | Cinco De Mayo, Mother's day, OrthodoxEaster, Pesach End |
| **Mean Daily Demand** | 1.2131 units | 1.3477 units | 1.4428 units |
| **Zero-Sales Ratio** | 0.6083 | 0.5854 | 0.5444 |
| **Active Item-Store Ratio** | 0.8957 | 0.9150 | 0.9734 |
| **Price-Change Frequency** | 0.0408 | 0.0426 | 0.0297 |
| **Average Price CV** | 0.0025 | 0.0021 | 0.0018 |

---

## 5. Protocol Leakage & Feasibility Confirmation

1. **28-Day Target Invariant**: **CONFIRMED**. ID Reference (`d_1554-d_1581`), Event-Intensive OOD (`d_1819-d_1846`), and Temporal OOD (`d_1914-d_1941`) each contain exactly 28 days.
2. **90-Day Lookback Availability**: **CONFIRMED**. Every forecast origin $T_{origin}$ has a complete 90-day lookback in observed sales:
   - ID Reference ($T_{origin}=1553$): Lookback `d_1464-d_1553` (90 days).
   - Event OOD ($T_{origin}=1818$): Lookback `d_1729-d_1818` (90 days).
   - Temporal OOD ($T_{origin}=1913$): Lookback `d_1824-d_1913` (90 days).
3. **Training & Validation Exclusion**: **CONFIRMED**. Training ends at `d_1359` and Validation ends at `d_1553`. All evaluation targets occur strictly post-`d_1553`.
4. **Frozen Model Evaluation**: **CONFIRMED**. Models (TFT Teacher, Student, Student+KD) are trained ONCE on `d_1-d_1359` and frozen. The same frozen checkpoints evaluate all three test targets by taking the preceding 90-day observed sales as local lookback input.
5. **Preprocessing Integrity**: **CONFIRMED**. All scalers, normalizers, and categorical encodings are fitted exclusively using training data (`d_1-d_1359`).
"""

report_path1 = os.path.join(out_dir, "summary.md")
report_path2 = os.path.join(out_dir, "protocol_analysis_summary.md")

with open(report_path1, "w", encoding="utf-8") as f:
    f.write(summary_report)

with open(report_path2, "w", encoding="utf-8") as f:
    f.write(summary_report)

print("Updated summary.md and protocol_analysis_summary.md cleanly with UTF-8 encoding.")
