import os
import pandas as pd
import numpy as np

def generate_protocol_tables():
    # Load raw inputs
    cal_path = "input/calendar.csv"
    sales_path = "input/sales_train_evaluation.csv"
    prices_path = "input/sell_prices.csv"
    
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Calendar file not found at {cal_path}")
        
    cal = pd.read_csv(cal_path)
    cal['d_num'] = cal['d'].apply(lambda x: int(x.split('_')[1]))

    # Limit to labeled target range d_1 to d_1941
    cal_eval = cal[cal['d_num'] <= 1941].copy()

    # Event flags and occurrences
    cal_eval['has_event_1'] = cal_eval['event_name_1'].notna()
    cal_eval['has_event_2'] = cal_eval['event_name_2'].notna()
    cal_eval['is_event_day'] = cal_eval['has_event_1'] | cal_eval['has_event_2']
    cal_eval['event_count'] = cal_eval['has_event_1'].astype(int) + cal_eval['has_event_2'].astype(int)

    # 1. Full-History Ranking (event_window_candidates_all_history.csv)
    all_cands = []
    for d_start in range(91, 1941 - 28 + 2):
        d_end = d_start + 27
        sub = cal_eval[(cal_eval['d_num'] >= d_start) & (cal_eval['d_num'] <= d_end)]
        
        event_days = sub['is_event_day'].sum()
        total_events = sub['event_count'].sum()
        
        e1 = sub['event_name_1'].dropna().tolist()
        e2 = sub['event_name_2'].dropna().tolist()
        all_events = e1 + e2
        unique_events = sorted(list(set(all_events)))
        
        t1 = sub['event_type_1'].dropna().tolist()
        t2 = sub['event_type_2'].dropna().tolist()
        type_counts = pd.Series(t1 + t2).value_counts().to_dict()
        comp_str = "; ".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "None"
        
        multi_event_days = (sub['has_event_1'] & sub['has_event_2']).sum()
        
        all_cands.append({
            'd_start': d_start,
            'd_end': d_end,
            'm5_day_range': f"d_{d_start}-d_{d_end}",
            'calendar_date_range': f"{sub['date'].min()} to {sub['date'].max()}",
            'event_day_count': event_days,
            'total_named_event_occurrences': total_events,
            'unique_event_name_count': len(unique_events),
            'unique_event_names': ", ".join(unique_events),
            'event_type_composition': comp_str,
            'multi_event_day_count': multi_event_days,
            'snap_ca_days': sub['snap_CA'].sum(),
            'snap_tx_days': sub['snap_TX'].sum(),
            'snap_wi_days': sub['snap_WI'].sum(),
            'snap_any_state_days': ((sub['snap_CA'] == 1) | (sub['snap_TX'] == 1) | (sub['snap_WI'] == 1)).sum(),
            'has_90d_lookback': True
        })

    df_all_history = pd.DataFrame(all_cands)
    df_all_history = df_all_history.sort_values(by=['event_day_count', 'd_start'], ascending=[False, True]).reset_index(drop=True)
    df_all_history.to_csv("event_window_candidates_all_history.csv", index=False)
    print("Saved event_window_candidates_all_history.csv with shape:", df_all_history.shape)

    # 2. Future-Eligible Ranking (event_window_candidates_future_eligible.csv)
    future_cands = []
    for d_start in range(1858, 1941 - 28 + 2):
        d_end = d_start + 27
        sub = cal_eval[(cal_eval['d_num'] >= d_start) & (cal_eval['d_num'] <= d_end)]
        
        event_days = sub['is_event_day'].sum()
        total_events = sub['event_count'].sum()
        
        e1 = sub['event_name_1'].dropna().tolist()
        e2 = sub['event_name_2'].dropna().tolist()
        unique_events = sorted(list(set(e1 + e2)))
        
        t1 = sub['event_type_1'].dropna().tolist()
        t2 = sub['event_type_2'].dropna().tolist()
        type_counts = pd.Series(t1 + t2).value_counts().to_dict()
        comp_str = "; ".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "None"
        
        multi_event_days = (sub['has_event_1'] & sub['has_event_2']).sum()
        
        future_cands.append({
            'd_start': d_start,
            'd_end': d_end,
            'm5_day_range': f"d_{d_start}-d_{d_end}",
            'calendar_date_range': f"{sub['date'].min()} to {sub['date'].max()}",
            'event_day_count': event_days,
            'total_named_event_occurrences': total_events,
            'unique_event_name_count': len(unique_events),
            'unique_event_names': ", ".join(unique_events),
            'event_type_composition': comp_str,
            'multi_event_day_count': multi_event_days,
            'snap_ca_days': sub['snap_CA'].sum(),
            'snap_tx_days': sub['snap_TX'].sum(),
            'snap_wi_days': sub['snap_WI'].sum(),
            'snap_any_state_days': ((sub['snap_CA'] == 1) | (sub['snap_TX'] == 1) | (sub['snap_WI'] == 1)).sum(),
            'has_90d_lookback': True,
            'occurs_after_validation': True
        })

    df_future = pd.DataFrame(future_cands)
    df_future = df_future.sort_values(by=['event_day_count', 'd_start'], ascending=[False, True]).reset_index(drop=True)
    df_future.to_csv("event_window_candidates_future_eligible.csv", index=False)
    print("Saved event_window_candidates_future_eligible.csv with shape:", df_future.shape)

    # 3. Descriptive Fixed-Block Summary (event_window_summary.csv)
    blocks = []
    block_id = 1
    for d_start in range(1, 1941, 28):
        d_end = min(d_start + 27, 1941)
        sub = cal_eval[(cal_eval['d_num'] >= d_start) & (cal_eval['d_num'] <= d_end)]
        
        event_days = sub['is_event_day'].sum()
        total_events = sub['event_count'].sum()
        
        e1 = sub['event_name_1'].dropna().tolist()
        e2 = sub['event_name_2'].dropna().tolist()
        unique_events = sorted(list(set(e1 + e2)))
        
        t1 = sub['event_type_1'].dropna().tolist()
        t2 = sub['event_type_2'].dropna().tolist()
        type_counts = pd.Series(t1 + t2).value_counts().to_dict()
        comp_str = "; ".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "None"
        
        blocks.append({
            'block_id': block_id,
            'm5_day_range': f"d_{d_start}-d_{d_end}",
            'calendar_date_range': f"{sub['date'].min()} to {sub['date'].max()}",
            'num_days': len(sub),
            'event_day_count': event_days,
            'total_named_event_occurrences': total_events,
            'unique_event_name_count': len(unique_events),
            'event_type_composition': comp_str,
            'snap_ca_days': sub['snap_CA'].sum(),
            'snap_tx_days': sub['snap_TX'].sum(),
            'snap_wi_days': sub['snap_WI'].sum(),
            'note': "Descriptive fixed-block summary (not used for scenario selection)"
        })
        block_id += 1

    df_blocks = pd.DataFrame(blocks)
    df_blocks.to_csv("event_window_summary.csv", index=False)
    print("Saved event_window_summary.csv with shape:", df_blocks.shape)

    # 4. Feasible Temporal Designs (feasible_temporal_designs.csv)
    designs = [
        {
            'design_id': 'Design_1_SingleCutoff_4Blocks',
            'type': 'Primary Single-Cutoff (Recommended)',
            'description': '4 non-overlapping 28-day evaluation blocks post-training (T_train_end=1829)',
            'training_m5_range': 'd_1-d_1829',
            'training_calendar_dates': '2011-01-29 to 2016-01-31',
            'training_history_days': 1829,
            'validation_m5_range': 'd_1830-d_1857',
            'validation_dates': '2016-02-01 to 2016-02-28',
            'val_event_day_count': 5,
            'id_reference_m5_range': 'd_1858-d_1885',
            'id_reference_dates': '2016-02-29 to 2016-03-27',
            'id_event_day_count': 3,
            'extended_gap_ood_m5_range': 'd_1886-d_1913',
            'extended_gap_ood_dates': '2016-03-28 to 2016-04-24',
            'ext_gap_event_day_count': 0,
            'event_intensive_ood_m5_range': 'd_1914-d_1941',
            'event_intensive_ood_dates': '2016-04-25 to 2016-05-22',
            'event_ood_event_day_count': 4,
            'lookback_feasibility': 'All evaluation windows have complete 90-day lookback',
            'overlap_status': 'Strictly non-overlapping',
            'temporal_gaps': 'No temporal gaps between evaluation blocks',
            'feasibility_status': 'Fully Feasible'
        },
        {
            'design_id': 'Design_2_SingleCutoff_StandardTrain',
            'type': 'Primary Single-Cutoff (Standard Cutoff T_train_end=1857)',
            'description': 'Standard M5 train cutoff (d_1-d_1857). 56 days post-validation allows 2 distinct 28-day test targets.',
            'training_m5_range': 'd_1-d_1857',
            'training_calendar_dates': '2011-01-29 to 2016-02-28',
            'training_history_days': 1857,
            'validation_m5_range': 'd_1858-d_1885',
            'validation_dates': '2016-02-29 to 2016-03-27',
            'val_event_day_count': 3,
            'id_reference_m5_range': 'd_1886-d_1913',
            'id_reference_dates': '2016-03-28 to 2016-04-24',
            'id_event_day_count': 0,
            'event_intensive_ood_m5_range': 'd_1914-d_1941',
            'event_intensive_ood_dates': '2016-04-25 to 2016-05-22',
            'event_ood_event_day_count': 4,
            'extended_gap_ood_m5_range': 'd_1914-d_1941 (Dual Role)',
            'extended_gap_ood_dates': '2016-04-25 to 2016-05-22',
            'ext_gap_event_day_count': 4,
            'lookback_feasibility': 'All evaluation windows have complete 90-day lookback',
            'overlap_status': 'Event OOD and Extended Gap share d_1914-d_1941 (dual role due to 56d post-val length)',
            'temporal_gaps': 'No temporal gaps',
            'feasibility_status': 'Feasible with Dual-Role OOD Target'
        },
        {
            'design_id': 'Design_3_Secondary_RollingOrigin',
            'type': 'Secondary Rolling-Origin Alternative',
            'description': 'Separate historical training through d_730 specifically for evaluating historical peak event window d_731-d_758.',
            'training_m5_range': 'Fold A: d_1-d_730; Fold B: d_1-d_1857',
            'training_calendar_dates': 'Fold A: 2011-01-29 to 2013-01-27; Fold B: 2011-01-29 to 2016-02-28',
            'training_history_days': 'Fold A: 730 days; Fold B: 1857 days',
            'validation_m5_range': 'd_1858-d_1885',
            'validation_dates': '2016-02-29 to 2016-03-27',
            'val_event_day_count': 3,
            'id_reference_m5_range': 'd_1886-d_1913',
            'id_reference_dates': '2016-03-28 to 2016-04-24',
            'id_event_day_count': 0,
            'event_intensive_ood_m5_range': 'd_731-d_758 (Historical Retrospective)',
            'event_intensive_ood_dates': '2013-01-28 to 2013-02-24',
            'event_ood_event_day_count': 6,
            'extended_gap_ood_m5_range': 'd_1914-d_1941',
            'extended_gap_ood_dates': '2016-04-25 to 2016-05-22',
            'ext_gap_event_day_count': 4,
            'lookback_feasibility': 'Complete 90-day lookbacks',
            'overlap_status': 'Non-overlapping targets',
            'temporal_gaps': 'Fold A has 1127-day gap to post-training era; different training length',
            'feasibility_status': 'Secondary Rolling-Origin Alternative'
        }
    ]

    df_designs = pd.DataFrame(designs)
    df_designs.to_csv("feasible_temporal_designs.csv", index=False)
    print("Saved feasible_temporal_designs.csv with shape:", df_designs.shape)

if __name__ == "__main__":
    generate_protocol_tables()
