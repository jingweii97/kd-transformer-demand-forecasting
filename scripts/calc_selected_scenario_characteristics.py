import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

def calc_selected_scenario_characteristics():
    cal_path = "input/calendar.csv"
    sales_path = "input/sales_train_evaluation.csv"
    prices_path = "input/sell_prices.csv"

    cal = pd.read_csv(cal_path)
    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(prices_path)

    cal['d_num'] = cal['d'].apply(lambda x: int(x.split('_')[1]))
    cal_eval = cal[cal['d_num'] <= 1941].copy()
    cal_eval['has_event_1'] = cal_eval['event_name_1'].notna()
    cal_eval['has_event_2'] = cal_eval['event_name_2'].notna()
    cal_eval['is_event_day'] = cal_eval['has_event_1'] | cal_eval['has_event_2']
    cal_eval['event_count'] = cal_eval['has_event_1'].astype(int) + cal_eval['has_event_2'].astype(int)

    d_cols = [c for c in sales.columns if c.startswith('d_')]
    sales_mat = sales[d_cols].values # shape: (30490, 1941)

    scenarios = [
        {'name': 'Training_Reference_Design1', 'd_start': 1, 'd_end': 1829},
        {'name': 'Validation_Design1', 'd_start': 1830, 'd_end': 1857},
        {'name': 'ID_Reference_Design1', 'd_start': 1858, 'd_end': 1885},
        {'name': 'Extended_Gap_OOD_Design1', 'd_start': 1886, 'd_end': 1913},
        {'name': 'Event_Intensive_OOD_Design1', 'd_start': 1914, 'd_end': 1941},
        {'name': 'Historical_Peak_Event_Descriptive', 'd_start': 731, 'd_end': 758}
    ]

    # Precompute training sample for KS and Wasserstein
    train_sales = sales_mat[:, :1829].flatten()
    np.random.seed(42)
    train_sample = np.random.choice(train_sales, size=100000, replace=False)

    results = []

    for sc in scenarios:
        d_start = sc['d_start']
        d_end = sc['d_end']
        name = sc['name']
        
        idx_start = d_start - 1
        idx_end = d_end
        sc_sales = sales_mat[:, idx_start:idx_end]
        sc_flat = sc_sales.flatten()
        
        d_mean = float(np.mean(sc_flat))
        d_var = float(np.var(sc_flat))
        d_std = float(np.std(sc_flat))
        q25, q50, q75, q90, q95, q99 = np.quantile(sc_flat, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        zero_ratio = float(np.mean(sc_flat == 0))
        active_sku_ratio = float(np.mean(np.sum(sc_sales, axis=1) > 0))
        
        sc_sample = np.random.choice(sc_flat, size=min(100000, len(sc_flat)), replace=False) if len(sc_flat) > 100000 else sc_flat
        ks_stat, _ = ks_2samp(train_sample, sc_sample)
        w_dist = wasserstein_distance(train_sample, sc_sample)
        
        sub_cal = cal_eval[(cal_eval['d_num'] >= d_start) & (cal_eval['d_num'] <= d_end)]
        num_days = len(sub_cal)
        event_days = int(sub_cal['is_event_day'].sum())
        event_freq = float(event_days / num_days) if num_days > 0 else 0.0
        total_events = int(sub_cal['event_count'].sum())
        
        e1 = sub_cal['event_name_1'].dropna().tolist()
        e2 = sub_cal['event_name_2'].dropna().tolist()
        unique_events = sorted(list(set(e1 + e2)))
        
        t1 = sub_cal['event_type_1'].dropna().tolist()
        t2 = sub_cal['event_type_2'].dropna().tolist()
        type_counts = pd.Series(t1 + t2).value_counts().to_dict()
        comp_str = "; ".join([f"{k}:{v}" for k, v in sorted(type_counts.items())]) if type_counts else "None"
        
        snap_ca = int(sub_cal['snap_CA'].sum())
        snap_tx = int(sub_cal['snap_TX'].sum())
        snap_wi = int(sub_cal['snap_WI'].sum())
        
        # Price stats
        wm_weeks = sub_cal['wm_yr_wk'].unique()
        sub_prices = prices[prices['wm_yr_wk'].isin(wm_weeks)].copy()
        
        if len(sub_prices) > 0:
            price_stats = sub_prices.groupby(['store_id', 'item_id'])['sell_price'].agg(['std', 'mean'])
            price_cv = price_stats['std'] / price_stats['mean']
            avg_price_cv = float(price_cv.fillna(0).mean())
        else:
            avg_price_cv = 0.0
            
        results.append({
            'scenario_name': name,
            'm5_day_range': f"d_{d_start}-d_{d_end}",
            'num_days': num_days,
            'demand_mean': round(d_mean, 4),
            'demand_var': round(d_var, 4),
            'demand_std': round(d_std, 4),
            'demand_q25': round(float(q25), 4),
            'demand_q50': round(float(q50), 4),
            'demand_q75': round(float(q75), 4),
            'demand_q90': round(float(q90), 4),
            'demand_q95': round(float(q95), 4),
            'demand_q99': round(float(q99), 4),
            'zero_sales_ratio': round(zero_ratio, 4),
            'active_sku_ratio': round(active_sku_ratio, 4),
            'ks_stat_vs_train': round(float(ks_stat), 4),
            'wasserstein_dist_vs_train': round(float(w_dist), 4),
            'event_day_count': event_days,
            'event_day_frequency': round(event_freq, 4),
            'total_named_event_occurrences': total_events,
            'unique_event_name_count': len(unique_events),
            'unique_event_names': ", ".join(unique_events),
            'event_type_composition': comp_str,
            'snap_ca_days': snap_ca,
            'snap_tx_days': snap_tx,
            'snap_wi_days': snap_wi,
            'avg_price_cv': round(avg_price_cv, 4)
        })

    df_sc = pd.DataFrame(results)
    df_sc.to_csv("selected_scenario_characteristics.csv", index=False)
    print("Saved selected_scenario_characteristics.csv with shape:", df_sc.shape)

if __name__ == "__main__":
    calc_selected_scenario_characteristics()
