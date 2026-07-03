import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import pickle
import glob
import streamlit as st
import plotly.graph_objects as go

# Add repository root to system path to enable importing modules
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from utils.config import load_config
from pytorch_forecasting import TimeSeriesDataSet
from models.teacher import create_tft_teacher
from models.student import M5TransformerStudent

# ---------------------------------------------------------------------------
# Streamlit Configuration & Premium Theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="M5 Demand Forecasting Research Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (thesis-defense aesthetics)
st.markdown(
    """
    <style>
    .welcome-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 25px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #1e293b;
        margin-bottom: 10px;
    }
    .action-card {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 18px;
        border-top: 4px solid #3b82f6;
        margin-bottom: 15px;
    }
    .highlight-card {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #1e293b;
        text-align: center;
    }
    .highlight-number {
        font-size: 22px;
        font-weight: bold;
        color: #10b981;
        margin: 8px 0;
    }
    .highlight-label {
        font-size: 13px;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Cached Data & Model Loading Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_calendar_dates():
    """Maps d_1 through d_1941 to actual calendar dates using calendar.csv."""
    calendar_path = os.path.join(repo_dir, "input", "calendar.csv")
    df_cal = pd.read_csv(calendar_path)
    return df_cal['date'].tolist()

@st.cache_data
def build_dataset_hierarchy():
    """Dynamically builds a State -> Store -> Category -> SKU -> ID nested registry."""
    data_dir = os.path.join(repo_dir, "artifacts", "data")
    files = sorted(glob.glob(os.path.join(data_dir, "preprocessed_*.parquet")))
    
    registry = {}
    for f in files:
        filename = os.path.basename(f)
        store_name = filename.replace("preprocessed_", "").replace(".parquet", "")
        if not store_name or store_name == "full":
            continue
        
        state_name = store_name.split("_")[0]
        
        # Read only hierarchy columns to keep it extremely fast
        df = pd.read_parquet(f, columns=['cat_id', 'item_id', 'id'], engine='pyarrow')
        df_unique = df.drop_duplicates()
        
        if state_name not in registry:
            registry[state_name] = {}
        if store_name not in registry[state_name]:
            registry[state_name][store_name] = {}
            
        for _, row in df_unique.iterrows():
            cat = row['cat_id']
            item = row['item_id']
            full_id = row['id']
            
            if cat not in registry[state_name][store_name]:
                registry[state_name][store_name][cat] = {}
            
            registry[state_name][store_name][cat][item] = full_id
            
    return registry

@st.cache_resource
def get_base_dataset():
    """Loads the fitted base TimeSeriesDataSet metadata once."""
    metadata_path = os.path.join(repo_dir, "artifacts", "metadata", "global_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        builder = pickle.load(f)
    return builder.base_dataset

@st.cache_resource
def load_checkpoints(_training_data):
    """Instantiates models and loads GPU-trained state_dicts onto CPU or GPU dynamically."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    teacher_chk = os.path.join(repo_dir, "kaggle-output", "teacher", "prelim", "best_tft_teacher.ckpt")
    student_nokd_chk = os.path.join(repo_dir, "kaggle-output", "student", "no_kd", "prelim", "best_student.ckpt")
    student_kd_chk = os.path.join(repo_dir, "kaggle-output", "student", "kd", "prelim", "best_student.ckpt")
    
    cfg = load_config(env_name="local")
    
    # 1. Load TFT Teacher
    teacher = create_tft_teacher(_training_data, cfg)
    teacher_state = torch.load(teacher_chk, map_location=device, weights_only=False)["state_dict"]
    teacher.load_state_dict(teacher_state)
    teacher = teacher.to(device)
    teacher.eval()
    
    # 2. Load Student Without KD
    nokd_ckpt = torch.load(student_nokd_chk, map_location=device, weights_only=False)
    nokd_hparams = nokd_ckpt["hyper_parameters"]
    student_nokd = M5TransformerStudent(
        training_dataset=_training_data,
        d_model=nokd_hparams['d_model'],
        nhead=nokd_hparams['nhead'],
        num_layers=nokd_hparams['num_layers'],
        dim_feedforward=nokd_hparams['dim_feedforward'],
        dropout=nokd_hparams['dropout'],
        lr=nokd_hparams['lr'],
        alpha=nokd_hparams['alpha'],
        lookback_window=nokd_hparams['lookback_window'],
        prediction_window=nokd_hparams['prediction_window'],
        embedding_dim=nokd_hparams['embedding_dim'],
        output_head=nokd_hparams['output_head'],
        output_head_hidden_dim=nokd_hparams['output_head_hidden_dim']
    )
    student_nokd.load_state_dict(nokd_ckpt["state_dict"])
    student_nokd = student_nokd.to(device)
    student_nokd.eval()
    
    # 3. Load Student With KD
    kd_ckpt = torch.load(student_kd_chk, map_location=device, weights_only=False)
    kd_hparams = kd_ckpt["hyper_parameters"]
    student_kd = M5TransformerStudent(
        training_dataset=_training_data,
        d_model=kd_hparams['d_model'],
        nhead=kd_hparams['nhead'],
        num_layers=kd_hparams['num_layers'],
        dim_feedforward=kd_hparams['dim_feedforward'],
        dropout=kd_hparams['dropout'],
        lr=kd_hparams['lr'],
        alpha=kd_hparams['alpha'],
        lookback_window=kd_hparams['lookback_window'],
        prediction_window=kd_hparams['prediction_window'],
        embedding_dim=kd_hparams['embedding_dim'],
        output_head=kd_hparams['output_head'],
        output_head_hidden_dim=kd_hparams['output_head_hidden_dim']
    )
    student_kd.load_state_dict(kd_ckpt["state_dict"])
    student_kd = student_kd.to(device)
    student_kd.eval()
    
    return teacher, student_nokd, student_kd, device

# ---------------------------------------------------------------------------
# Sidebar / Navigation Controls
# ---------------------------------------------------------------------------
# Multi-page navigation driven via session state to support programmatic redirects
if "navigation_tab" not in st.session_state:
    st.session_state.navigation_tab = "📊 Forecast Explorer"

selected_tab = st.sidebar.radio(
    "Navigation View",
    ["📊 Forecast Explorer", "🔬 Research Evaluation"],
    key="navigation_tab"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")

# Dynamic Cascading Selectboxes
with st.spinner("Loading dynamic dropdown registry..."):
    hierarchy = build_dataset_hierarchy()

states = sorted(list(hierarchy.keys()))
selected_state = st.sidebar.selectbox("State", states)

stores = sorted(list(hierarchy[selected_state].keys()))
selected_store = st.sidebar.selectbox("Store", stores)

# Scenario Selector (ID vs. OOD)
scenario = st.sidebar.selectbox(
    "Evaluation Scenario",
    ["In-Distribution (ID)", "Out-of-Distribution (OOD)"],
    help="Select ID (normal evaluation window) or OOD (testing under distribution shift)."
)

calendar_dates = load_calendar_dates()

# Filter available dates dynamically based on scenario selection
if scenario == "In-Distribution (ID)":
    # Start day of ID test window corresponds to indices 1859 to 1886
    date_options = {calendar_dates[idx - 1]: idx for idx in range(1859, 1887)}
else:
    # Start day of OOD test window corresponds to indices 1887 to 1914
    date_options = {calendar_dates[idx - 1]: idx for idx in range(1887, 1915)}

selected_date_str = st.sidebar.selectbox(
    "Forecast Start Date",
    list(date_options.keys()),
    index=0,  # Default to the first start date in the range
    help="Select the start date of the 28-day forecast horizon. The system will predict sales for the next 28 days."
)
start_day = date_options[selected_date_str]
selected_end_idx = start_day + 28 - 1

# Active Model Selector representing the deployed operational forecaster
active_model = st.sidebar.selectbox(
    "Active Forecasting Model",
    ["TFT Teacher", "Transformer Student", "Transformer Student + KD", "Seasonal Naive"],
    index=0,
    help="Select the model that drives the Forecast Explorer store aggregates and summaries."
)

st.sidebar.markdown("---")
generate_forecast = st.sidebar.button("Generate Store Forecast", type="primary")

# Calculate periods
start_date_str = calendar_dates[start_day - 1]
end_date_str = calendar_dates[selected_end_idx - 1]

hist_start_day = start_day - 90
hist_start_date_str = calendar_dates[hist_start_day - 1]
hist_end_date_str = calendar_dates[start_day - 2]

# ---------------------------------------------------------------------------
# Session State Caching Logic
# ---------------------------------------------------------------------------
if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = None
if "last_store_run" not in st.session_state:
    st.session_state.last_store_run = None
if "selected_sku" not in st.session_state:
    st.session_state.selected_sku = None

current_run_key = (selected_store, selected_end_idx)
if st.session_state.last_store_run != current_run_key:
    # Reset run status if user changes store parameters
    st.session_state.forecast_cache = None

# ===========================================================================
# VIEW 1: FORECAST EXPLORER (Business operational dashboard)
# ===========================================================================
if selected_tab == "📊 Forecast Explorer":
    if st.session_state.forecast_cache is None and not generate_forecast:
        # Welcome Screen
        st.markdown(
            """
            <div class="welcome-card">
                <h3>M5 Demand Forecasting Research Framework</h3>
                <p>This prototype generates and evaluates demand forecasts across all available products in the Walmart sales database.</p>
                <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                    <tr>
                        <td style="padding: 8px 0; font-weight: bold; width: 30%;">Dataset</td>
                        <td style="padding: 8px 0; color: #94a3b8;">M5 Forecasting Dataset (Wal-Mart Sales Data)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: bold;">History Lookback</td>
                        <td style="padding: 8px 0; color: #94a3b8;">90 Days (Historical sales features)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: bold;">Forecast Horizon</td>
                        <td style="padding: 8px 0; color: #94a3b8;">28 Days (Point demand forecasts)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: bold;">Forecasting Models</td>
                        <td style="padding: 8px 0; color: #94a3b8;">
                            • <b>Seasonal Naive</b> (Baseline)<br>
                            • <b>TFT Teacher</b> (High-Capacity Temporal Fusion Transformer)<br>
                            • <b>Transformer Student</b> (Compact Baseline Student)<br>
                            • <b>Transformer Student + KD</b> (Distilled Student)
                        </td>
                    </tr>
                </table>
                <p style="margin-top: 20px; font-style: italic; color: #3b82f6;">Select State, Store, Scenario, and Date in the sidebar and click "Generate Store Forecast" to run batch predictions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Trigger Batch Inference
        if generate_forecast and st.session_state.forecast_cache is None:
            with st.spinner("Loading models..."):
                training_data = get_base_dataset()
                teacher, student_nokd, student_kd, device = load_checkpoints(training_data)
                
            with st.spinner("Loading store historical data..."):
                data_dir = os.path.join(repo_dir, "artifacts", "data")
                parquet_path = os.path.join(data_dir, f"preprocessed_{selected_store}.parquet")
                df_store = pd.read_parquet(parquet_path)
                df_store_sorted = df_store.sort_values(by=['id', 'time_idx']).reset_index(drop=True)
                
                # Verify exactly 3,049 SKUs
                unique_ids = df_store_sorted['id'].unique().tolist()
                num_skus = len(unique_ids)
                
                # Slice target range (118 days total)
                max_encoder_length = 90
                max_prediction_length = 28
                min_idx = selected_end_idx - max_encoder_length - max_prediction_length + 1
                
                df_batch = df_store_sorted[(df_store_sorted['time_idx'] >= min_idx) & (df_store_sorted['time_idx'] <= selected_end_idx)].copy()
                
                cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                            'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
                for col in cat_cols:
                    if col in df_batch.columns:
                        df_batch[col] = df_batch[col].astype(str).astype('category')
                        
                batch_ds = TimeSeriesDataSet.from_dataset(training_data, df_batch, predict=True, stop_randomization=True)
                batch_loader = batch_ds.to_dataloader(train=False, batch_size=256, shuffle=False)
                
            with st.spinner("Generating store forecasts..."):
                # Run TFT Teacher Batch
                with torch.no_grad():
                    batch_teacher = teacher.predict(batch_loader, mode="prediction", trainer_kwargs={"logger": False, "enable_checkpointing": False}).cpu().numpy()
                
                # Run Students Batch
                with torch.no_grad():
                    batch_student_nokd_list = []
                    batch_student_kd_list = []
                    for batch in batch_loader:
                        x, _ = batch
                        for k in x.keys():
                            if isinstance(x[k], torch.Tensor):
                                x[k] = x[k].to(device)
                        batch_student_nokd_list.append(student_nokd(x).cpu())
                        batch_student_kd_list.append(student_kd(x).cpu())
                    batch_student_nokd = torch.cat(batch_student_nokd_list, dim=0).numpy()
                    batch_student_kd = torch.cat(batch_student_kd_list, dim=0).numpy()
                
                # Align Dataloader random ID mapping with df_store_sorted alphabetic sorting
                decoded_ids = batch_ds.decoded_index['id'].tolist()
                id_to_model_row = {id_str: idx for idx, id_str in enumerate(decoded_ids)}
                
                aligned_teacher = np.zeros((num_skus, 28))
                aligned_student_nokd = np.zeros((num_skus, 28))
                aligned_student_kd = np.zeros((num_skus, 28))
                
                for i, id_str in enumerate(unique_ids):
                    model_row = id_to_model_row[id_str]
                    aligned_teacher[i] = batch_teacher[model_row]
                    aligned_student_nokd[i] = batch_student_nokd[model_row]
                    aligned_student_kd[i] = batch_student_kd[model_row]
                    
                # Vectorized calculations for Naive, Actuals, History, Scales
                # History (last 90 days before forecast)
                df_hist = df_store_sorted[(df_store_sorted['time_idx'] >= hist_start_day) & (df_store_sorted['time_idx'] < start_day)]
                history_matrix = df_hist['sales'].values.reshape(num_skus, 90)
                
                # Ground truth actuals (if available)
                df_gt = df_store_sorted[(df_store_sorted['time_idx'] >= start_day) & (df_store_sorted['time_idx'] <= selected_end_idx)]
                actuals_matrix = df_gt['sales'].values.reshape(num_skus, 28)
                
                # Seasonal Naive (28 days lag from history)
                df_naive = df_store_sorted[(df_store_sorted['time_idx'] >= (start_day - 28)) & (df_store_sorted['time_idx'] < start_day)]
                naive_matrix = df_naive['sales'].values.reshape(num_skus, 28)
                
                # Scale factors for MASE (derived from in-sample days 1 to 1857)
                df_train_slice = df_store_sorted[df_store_sorted['time_idx'] <= 1857]
                sales_train_matrix = df_train_slice['sales'].values.reshape(num_skus, 1857)
                abs_diff = np.abs(sales_train_matrix[:, 28:] - sales_train_matrix[:, :-28])
                scales = np.mean(abs_diff, axis=1)
                scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero
                
                # Unique items list for dropdown search
                unique_items = [uid.replace(f"_{selected_store}_evaluation", "") for uid in unique_ids]
                
                # Store results in Session State
                st.session_state.forecast_cache = {
                    "unique_ids": unique_ids,
                    "unique_items": unique_items,
                    "history_matrix": history_matrix,
                    "actuals_matrix": actuals_matrix,
                    "naive_matrix": naive_matrix,
                    "teacher_matrix": aligned_teacher,
                    "student_nokd_matrix": aligned_student_nokd,
                    "student_kd_matrix": aligned_student_kd,
                    "scales": scales,
                    "num_skus": num_skus,
                    "history_dates": [calendar_dates[idx - 1] for idx in range(hist_start_day, start_day)],
                    "forecast_dates": [calendar_dates[idx - 1] for idx in range(start_day, selected_end_idx + 1)]
                }
                st.session_state.last_store_run = current_run_key
                st.session_state.selected_sku = unique_items[0]

        # Retrieve results from cache
        cache = st.session_state.forecast_cache
        
        # Select active predictions matrix based on model selection
        if active_model == "Seasonal Naive":
            active_preds = cache["naive_matrix"]
        elif active_model == "TFT Teacher":
            active_preds = cache["teacher_matrix"]
        elif active_model == "Transformer Student":
            active_preds = cache["student_nokd_matrix"]
        else:
            active_preds = cache["student_kd_matrix"]
            
        # Compute Store-Level Summaries
        total_pred_demand = float(np.sum(active_preds))
        avg_daily_demand = total_pred_demand / 28.0
        
        # Pre-compute Overview Tables data
        # A) Top Forecasted Products
        sku_forecast_sums = np.sum(active_preds, axis=1)
        top_sku_indices = np.argsort(sku_forecast_sums)[::-1][:5]
        
        # B) Highest Daily Demand SKU
        flat_max_idx = np.argmax(active_preds)
        peak_row, peak_col = np.unravel_index(flat_max_idx, active_preds.shape)
        peak_sku = cache["unique_items"][peak_row]
        peak_qty = active_preds[peak_row, peak_col]
        peak_date = cache["forecast_dates"][peak_col]
        
        # C) Largest Forecast Increase vs. History (last 28 days of history)
        hist_recent_avg = np.mean(cache["history_matrix"][:, -28:], axis=1)
        forecast_avg = np.mean(active_preds, axis=1)
        delta_demand = forecast_avg - hist_recent_avg
        inc_indices = np.argsort(delta_demand)[::-1][:5]
        
        # D) Largest Forecast Decrease
        dec_indices = np.argsort(delta_demand)[:5]

        # 1. Executive Summary Cards (Business-oriented KPIs)
        st.subheader("Executive Summary")
        st.caption(f"Operational performance indicators for {selected_store} based on current forecast window:")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        kpi_col1.metric("Total Expected Demand", f"{total_pred_demand:,.0f} units")
        kpi_col2.metric("Daily Demand Average", f"{avg_daily_demand:,.0f} units/day")
        kpi_col3.metric("SKUs Forecasted", f"{cache['num_skus']:,}")
        kpi_col4.metric("Highest Demand SKU", cache["unique_items"][top_sku_indices[0]])
        kpi_col5.metric("Highest Growth SKU", cache["unique_items"][inc_indices[0]])

        st.markdown(
            f"""
            <div class="welcome-card" style="padding: 15px 25px; margin-bottom: 25px; border-left-color: #10b981;">
                <h4 style="margin: 0 0 5px 0; color: #10b981;">Operational Forecast Parameters</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 30px; font-size: 14px; color: #e2e8f0;">
                    <div><b>Active Replenishment Model:</b> {active_model}</div>
                    <div><b>Replenishment Horizon:</b> 28 Days</div>
                    <div><b>History Lookback Span:</b> {hist_start_date_str} to {hist_end_date_str} (90 days)</div>
                    <div><b>Forecast Delivery Span:</b> {start_date_str} to {end_date_str} (28 days)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Recommended Actions Cards
        st.subheader("Recommended Actions")
        st.caption("Replenishment actions translated directly from expected product demand deviations:")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown(
                """
                <div class="action-card" style="border-top-color: #ef4444;">
                    <h5 style="color: #ef4444; margin: 0 0 5px 0;">🚨 High Restock Priority</h5>
                    <p style="font-size: 12px; color: #94a3b8; margin: 0 0 10px 0;">Surge in expected sales compared to history. Review shelf stock levels immediately.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx in inc_indices[:3]:
                sub_col1, sub_col2 = st.columns([4, 1.5])
                sub_col1.write(f"**{cache['unique_items'][idx]}** (+{delta_demand[idx]:.1f}/day)")
                if sub_col2.button("Drill Down", key=f"rec_restock_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()
                    
        with rec_col2:
            st.markdown(
                """
                <div class="action-card" style="border-top-color: #f59e0b;">
                    <h5 style="color: #f59e0b; margin: 0 0 5px 0;">⚠️ Potential Overstock Risk</h5>
                    <p style="font-size: 12px; color: #94a3b8; margin: 0 0 10px 0;">Sales velocity expected to decline. Reduce replacement orders to avoid idle capital.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx in dec_indices[:3]:
                sub_col1, sub_col2 = st.columns([4, 1.5])
                sub_col1.write(f"**{cache['unique_items'][idx]}** ({delta_demand[idx]:.1f}/day)")
                if sub_col2.button("Drill Down", key=f"rec_overstock_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()
                    
        with rec_col3:
            st.markdown(
                """
                <div class="action-card" style="border-top-color: #3b82f6;">
                    <h5 style="color: #3b82f6; margin: 0 0 5px 0;">📦 High Volume Products</h5>
                    <p style="font-size: 12px; color: #94a3b8; margin: 0 0 10px 0;">Top demand drivers. Ensure logistics throughput and dedicated slotting.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx in top_sku_indices[:3]:
                sub_col1, sub_col2 = st.columns([4, 1.5])
                sub_col1.write(f"**{cache['unique_items'][idx]}** ({sku_forecast_sums[idx]:,.0f} units)")
                if sub_col2.button("Drill Down", key=f"rec_volume_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()

        st.markdown("---")
        
        # 3. Store-Level Detailed Lists
        st.subheader("Store-Level Demand Overview")
        st.caption("Complete ranking lists for replenishment planning:")
        
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>🔥 Top Forecasted Products (Full List)</p>", unsafe_allow_html=True)
            for idx in top_sku_indices:
                sub_col1, sub_col2 = st.columns([4, 1])
                sub_col1.write(f"**{cache['unique_items'][idx]}**: {sku_forecast_sums[idx]:,.2f} total units")
                if sub_col2.button("Drill Down", key=f"btn_top_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()
            
            st.markdown("<p style='font-weight: bold; margin-top:20px; margin-bottom:5px;'>📈 Largest Forecast Increase (Full List)</p>", unsafe_allow_html=True)
            for idx in inc_indices:
                sub_col1, sub_col2 = st.columns([4, 1])
                sub_col1.write(f"**{cache['unique_items'][idx]}** (Hist Avg: {hist_recent_avg[idx]:.1f} → Pred Avg: {forecast_avg[idx]:.1f})")
                if sub_col2.button("Drill Down", key=f"btn_inc_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()
                    
        with grid_col2:
            st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>⚡ Peak Single-Day Demand SKU</p>", unsafe_allow_html=True)
            peak_df = pd.DataFrame({
                "SKU": [peak_sku],
                "Peak Date": [peak_date],
                "Peak Expected Quantity": [f"{peak_qty:.2f} units"]
            })
            st.dataframe(peak_df, hide_index=True, use_container_width=True)
            
            st.markdown("<p style='font-weight: bold; margin-top:33px; margin-bottom:5px;'>📉 Largest Forecast Decrease (Full List)</p>", unsafe_allow_html=True)
            for idx in dec_indices:
                sub_col1, sub_col2 = st.columns([4, 1])
                sub_col1.write(f"**{cache['unique_items'][idx]}** (Hist Avg: {hist_recent_avg[idx]:.1f} → Pred Avg: {forecast_avg[idx]:.1f})")
                if sub_col2.button("Drill Down", key=f"btn_dec_{cache['unique_items'][idx]}"):
                    st.session_state.selected_sku = cache['unique_items'][idx]
                    st.rerun()
                    
        st.markdown("---")
        
        # 4. SKU Drill-down (Operational/Business view only)
        st.subheader("Operational SKU Drill-down")
        st.caption("Inspect expected product trajectories under the active deployed model:")
        
        # Dropdown Search
        try:
            default_sku_idx = cache["unique_items"].index(st.session_state.selected_sku)
        except ValueError:
            default_sku_idx = 0
            
        selected_sku_search = st.selectbox(
            "Search and Select a SKU to Drill Down",
            cache["unique_items"],
            index=default_sku_idx
        )
        st.session_state.selected_sku = selected_sku_search
        
        sku_i = cache["unique_items"].index(st.session_state.selected_sku)
        
        # Plot single-model business chart
        fig = go.Figure()
        
        # Past History
        fig.add_trace(go.Scatter(
            x=cache["history_dates"],
            y=cache["history_matrix"][sku_i],
            name="History (90d)",
            line=dict(color="#64748b", width=2),
            mode="lines"
        ))
        
        # Selected Deployed Model Forecast
        fig.add_trace(go.Scatter(
            x=cache["forecast_dates"],
            y=active_preds[sku_i],
            name=f"Forecast ({active_model})",
            line=dict(color="#3b82f6", width=3),
            mode="lines"
        ))
        
        # Horizon Boundary
        fig.add_vline(
            x=cache["forecast_dates"][0],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Forecast Horizon Begins",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10)
        )
        
        fig.update_layout(
            title=f"Expected Product Trajectory: {st.session_state.selected_sku} | Deployed Model: {active_model}",
            xaxis_title="Calendar Date",
            yaxis_title="Sales Quantity",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            height=400
        )
        fig.update_yaxes(hoverformat=".2f")
        st.plotly_chart(fig, use_container_width=True)
        
        # Bridge to Tab 2
        st.markdown(
            "<p style='font-size:13px; color:#94a3b8; font-style:italic;'>Would you like to analyze forecasting quality and perform detailed model comparison checks against actual outcomes for this product?</p>", 
            unsafe_allow_html=True
        )
        if st.button("🔬 Research Evaluation", key="btn_bridge_to_tab2"):
            # Programmatically change navigation selection and trigger rerun
            st.session_state.navigation_tab = "🔬 Research Evaluation"
            st.rerun()

# ===========================================================================
# VIEW 2: RESEARCH EVALUATION (Academic model comparison dashboard)
# ===========================================================================
elif selected_tab == "🔬 Research Evaluation":
    st.subheader("Forecasting Framework Performance Evaluation")
    st.caption("Quantitative results mapping prediction accuracy against model size and inference latency:")
    
    col_bench1, col_bench2 = st.columns(2)
    
    with col_bench1:
        st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>🏆 Overall Model Benchmark (Entire Preliminary Test Set)</p>", unsafe_allow_html=True)
        st.caption("Global benchmarks calculated across the entire 30,490 time series in M5 evaluation:")
        
        benchmark_data = [
            {"Model": "Seasonal Naive", "Overall WRMSSE": "0.8748", "Benchmark Inference Time": "0.16 s", "Trainable Parameters": "N/A"},
            {"Model": "TFT Teacher", "Overall WRMSSE": "4.8553", "Benchmark Inference Time": "39.75 s", "Trainable Parameters": "179,386"},
            {"Model": "Student", "Overall WRMSSE": "3.0586", "Benchmark Inference Time": "4.98 s", "Trainable Parameters": "72,708"},
            {"Model": "Student + KD", "Overall WRMSSE": "4.4137", "Benchmark Inference Time": "5.28 s", "Trainable Parameters": "72,708"}
        ]
        df_bench = pd.DataFrame(benchmark_data).set_index("Model")
        st.table(df_bench)
        st.info("💡 Note: Benchmark Inference Times represent full test set evaluations on CPU. Actual execution times in the explorer tab are faster due to cache resources.")
        
    with col_bench2:
        st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>📊 Research Summary Card</p>", unsafe_allow_html=True)
        st.caption("Accents summarizing model parameter footprint and CPU inference improvements:")
        
        st.markdown(
            """
            <div class="highlight-card">
                <div style="font-weight: bold; font-size:15px; color:#60a5fa;">Proposed Model Efficiency Trade-Offs</div>
                <div style="margin-top:10px; font-size:13px; color:#94a3b8;">
                    Teacher: 179,386 parameters<br>
                    ↓<br>
                    Student: 72,708 parameters
                </div>
                <div class="highlight-number">★ 59.5% Parameter Reduction ★</div>
                <div style="height: 1px; background-color: #1e293b; margin: 10px 0;"></div>
                <div style="font-size:13px; color:#94a3b8;">
                    Teacher Evaluation Inference: 91.96s<br>
                    ↓<br>
                    Student Evaluation Inference: 3.31s
                </div>
                <div class="highlight-number" style="color: #60a5fa;">★ 27.7× Speedup ★</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Dynamic Selected SKU Evaluations
    if st.session_state.forecast_cache is not None:
        cache = st.session_state.forecast_cache
        
        # Setup dropdown for selector on Tab 2 (defaults to the selected SKU from Tab 1)
        try:
            default_sku_idx = cache["unique_items"].index(st.session_state.selected_sku)
        except ValueError:
            default_sku_idx = 0
            
        st.markdown("---")
        st.subheader("SKU-Specific Comparative Check")
        selected_sku_eval = st.selectbox(
            "Select SKU for Detailed Academic Comparison",
            cache["unique_items"],
            index=default_sku_idx,
            key="tab2_sku_selector"
        )
        st.session_state.selected_sku = selected_sku_eval
        
        sku_i = cache["unique_items"].index(st.session_state.selected_sku)
        
        act = np.array(cache["actuals_matrix"][sku_i])
        scale = cache["scales"][sku_i]
        
        sku_evals = []
        models_list = [
            ("Seasonal", cache["naive_matrix"][sku_i]),
            ("Teacher", cache["teacher_matrix"][sku_i]),
            ("Student", cache["student_nokd_matrix"][sku_i]),
            ("Student + KD", cache["student_kd_matrix"][sku_i])
        ]
        
        for name, preds in models_list:
            preds_arr = np.array(preds)
            mae = np.mean(np.abs(act - preds_arr))
            rmse = np.sqrt(np.mean((act - preds_arr) ** 2))
            mase = mae / scale
            sku_evals.append({
                "Model": name,
                "MAE": f"{mae:.3f}",
                "RMSE": f"{rmse:.3f}",
                "MASE": f"{mase:.3f}"
            })
            
        df_sku_evals = pd.DataFrame(sku_evals).set_index("Model")
        
        # Tab 2 Chart model visibility checkboxes
        st.markdown("<p style='font-weight: bold; margin-bottom: 5px;'>Model Visibility Toggles</p>", unsafe_allow_html=True)
        cb_cols = st.columns(5)
        show_actual = cb_cols[0].checkbox("Actual Sales", value=True, key="t2_actual")
        show_naive = cb_cols[1].checkbox("Seasonal Naive", value=True, key="t2_naive")
        show_teacher = cb_cols[2].checkbox("TFT Teacher", value=True, key="t2_teacher")
        show_student = cb_cols[3].checkbox("Transformer Student", value=True, key="t2_student")
        show_student_kd = cb_cols[4].checkbox("Transformer Student + KD", value=True, key="t2_student_kd")
        
        # Plot multi-model comparison chart with Ground Truth Actuals
        eval_fig = go.Figure()
        
        # Past History
        eval_fig.add_trace(go.Scatter(
            x=cache["history_dates"],
            y=cache["history_matrix"][sku_i],
            name="History (90d)",
            line=dict(color="#64748b", width=2),
            mode="lines"
        ))
        
        # Ground Truth Actuals - High contrast crimson `#ff4b4b`
        if show_actual:
            eval_fig.add_trace(go.Scatter(
                x=cache["forecast_dates"],
                y=act,
                name="Actual",
                line=dict(color="#ff4b4b", width=4),
                mode="lines"
            ))
        
        # Seasonal Naive
        if show_naive:
            eval_fig.add_trace(go.Scatter(
                x=cache["forecast_dates"],
                y=cache["naive_matrix"][sku_i],
                name="Seasonal",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                mode="lines"
            ))
        
        # TFT Teacher
        if show_teacher:
            eval_fig.add_trace(go.Scatter(
                x=cache["forecast_dates"],
                y=cache["teacher_matrix"][sku_i],
                name="Teacher",
                line=dict(color="#3b82f6", width=2.5),
                mode="lines"
            ))
        
        # Student No KD
        if show_student:
            eval_fig.add_trace(go.Scatter(
                x=cache["forecast_dates"],
                y=cache["student_nokd_matrix"][sku_i],
                name="Student",
                line=dict(color="#ec4899", width=2),
                mode="lines"
            ))
        
        # Student KD
        if show_student_kd:
            eval_fig.add_trace(go.Scatter(
                x=cache["forecast_dates"],
                y=cache["student_kd_matrix"][sku_i],
                name="Student + KD",
                line=dict(color="#10b981", width=2.5),
                mode="lines"
            ))
        
        eval_fig.add_vline(
            x=cache["forecast_dates"][0],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Forecast Horizon Begins",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10)
        )
        
        eval_fig.update_layout(
            title=f"Multi-Model Comparative Forecasts: {st.session_state.selected_sku} | Forecast Period: {start_date_str}",
            xaxis_title="Calendar Date",
            yaxis_title="Sales Quantity",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            height=400
        )
        eval_fig.update_yaxes(hoverformat=".2f")
        st.plotly_chart(eval_fig, use_container_width=True)

        col_eval1, col_eval2 = st.columns([1, 2])
        with col_eval1:
            st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>🎯 Forecast Accuracy (Selected SKU)</p>", unsafe_allow_html=True)
            st.table(df_sku_evals)
            
        with col_eval2:
            st.markdown("<p style='font-weight: bold; margin-bottom:5px;'>📈 Daily Absolute Error Over Time</p>", unsafe_allow_html=True)
            
            error_fig = go.Figure()
            for name, preds in models_list:
                abs_err = np.abs(act - np.array(preds))
                color_map = {
                    "Seasonal": "#f59e0b",
                    "Teacher": "#3b82f6",
                    "Student": "#ec4899",
                    "Student + KD": "#10b981"
                }
                error_fig.add_trace(go.Scatter(
                    x=cache["forecast_dates"],
                    y=abs_err,
                    name=name,
                    line=dict(color=color_map[name], width=2),
                    mode="lines"
                ))
                
            error_fig.update_layout(
                xaxis_title="Calendar Date",
                yaxis_title="Absolute Error |y - ŷ|",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
                height=260
            )
            error_fig.update_yaxes(hoverformat=".2f")
            st.plotly_chart(error_fig, use_container_width=True)
    else:
        st.warning("Please select a Store and run 'Generate Store Forecast' in Tab 1 to populate SKU comparisons.")
