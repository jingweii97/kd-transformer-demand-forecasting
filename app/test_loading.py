import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle

repo_dir = r"c:\Users\jw\OneDrive - Universiti Malaya\Sem_2 Study Material\WQF7023\repo"
sys.path.append(repo_dir)

from utils.config import load_config
from pytorch_forecasting import TimeSeriesDataSet
from models.teacher import create_tft_teacher
from models.student import M5TransformerStudent

def main():
    cfg = load_config(env_name="local")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing verification on device: {device}")
    
    # 1. Paths
    metadata_path = os.path.join(repo_dir, "artifacts", "metadata", "global_metadata.pkl")
    teacher_chk = os.path.join(repo_dir, "kaggle-output", "teacher", "prelim", "best_tft_teacher.ckpt")
    student_nokd_chk = os.path.join(repo_dir, "kaggle-output", "student", "no_kd", "prelim", "best_student.ckpt")
    student_kd_chk = os.path.join(repo_dir, "kaggle-output", "student", "kd", "prelim", "best_student.ckpt")
    parquet_path = os.path.join(repo_dir, "artifacts", "data", "preprocessed_CA_1.parquet")

    # 2. Load metadata
    with open(metadata_path, 'rb') as f:
        builder = pickle.load(f)
    training_data = builder.base_dataset

    # 3. Load models
    teacher = create_tft_teacher(training_data, cfg)
    teacher_state = torch.load(teacher_chk, map_location=device, weights_only=False)["state_dict"]
    teacher.load_state_dict(teacher_state)
    teacher = teacher.to(device).eval()

    nokd_ckpt = torch.load(student_nokd_chk, map_location=device, weights_only=False)
    nokd_hparams = nokd_ckpt["hyper_parameters"]
    student_nokd = M5TransformerStudent(
        training_dataset=training_data,
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
    student_nokd = student_nokd.to(device).eval()

    kd_ckpt = torch.load(student_kd_chk, map_location=device, weights_only=False)
    kd_hparams = kd_ckpt["hyper_parameters"]
    student_kd = M5TransformerStudent(
        training_dataset=training_data,
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
    student_kd = student_kd.to(device).eval()

    # 4. Load full store data for batch mode
    df_store = pd.read_parquet(parquet_path)
    
    # Slice to ID Test window ending at day 1913
    end_day = 1913
    max_encoder_length = 90
    max_prediction_length = 28
    min_idx = end_day - max_encoder_length - max_prediction_length + 1
    
    df_batch = df_store[(df_store['time_idx'] >= min_idx) & (df_store['time_idx'] <= end_day)].copy()
    
    cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        df_batch[col] = df_batch[col].astype(str).astype('category')

    # Construct batch dataset
    batch_ds = TimeSeriesDataSet.from_dataset(training_data, df_batch, predict=True, stop_randomization=True)
    batch_loader = batch_ds.to_dataloader(train=False, batch_size=256, shuffle=False)

    # Run batch predictions (similar to evaluate_models.py)
    print("Running batch predictions...")
    with torch.no_grad():
        batch_teacher = teacher.predict(batch_loader, mode="prediction", trainer_kwargs={"logger": False, "enable_checkpointing": False}).cpu().numpy()
        
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

    # Find the index of SKU "FOODS_1_001_CA_1_evaluation" in the batch index mapping
    sku_id = "FOODS_1_001_CA_1_evaluation"
    decoded_idx = batch_ds.decoded_index
    print(f"Decoded index size: {len(decoded_idx)}")
    # Find matching row index
    sku_row_idx = decoded_idx[decoded_idx['id'] == sku_id].index[0]
    print(f"Row index for SKU {sku_id} in batch dataset: {sku_row_idx}")

    # Extract target batch predictions
    target_teacher = batch_teacher[sku_row_idx]
    target_student_nokd = batch_student_nokd[sku_row_idx]
    target_student_kd = batch_student_kd[sku_row_idx]

    # 5. Run single SKU predictions using our dashboard pipeline
    print("Running single SKU predictions...")
    df_sku = df_store[df_store['id'] == sku_id].copy()
    df_sku_sliced = df_sku[(df_sku['time_idx'] >= min_idx) & (df_sku['time_idx'] <= end_day)].copy()
    for col in cat_cols:
        df_sku_sliced[col] = df_sku_sliced[col].astype(str).astype('category')

    sku_ds = TimeSeriesDataSet.from_dataset(training_data, df_sku_sliced, predict=True, stop_randomization=True)
    sku_loader = sku_ds.to_dataloader(train=False, batch_size=1, shuffle=False)

    with torch.no_grad():
        single_teacher = teacher.predict(sku_loader, mode="prediction", trainer_kwargs={"logger": False, "enable_checkpointing": False}).cpu().numpy()[0]
        
        for batch in sku_loader:
            x, _ = batch
            for k in x.keys():
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].to(device)
            single_student_nokd = student_nokd(x).cpu().numpy()[0]
            single_student_kd = student_kd(x).cpu().numpy()[0]
            break

    # 6. Verify mathematically identical results
    teacher_diff = np.abs(target_teacher - single_teacher).max()
    nokd_diff = np.abs(target_student_nokd - single_student_nokd).max()
    kd_diff = np.abs(target_student_kd - single_student_kd).max()

    print("\n--- Alignment Verification Results ---")
    print(f"TFT Teacher difference: {teacher_diff:.8f}")
    print(f"Student No-KD difference: {nokd_diff:.8f}")
    print(f"Student KD difference: {kd_diff:.8f}")

    assert teacher_diff < 1e-4, f"Teacher predictions mismatch! Diff: {teacher_diff}"
    assert nokd_diff < 1e-4, f"Student No-KD predictions mismatch! Diff: {nokd_diff}"
    assert kd_diff < 1e-4, f"Student KD predictions mismatch! Diff: {kd_diff}"
    print("SUCCESS: Single-SKU predictions match the batch pipeline predictions perfectly!")

if __name__ == '__main__':
    main()
