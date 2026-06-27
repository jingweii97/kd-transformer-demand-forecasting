from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
from torch.utils.data import IterableDataset, DataLoader
import psutil
import os
import glob
import pickle
import pandas as pd
import numpy as np
from utils.paths import resolve_path
from data.cache import load_from_cache, STORES

class StorePartitionedDataset(IterableDataset):
    def __init__(self, base_dataset, cfg, batch_size, is_train=True, max_idx=None, predict=True, shuffle=True, partition_manager=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.batch_size = batch_size
        self.is_train = is_train
        self.max_idx = max_idx
        self.predict = predict
        self.shuffle = shuffle
        self.partition_manager = partition_manager
        
        # Determine the stores to load
        store_filter = cfg.environment.store_filter
        self.stores = [store_filter] if store_filter else list(STORES)

    def __iter__(self):
        import gc
        from utils.paths import get_dataset_dir
        train_end = self.cfg.dataset.splits.train.end
        artifacts_dir = get_dataset_dir(self.cfg)
        
        # Expose debug parameters
        max_stores = getattr(self.cfg.environment, "max_stores", None)
        max_batches_per_store = getattr(self.cfg.environment, "max_batches_per_store", None)
        
        # Shuffling partition order for training
        stores_list = list(self.stores)
        if self.is_train and self.shuffle:
            import random
            random.shuffle(stores_list)
            
        # Limit stores list if max_stores is defined
        if max_stores is not None:
            stores_list = stores_list[:max_stores]
            
        max_encoder_length = self.cfg.dataset.lookback_window
        max_prediction_length = self.cfg.dataset.prediction_window
        
        decoded_indices = []
            
        for store in stores_list:
            print(f"Streaming {'training' if self.is_train else 'evaluation'} partition for store: {store}")
            
            # Load partition Parquet cache
            df_part = load_from_cache(
                artifacts_dir=artifacts_dir,
                store_filter=store
            )
            if df_part is None:
                raise FileNotFoundError(f"Cache not found for store: {store}")
                
            if self.is_train:
                df_part_sliced = df_part[df_part['time_idx'] <= train_end].copy()
            else:
                # Slicing evaluation window: lookback + prediction ending at max_idx
                min_idx = self.max_idx - max_encoder_length - max_prediction_length + 1
                df_part_sliced = df_part[(df_part['time_idx'] >= min_idx) & (df_part['time_idx'] <= self.max_idx)].copy()
                
            del df_part
            
            # Re-convert to category columns for consistency
            cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                        'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
            for col in cat_cols:
                if col in df_part_sliced.columns:
                    df_part_sliced[col] = df_part_sliced[col].astype(str).astype('category')
                    
            if len(df_part_sliced) == 0:
                continue
                
            # Construct dataset using PyTorch Forecasting API exactly as intended
            if self.is_train:
                part_ds = TimeSeriesDataSet.from_dataset(self.base_dataset, df_part_sliced)
            else:
                part_ds = TimeSeriesDataSet.from_dataset(
                    self.base_dataset,
                    df_part_sliced,
                    predict=self.predict,
                    stop_randomization=True
                )
            del df_part_sliced
            
            # Collect decoded index metadata
            if self.partition_manager is not None:
                decoded_indices.append(part_ds.decoded_index)
            
            # Create partition-level DataLoader
            # num_workers comes from cfg so the value is environment-appropriate
            # (0 = local/Windows, 2 = Kaggle, 4 = DICC). The outer DataLoader
            # wrappers in StorePartitionManager intentionally keep num_workers=0
            # because wrapping an IterableDataset with workers > 0 duplicates data.
            part_loader = part_ds.to_dataloader(
                train=self.is_train,
                batch_size=self.batch_size,
                shuffle=self.is_train,
                num_workers=self.cfg.environment.num_workers
            )
            
            # Yield batches directly
            batch_count = 0
            for batch in part_loader:
                yield batch
                batch_count += 1
                if max_batches_per_store is not None and batch_count >= max_batches_per_store:
                    print(f"Debug Mode: reached max batches per store limit ({max_batches_per_store})")
                    break
                    
            # Memory cleanup after each partition
            del part_loader
            del part_ds
            gc.collect()
            
        # Concatenate and save decoded index in partition manager
        if self.partition_manager is not None and len(decoded_indices) > 0:
            self.partition_manager._decoded_index = pd.concat(decoded_indices, ignore_index=True)

class StoreMetadataBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.categorical_encoders = {}
        self.target_normalizer = None
        self.base_dataset = None

    def fit(self):
        print("Fitting global categorical encoders from complete known domains...")
        # 1. Fit categorical encoders globally
        input_dir_abs = resolve_path(self.cfg.environment.input_dir)
        
        # Read the calendar event classes
        df_cal = pd.read_csv(os.path.join(input_dir_abs, "calendar.csv"))
        
        # Read the canonical wide metadata file sales_train_evaluation.csv
        df_sales_meta = pd.read_csv(
            os.path.join(input_dir_abs, "sales_train_evaluation.csv"),
            usecols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        )
        
        # Complete known category domains mapping
        cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
        
        unique_vals = {}
        # Static identifiers from sales_train_evaluation.csv
        for col in ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
            unique_vals[col] = df_sales_meta[col].astype(str).unique()
        
        # Calendar/event features from calendar.csv
        unique_vals["event_name_1"] = df_cal["event_name_1"].fillna("None").unique()
        unique_vals["event_type_1"] = df_cal["event_type_1"].fillna("None").unique()
        unique_vals["weekday"] = df_cal["weekday"].unique()
        unique_vals["month"] = df_cal["month"].astype(str).unique()
        unique_vals["year"] = df_cal["year"].astype(str).unique()
        
        # Fit NaNLabelEncoders
        self.categorical_encoders = {}
        for col in cat_cols:
            sorted_vals = sorted(list(unique_vals[col]))
            encoder = NaNLabelEncoder(add_nan=True)
            encoder.fit(np.array(sorted_vals, dtype=object))
            self.categorical_encoders[col] = encoder
            
        del df_sales_meta
        del df_cal
        
        # 2. Fit target normalizer using training-period target values only (d_1-d_1857)
        # Required columns derived dynamically from cfg.dataset.group_ids
        print("Fitting target normalizer on complete training period...")
        group_cols = self.cfg.dataset.group_ids
        target_col = self.cfg.dataset.target
        train_end = self.cfg.dataset.splits.train.end
        
        from utils.paths import get_dataset_dir
        artifacts_dir = get_dataset_dir(self.cfg)
        cache_dir = os.path.join(artifacts_dir, "data")
        files = sorted(glob.glob(os.path.join(cache_dir, "preprocessed_*.parquet")))
        if not files:
            raise FileNotFoundError(
                "No cached Parquet files found. Please run prepare_dataset.py first."
            )
            
        norm_columns = [target_col] + group_cols + ['time_idx']
        norm_dfs = []
        for f in files:
            part_df = pd.read_parquet(f, engine='pyarrow', columns=norm_columns)
            part_df = part_df[part_df['time_idx'] <= train_end]
            for col in group_cols:
                part_df[col] = part_df[col].astype(str).astype('category')
            norm_dfs.append(part_df)
        df_norm = pd.concat(norm_dfs, ignore_index=True)
        
        self.target_normalizer = GroupNormalizer(groups=group_cols, transformation="softplus")
        self.target_normalizer.fit(df_norm[target_col], df_norm)
        del df_norm
        
        # 3. Instantiate base TimeSeriesDataSet structure using minimal metadata dataset
        print("Instantiating base TimeSeriesDataSet structure using minimal metadata dataset...")
        # lookback + prediction window length
        max_encoder_length = self.cfg.dataset.lookback_window
        max_prediction_length = self.cfg.dataset.prediction_window
        min_required_len = max_encoder_length + max_prediction_length
        
        # Load only min_required_len days of all stores
        dfs = [pd.read_parquet(f, engine='pyarrow', filters=[('time_idx', '<=', min_required_len)]) for f in files]
        df_subset = pd.concat(dfs, ignore_index=True)
        
        for col in cat_cols:
            if col in df_subset.columns:
                df_subset[col] = df_subset[col].astype(str).astype('category')
                
        self.base_dataset = TimeSeriesDataSet(
            df_subset,
            time_idx="time_idx",
            target=target_col,
            group_ids=group_cols,
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=self.cfg.dataset.features.static_categoricals,
            time_varying_known_categoricals=self.cfg.dataset.features.time_varying_known_categoricals,
            time_varying_known_reals=self.cfg.dataset.features.time_varying_known_reals,
            time_varying_unknown_reals=self.cfg.dataset.features.time_varying_unknown_reals,
            target_normalizer=self.target_normalizer,
            categorical_encoders=self.categorical_encoders,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

class StorePartitionManager:
    def __init__(self, base_dataset, cfg):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self._decoded_index = None

    def train_dataloader(self, batch_size):
        dataset_iter = StorePartitionedDataset(
            base_dataset=self.base_dataset,
            cfg=self.cfg,
            batch_size=batch_size,
            is_train=True,
            shuffle=True,
            partition_manager=self
        )
        return DataLoader(dataset_iter, batch_size=None, num_workers=0)

    def val_dataloader(self, batch_size, max_idx):
        dataset_iter = StorePartitionedDataset(
            base_dataset=self.base_dataset,
            cfg=self.cfg,
            batch_size=batch_size,
            is_train=False,
            max_idx=max_idx,
            predict=True,
            shuffle=False,
            partition_manager=self
        )
        return DataLoader(dataset_iter, batch_size=None, num_workers=0)

    def test_dataloader(self, batch_size, max_idx, predict=True):
        dataset_iter = StorePartitionedDataset(
            base_dataset=self.base_dataset,
            cfg=self.cfg,
            batch_size=batch_size,
            is_train=False,
            max_idx=max_idx,
            predict=predict,
            shuffle=False,
            partition_manager=self
        )
        return DataLoader(dataset_iter, batch_size=None, num_workers=0)

    def get_decoded_index(self):
        """Expose immutable-like copy of decoded_index metadata after prediction/iteration."""
        if self._decoded_index is None:
            return None
        return self._decoded_index.copy()

def build_timeseries_dataset(df, cfg, is_train=True, training_dataset=None, max_idx=None, predict=True):
    """
    Constructs and returns a standard TimeSeriesDataSet object.
    For training: constructs/loads StoreMetadataBuilder and returns base_dataset.
    For evaluation: slices df and uses TimeSeriesDataSet.from_dataset (for backwards compatibility).
    """
    if is_train:
        from utils.paths import get_dataset_dir
        dataset_dir = get_dataset_dir(cfg)
        metadata_path = os.path.join(dataset_dir, "metadata", "global_metadata.pkl")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Global metadata cache file not found at: '{metadata_path}'. "
                "Ensure prepare_dataset.py has been run to generate the cache and metadata "
                "before starting model training."
            )
            
        print(f"Loading global metadata builder from cache: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            builder = pickle.load(f)
            
        # Rebind configuration
        builder.cfg = cfg
        builder.base_dataset.cfg = cfg
        return builder.base_dataset
    else:
        assert training_dataset is not None, "training_dataset must be provided."
        assert max_idx is not None, "max_idx must be provided."
        max_encoder_length = cfg.dataset.lookback_window
        max_prediction_length = cfg.dataset.prediction_window
        
        min_idx = max_idx - max_encoder_length - max_prediction_length + 1
        df_eval = df[(df['time_idx'] >= min_idx) & (df['time_idx'] <= max_idx)].copy()
        
        cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
        for col in cat_cols:
            if col in df_eval.columns:
                df_eval[col] = df_eval[col].astype(str).astype('category')
        
        return TimeSeriesDataSet.from_dataset(
            training_dataset,
            df_eval,
            predict=predict,
            stop_randomization=True
        )
