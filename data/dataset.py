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
from utils.paths import get_repo_root
from data.origin_sampling import load_training_origins
from data.cache import load_from_cache, STORES, resolve_stores
import torch

class StorePartitionedDataset(IterableDataset):
    def __init__(self, base_dataset, cfg, batch_size, is_train=True, max_idx=None, predict=True, shuffle=True, partition_manager=None, exp_name=None, series_coefficients=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.batch_size = batch_size
        self.is_train = is_train
        self.max_idx = max_idx
        self.predict = predict
        self.shuffle = shuffle
        self.partition_manager = partition_manager
        self.exp_name = exp_name
        self.series_coefficients = series_coefficients
        
        # Determine the stores to load
        self.stores = resolve_stores(cfg.environment.store_filter)

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
        explicit_validation_origins = None
        if not self.is_train:
            # Phase-balanced v2 validation uses one full panel from each of the
            # seven scheduled forecast starts. Legacy experiments keep the
            # single-origin predict=True validation behaviour.
            explicit_validation_origins = load_training_origins(
                getattr(self.cfg.dataset, "validation_origin_sampling", None),
                repo_root=get_repo_root(),
            )
            if explicit_validation_origins is not None:
                if self.max_idx is None:
                    raise ValueError("Explicit validation origins require max_idx")
                if (
                    explicit_validation_origins[0] - max_encoder_length < 1
                    or explicit_validation_origins[-1] + max_prediction_length - 1 > self.max_idx
                ):
                    raise ValueError(
                        "Explicit validation origins are outside the validation loader range"
                    )
        
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

            # Load soft targets if running KD during training
            store_soft_targets = None
            global_to_local = None
            if self.is_train and getattr(self.cfg.student, "kd", False) and self.exp_name is not None:
                soft_target_exp_name = getattr(
                    self.cfg.student, "soft_targets_exp_name", None
                ) or self.exp_name
                # First check whether cfg.student.soft_targets_path exists and is a directory
                soft_targets_path = getattr(self.cfg.student, "soft_targets_path", None)
                if soft_targets_path and os.path.isdir(resolve_path(soft_targets_path)):
                    st_path = os.path.join(resolve_path(soft_targets_path), f"{soft_target_exp_name}_{store}.pt")
                else:
                    # Resolve soft targets path for the current store using fallback logic
                    exp_dir = getattr(self.cfg.environment, "experiment_artifacts_dir", None)
                    if exp_dir is not None:
                        from utils.paths import get_experiment_dir
                        exp_art_dir = get_experiment_dir(self.cfg)
                        path1 = os.path.join(exp_art_dir, "soft_targets", f"{soft_target_exp_name}_{store}.pt")
                        path2 = os.path.join(exp_art_dir, "outputs", "soft_targets", f"{soft_target_exp_name}_{store}.pt")
                        if os.path.exists(path1):
                            st_path = path1
                        else:
                            st_path = path2
                    else:
                        artifacts_dir = resolve_path(self.cfg.environment.artifacts_dir)
                        st_path = os.path.join(artifacts_dir, "soft_targets", f"{soft_target_exp_name}_{store}.pt")
                
                print(f"Loading pre-computed teacher forecasts for store {store} from: {st_path}")
                if os.path.exists(st_path):
                    st_data = torch.load(st_path, map_location="cpu", weights_only=False)
                    unique_groups = st_data["unique_groups"]
                    store_soft_targets = st_data["tensor"]
                    
                    # Create global to local mapping mapping global group_id to store-local index
                    num_total_series = len(self.base_dataset._categorical_encoders['id'].classes_)
                    global_to_local = torch.full((num_total_series,), -1, dtype=torch.long)
                    global_to_local[torch.tensor(unique_groups, dtype=torch.long)] = torch.arange(len(unique_groups))
                else:
                    raise FileNotFoundError(f"Soft targets file not found at {st_path}")
                
            if self.is_train:
                df_part_sliced = df_part[df_part['time_idx'] <= train_end].copy()
            else:
                # Retain the history required by the earliest scheduled phase.
                min_idx = (
                    explicit_validation_origins[0] - max_encoder_length
                    if explicit_validation_origins is not None
                    else self.max_idx - max_encoder_length - max_prediction_length + 1
                )
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

            use_wrmsse_weights = self.series_coefficients is not None
            if use_wrmsse_weights:
                mapped = df_part_sliced["id"].astype(str).map(self.series_coefficients)
                if mapped.isna().any():
                    missing = df_part_sliced.loc[mapped.isna(), "id"].astype(str).unique()[:5]
                    raise KeyError(f"Missing WRMSSE-informed coefficient(s): {missing.tolist()}")
                df_part_sliced["wrmsse_informed_coefficient"] = mapped.astype("float32")
                
            # Construct dataset using PyTorch Forecasting API exactly as intended
            if self.is_train:
                dataset_kwargs = {"weight": "wrmsse_informed_coefficient"} if use_wrmsse_weights else {}
                part_ds = TimeSeriesDataSet.from_dataset(self.base_dataset, df_part_sliced, **dataset_kwargs)
                
                # Apply an explicit controlled schedule when configured. This
                # is used by phase-balanced v2; otherwise preserve the legacy
                # modulo-stride behaviour exactly.
                # Uses the official filter() API on time_idx_first_prediction — the decoder
                # start time index exposed by decoded_index — so subsampling is aligned to
                # consistent calendar positions across all series rather than arbitrary row
                # offsets.  stride=1 is a no-op (all windows retained).
                explicit_origins = load_training_origins(
                    getattr(self.cfg.dataset, "training_origin_sampling", None),
                    repo_root=get_repo_root(),
                )
                if explicit_origins is not None:
                    allowed = set(explicit_origins)
                    part_ds = part_ds.filter(
                        lambda idx: idx["time_idx_first_prediction"].isin(allowed)
                    )
                    observed = sorted(part_ds.decoded_index["time_idx_first_prediction"].unique().tolist())
                    if observed != explicit_origins:
                        raise AssertionError(
                            "Explicit training origins were not realized exactly; "
                            f"expected={explicit_origins[:3]}...{explicit_origins[-3:]}, "
                            f"observed={observed[:3]}...{observed[-3:]}"
                        )
                else:
                    stride = getattr(self.cfg.dataset, "window_stride", 1)
                    if stride > 1:
                        time_col = "time_idx_first_prediction"
                        part_ds = part_ds.filter(
                            lambda idx: idx[time_col] % stride == 0
                        )

            else:
                part_ds = TimeSeriesDataSet.from_dataset(
                    self.base_dataset,
                    df_part_sliced,
                    predict=False if explicit_validation_origins is not None else self.predict,
                    stop_randomization=True,
                    **({"weight": "wrmsse_informed_coefficient"} if use_wrmsse_weights else {})
                )
                if explicit_validation_origins is not None:
                    allowed = set(explicit_validation_origins)
                    part_ds = part_ds.filter(
                        lambda idx: idx["time_idx_first_prediction"].isin(allowed)
                    )
                    observed = sorted(
                        part_ds.decoded_index["time_idx_first_prediction"].unique().tolist()
                    )
                    if observed != explicit_validation_origins:
                        raise AssertionError(
                            "Explicit validation origins were not realized exactly; "
                            f"expected={explicit_validation_origins}, observed={observed}"
                        )
            del df_part_sliced
            
            # Collect decoded index metadata
            if self.partition_manager is not None:
                decoded_indices.append(part_ds.decoded_index)
            
            part_loader = part_ds.to_dataloader(
                train=self.is_train,
                batch_size=self.batch_size,
                shuffle=self.is_train,
                num_workers=self.cfg.environment.num_workers
            )
            
            print(f"Store: {store} | TimeSeriesDataSet len: {len(part_ds)} | DataLoader len: {len(part_loader)}")
            
            # Yield batches directly
            batch_count = 0
            for batch in part_loader:
                x, y = batch
                if use_wrmsse_weights:
                    if y[1] is None:
                        raise AssertionError("WRMSSE-informed batch is missing its coefficient tensor")
                    x["wrmsse_informed_coefficient"] = y[1][:, 0]
                if store_soft_targets is not None:
                    start_times = x['decoder_time_idx'][:, 0].long()

                    # ``x['groups']`` contains PyTorch Forecasting's internal
                    # ``__group_id__id`` codes.  Those codes are store-local
                    # for a partitioned dataset, whereas the soft-target cache
                    # is keyed by the public, globally encoded ``id`` values
                    # written by generate_soft_targets.py.  Decode the batch
                    # identity through the dataset before looking up the cache;
                    # never use the internal codes as global cache indices.
                    batch_index = part_ds.x_to_index(x)
                    batch_ids = batch_index["id"].astype(str).to_numpy()
                    global_group_ids = self.base_dataset._categorical_encoders["id"].transform(batch_ids)
                    global_group_ids = torch.as_tensor(global_group_ids, dtype=torch.long)

                    if len(global_group_ids) != len(start_times):
                        raise AssertionError(
                            "Soft-target lookup identity count does not match batch size: "
                            f"{len(global_group_ids)} != {len(start_times)}"
                        )
                    if (global_group_ids < 0).any() or (global_group_ids >= len(global_to_local)).any():
                        raise KeyError(
                            f"Soft-target global series code is out of bounds for store {store}"
                        )
                    if (start_times < 0).any() or (start_times >= store_soft_targets.shape[1]).any():
                        raise IndexError(
                            f"Soft-target forecast origin is out of bounds for store {store}: "
                            f"valid range is [0, {store_soft_targets.shape[1] - 1}]"
                        )

                    local_group_ids = global_to_local[global_group_ids]
                    if (local_group_ids < 0).any():
                        missing = batch_ids[local_group_ids.numpy() < 0][:5].tolist()
                        raise KeyError(
                            f"Soft-target cache has no mapping for {len(missing)} or more series in store "
                            f"{store}; examples: {missing}"
                        )
                    teacher_preds = store_soft_targets[local_group_ids, start_times]
                    if teacher_preds.shape[-1] != max_prediction_length:
                        raise AssertionError(
                            f"Soft-target horizon mismatch for store {store}: "
                            f"expected {max_prediction_length}, got {teacher_preds.shape[-1]}"
                        )
                    if not torch.isfinite(teacher_preds).all():
                        raise ValueError(
                            f"Soft-target cache contains missing or non-finite values for store {store}"
                        )
                    x['soft_targets'] = teacher_preds
                
                yield x, y
                batch_count += 1
                if max_batches_per_store is not None and batch_count >= max_batches_per_store:
                    print(f"Debug Mode: reached max batches per store limit ({max_batches_per_store})")
                    break
                    
            # Memory cleanup after each partition
            del part_loader
            del part_ds
            if store_soft_targets is not None:
                del store_soft_targets
                del global_to_local
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

    def fit(self, parquet_dir):
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
        
        cache_dir = resolve_path(parquet_dir)
        files = sorted(glob.glob(os.path.join(cache_dir, "preprocessed_*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"No cached Parquet files found at: '{cache_dir}'"
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
    def __init__(self, base_dataset, cfg, exp_name=None, series_coefficients=None):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.exp_name = exp_name
        self._decoded_index = None
        self.series_coefficients = series_coefficients

    def train_dataloader(self, batch_size):
        dataset_iter = StorePartitionedDataset(
            base_dataset=self.base_dataset,
            cfg=self.cfg,
            batch_size=batch_size,
            is_train=True,
            shuffle=True,
            partition_manager=self,
            exp_name=self.exp_name,
            series_coefficients=self.series_coefficients,
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
            partition_manager=self,
            exp_name=self.exp_name,
            series_coefficients=self.series_coefficients,
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
            partition_manager=self,
            exp_name=self.exp_name,
            series_coefficients=self.series_coefficients,
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
