import os
import glob
import yaml
import pandas as pd
from utils.paths import resolve_path

# ---------------------------------------------------------------------------
# Store registry — single source of truth for M5 store identifiers.
# Hardcoded for determinism; do not derive dynamically from raw CSV.
# ---------------------------------------------------------------------------
STORES = (
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
)

# ---------------------------------------------------------------------------
# Feature version — read from configs/feature_cache.yaml at import time.
# Increment the YAML value whenever feature engineering logic changes.
# All per-store caches with a mismatched or missing version will be
# automatically reprocessed by prepare_dataset.py.
# ---------------------------------------------------------------------------
def _load_feature_version():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "feature_cache.yaml"
    )
    with open(cfg_path) as f:
        return yaml.safe_load(f)["feature_version"]

FEATURE_VERSION = _load_feature_version()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def get_cache_path(artifacts_dir, store_filter):
    """
    Returns the absolute path to the Parquet cache file for a given store.
    """
    suffix = f"_{store_filter}" if store_filter else "_full"
    cache_dir = os.path.join(resolve_path(artifacts_dir), "data")
    return os.path.join(cache_dir, f"preprocessed{suffix}.parquet")


def _get_version_path(artifacts_dir, store_filter):
    """Returns the path to the .version sidecar file for a given store cache."""
    suffix = f"_{store_filter}" if store_filter else "_full"
    cache_dir = os.path.join(resolve_path(artifacts_dir), "data")
    return os.path.join(cache_dir, f"preprocessed{suffix}.version")


# ---------------------------------------------------------------------------
# Cache validity check
# ---------------------------------------------------------------------------
def is_cache_valid(artifacts_dir, store_filter):
    """
    Returns True only if:
      1. The Parquet file exists, AND
      2. The .version sidecar exists and matches the current FEATURE_VERSION.

    Returns False if the cache is missing, the sidecar is missing, or the
    stored version mismatches (i.e., features have changed since last run).
    """
    if not os.path.exists(get_cache_path(artifacts_dir, store_filter)):
        return False
    version_path = _get_version_path(artifacts_dir, store_filter)
    if not os.path.exists(version_path):
        return False  # Old cache without version marker — treat as stale
    with open(version_path) as f:
        stored = int(f.read().strip())
    return stored == FEATURE_VERSION


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_to_cache(df, artifacts_dir, store_filter):
    """
    Saves a DataFrame to Parquet cache format and writes a FEATURE_VERSION
    sidecar file alongside it.
    """
    cache_path = get_cache_path(artifacts_dir, store_filter)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving preprocessed dataset to: {cache_path}")
    df.to_parquet(cache_path, index=False, engine='pyarrow')
    # Write version marker so future runs can validate freshness
    with open(_get_version_path(artifacts_dir, store_filter), 'w') as f:
        f.write(str(FEATURE_VERSION))
    return cache_path


# ---------------------------------------------------------------------------
# Load — single store
# ---------------------------------------------------------------------------
def load_from_cache(artifacts_dir, store_filter):
    """
    Loads preprocessed DataFrame from Parquet cache for a single store.
    Returns None if the cache file does not exist.
    Raises RuntimeError if the cache exists but its feature version is stale.
    """
    cache_path = get_cache_path(artifacts_dir, store_filter)
    if not os.path.exists(cache_path):
        return None
    # Guard: never silently load a cache produced by a different feature_version.
    if not is_cache_valid(artifacts_dir, store_filter):
        version_path = _get_version_path(artifacts_dir, store_filter)
        stored_version = "missing"
        if os.path.exists(version_path):
            with open(version_path) as _vf:
                stored_version = _vf.read().strip()
        raise RuntimeError(
            f"Cache for store '{store_filter or 'full'}' is stale and cannot be loaded.\n"
            f"  Stored feature_version : {stored_version}\n"
            f"  Expected feature_version: {FEATURE_VERSION}\n"
            "Run prepare_dataset.py to regenerate the cache."
        )
    print(f"Loading preprocessed dataset from: {cache_path}")
    df = pd.read_parquet(cache_path, engine='pyarrow')
    # Re-convert to pandas category type (Parquet round-trip may lose this)
    cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


# ---------------------------------------------------------------------------
# Load — all stores (full dataset)
# ---------------------------------------------------------------------------
def load_all_from_cache(artifacts_dir):
    """
    Loads and concatenates all per-store Parquet files from artifacts/data/.
    Used for full-dataset (store_filter = "") runs in Phase 2.

    Memory behaviour:
      - All per-store DataFrames are read into a list, then pd.concat'd once.
      - `del dfs` releases the input list immediately after concat, before the
        category re-cast. Peak ≈ 2× the final concatenated DataFrame size.

    API constraint (documented):
      TimeSeriesDataSet requires the full DataFrame in RAM at construction
      time — pytorch-forecasting has no native support for streaming or
      incremental dataset construction. If that changes, this function is the
      correct hook to return Iterator[DataFrame] instead.

    Returns None if no cached Parquet files are found.
    """
    cache_dir = os.path.join(resolve_path(artifacts_dir), "data")
    files = sorted(glob.glob(os.path.join(cache_dir, "preprocessed_*.parquet")))
    if not files:
        return None

    # Guard: validate every version sidecar before loading any data.
    # A single stale file in a multi-store run would corrupt the concatenated
    # DataFrame; checking all upfront produces a clear, actionable error.
    stale_files = []
    for _f in files:
        _version_path = os.path.splitext(_f)[0] + ".version"
        if not os.path.exists(_version_path):
            stale_files.append(f"{os.path.basename(_f)}: missing version sidecar")
            continue
        with open(_version_path) as _vf:
            _stored = int(_vf.read().strip())
        if _stored != FEATURE_VERSION:
            stale_files.append(
                f"{os.path.basename(_f)}: version {_stored} != expected {FEATURE_VERSION}"
            )
    if stale_files:
        _details = "\n  ".join(stale_files)
        raise RuntimeError(
            f"Stale or missing version sidecars detected:\n  {_details}\n"
            "Run prepare_dataset.py to regenerate stale caches."
        )

    print(f"Loading {len(files)} store Parquet file(s)...")
    dfs = [pd.read_parquet(f, engine='pyarrow') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs  # release input frames before category re-cast

    cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'weekday', 'month', 'year', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Guard: TimeSeriesDataSet requires per-group monotonic time_idx.
    # Each per-store Parquet is already sorted by (id, time_idx) from
    # preprocessing.py (line 50), so concat yields group-contiguous sorted
    # blocks. This assertion verifies that invariant is preserved.
    assert (
        df.groupby("id")["time_idx"].is_monotonic_increasing.all()
    ), (
        "time_idx is not monotonically increasing per group after concat. "
        "TimeSeriesDataSet construction will fail."
    )

    return df


def resolve_stores(store_filter):
    """
    Resolves a store_filter string (e.g. 'CA_1', 'TX', or '') into a list of store IDs.
    - If store_filter is empty/None: returns all STORES.
    - If store_filter is a specific store name in STORES: returns [store_filter].
    - If store_filter matches a state prefix (CA, TX, WI): returns matching stores.
    """
    if not store_filter:
        return list(STORES)
    if store_filter in STORES:
        return [store_filter]
    matched = [s for s in STORES if s.startswith(store_filter)]
    if matched:
        return matched
    raise ValueError(f"Unknown store or state filter: {store_filter}")


def load_dataset_from_cache(artifacts_dir, store_filter):
    """
    Loads preprocessed DataFrame for resolved stores. Concatenates them for
    base dataset/encoder fitting.
    """
    stores = resolve_stores(store_filter)
    if len(stores) == len(STORES):
        return load_all_from_cache(artifacts_dir)
        
    dfs = []
    for s in stores:
        part_df = load_from_cache(artifacts_dir, s)
        if part_df is not None:
            dfs.append(part_df)
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    assert (
        df.groupby("id")["time_idx"].is_monotonic_increasing.all()
    ), "time_idx is not monotonically increasing per group after concat."
    return df

