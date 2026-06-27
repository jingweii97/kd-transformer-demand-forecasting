import os

def get_repo_root():
    """
    Returns the absolute path to the repository root directory.
    Assumes this file is located in <repo_root>/utils/paths.py.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path):
    """
    Resolves a relative path to an absolute path anchored at the repo root.
    If the path is already absolute, returns it as-is.
    """
    if path is None:
        return None
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(get_repo_root(), path))

def get_dataset_dir(cfg):
    """
    Resolves the preprocessing dataset directory.
    If dataset_artifacts_dir is configured, uses it. If it is missing, raises FileNotFoundError.
    Falls back to artifacts_dir otherwise.
    """
    ds_dir = getattr(cfg.environment, "dataset_artifacts_dir", None)
    if ds_dir is not None:
        abs_ds_dir = resolve_path(ds_dir)
        if not os.path.exists(abs_ds_dir):
            raise FileNotFoundError(
                f"Configured dataset_artifacts_dir does not exist at: '{abs_ds_dir}'. "
                "Please verify configuration or ensure the dataset is mounted correctly."
            )
        return abs_ds_dir
    return resolve_path(cfg.environment.artifacts_dir)

def get_experiment_dir(cfg):
    """
    Resolves the experiment inputs directory.
    If experiment_artifacts_dir is configured, uses it. If it is missing, raises FileNotFoundError.
    Falls back to artifacts_dir otherwise.
    """
    exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
    if exp_dir is not None:
        abs_exp_dir = resolve_path(exp_dir)
        if not os.path.exists(abs_exp_dir):
            raise FileNotFoundError(
                f"Configured experiment_artifacts_dir does not exist at: '{abs_exp_dir}'. "
                "Please verify configuration or ensure the experiment artifacts are mounted correctly."
            )
        return abs_exp_dir
    return resolve_path(cfg.environment.artifacts_dir)

def find_checkpoint(cfg, default_path, rel_subpath):
    """
    Resolves model checkpoint paths.
    1. If default_path explicitly exists, returns it.
    2. If experiment_artifacts_dir is configured, checks only inside it. If missing, raises FileNotFoundError.
    3. Otherwise, returns resolved default_path.
    """
    abs_default = resolve_path(default_path)
    if abs_default and os.path.exists(abs_default):
        return abs_default

    exp_dir = getattr(cfg.environment, "experiment_artifacts_dir", None)
    if exp_dir is not None:
        abs_exp_dir = resolve_path(exp_dir)
        if not os.path.exists(abs_exp_dir):
            raise FileNotFoundError(
                f"Configured experiment_artifacts_dir does not exist at: '{abs_exp_dir}'. "
                "Please check configuration or mount the experiments dataset."
            )
        
        # Check standard rel_subpath
        path1 = os.path.abspath(os.path.join(abs_exp_dir, rel_subpath))
        if os.path.exists(path1):
            return path1
            
        # Check inside outputs/ subdirectory of the experiment dir
        path2 = os.path.abspath(os.path.join(abs_exp_dir, "outputs", rel_subpath))
        if os.path.exists(path2):
            return path2
            
        raise FileNotFoundError(
            f"Checkpoint for '{rel_subpath}' not found in configured experiment_artifacts_dir at '{abs_exp_dir}'."
        )

    # Fallback/Default
    if not abs_default or not os.path.exists(abs_default):
        raise FileNotFoundError(
            f"Checkpoint file not found at default location: '{abs_default}'"
        )
    return abs_default
