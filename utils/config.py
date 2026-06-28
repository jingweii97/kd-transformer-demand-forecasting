import os
import yaml
from utils.paths import resolve_path

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, name):
        raise AttributeError(f"No such config parameter: {name}")

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

def load_config(env_name="local", experiment_name=None, config_dir="configs"):
    """
    Loads dataset, evaluation, teacher, student, and environment configurations,
    merges them into a Config namespace object, and performs schema validation.
    Supports optional experiment configuration overrides.
    """
    config_dir_abs = resolve_path(config_dir)
    
    # Load separate config files
    dataset_path = os.path.join(config_dir_abs, "dataset.yaml")
    evaluation_path = os.path.join(config_dir_abs, "evaluation.yaml")
    teacher_path = os.path.join(config_dir_abs, "teacher.yaml")
    student_path = os.path.join(config_dir_abs, "student.yaml")
    
    env_file = f"{env_name}.yaml"
    env_path = os.path.join(config_dir_abs, "environment", env_file)
    
    # Check if files exist
    for path in [dataset_path, evaluation_path, teacher_path, student_path, env_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
    with open(dataset_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f) or {}
    with open(evaluation_path, 'r') as f:
        evaluation_cfg = yaml.safe_load(f) or {}
    with open(teacher_path, 'r') as f:
        teacher_cfg = yaml.safe_load(f) or {}
    with open(student_path, 'r') as f:
        student_cfg = yaml.safe_load(f) or {}
    with open(env_path, 'r') as f:
        env_cfg = yaml.safe_load(f) or {}
        
    # Merge into a single dict structure
    merged_dict = {
        "dataset": dataset_cfg,
        "evaluation": evaluation_cfg,
        "teacher": teacher_cfg,
        "student": student_cfg,
        "environment": env_cfg
    }
    
    # Apply experiment overrides explicitly if requested
    if experiment_name:
        exp_path = os.path.join(config_dir_abs, "experiment", f"{experiment_name}.yaml")
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"Experiment configuration file not found: {exp_path}")
        with open(exp_path, 'r') as f:
            exp_cfg = yaml.safe_load(f) or {}
            
        if "store_filter" in exp_cfg:
            merged_dict["environment"]["store_filter"] = exp_cfg["store_filter"]
        if "window_stride" in exp_cfg:
            merged_dict["dataset"]["window_stride"] = exp_cfg["window_stride"]
        if "teacher" in exp_cfg and isinstance(exp_cfg["teacher"], dict):
            merged_dict["teacher"].update(exp_cfg["teacher"])
        if "student" in exp_cfg and isinstance(exp_cfg["student"], dict):
            merged_dict["student"].update(exp_cfg["student"])
            
    cfg = Config(merged_dict)
    validate_config(cfg)
    return cfg

def validate_config(cfg):
    """
    Validates configuration schema, checking types and missing fields.
    """
    # 1. Dataset Validation
    if not hasattr(cfg, "dataset"):
        raise ValueError("Config missing 'dataset' section.")
    
    required_dataset = ["target", "group_ids", "lookback_window", "prediction_window", "splits", "features"]
    for key in required_dataset:
        if not hasattr(cfg.dataset, key):
            raise ValueError(f"Dataset config missing required field: {key}")
            
    if not isinstance(cfg.dataset.lookback_window, int):
        raise TypeError(f"lookback_window must be int, got {type(cfg.dataset.lookback_window)}")
    if not isinstance(cfg.dataset.prediction_window, int):
        raise TypeError(f"prediction_window must be int, got {type(cfg.dataset.prediction_window)}")
    if not isinstance(cfg.dataset.group_ids, list):
        raise TypeError("group_ids must be a list of strings")
        
    # Splits validation
    required_splits = ["train", "validation", "id_test", "ood_test"]
    for s in required_splits:
        if not hasattr(cfg.dataset.splits, s):
            raise ValueError(f"Dataset splits missing: {s}")
        split_obj = getattr(cfg.dataset.splits, s)
        if not hasattr(split_obj, "start") or not hasattr(split_obj, "end"):
            raise ValueError(f"Split {s} must have 'start' and 'end' parameters")
        if not isinstance(split_obj.start, int) or not isinstance(split_obj.end, int):
            raise TypeError(f"Split {s} bounds must be integers")

    # Features validation
    required_features = [
        "static_categoricals", 
        "time_varying_known_categoricals", 
        "time_varying_known_reals", 
        "time_varying_unknown_reals"
    ]
    for feat in required_features:
        if not hasattr(cfg.dataset.features, feat):
            raise ValueError(f"Dataset features missing: {feat}")
        if not isinstance(getattr(cfg.dataset.features, feat), list):
            raise TypeError(f"Dataset feature group '{feat}' must be a list")

    # 2. Environment Validation
    if not hasattr(cfg, "environment"):
        raise ValueError("Config missing 'environment' section.")
        
    required_env = ["input_dir", "artifacts_dir", "outputs_dir", "accelerator", "devices", "num_workers", "precision"]
    for key in required_env:
        if not hasattr(cfg.environment, key):
            raise ValueError(f"Environment config missing required field: {key}")
            
    if not isinstance(cfg.environment.num_workers, int):
        raise TypeError(f"num_workers must be int, got {type(cfg.environment.num_workers)}")
        
    # 3. Student / Teacher Validation
    if not hasattr(cfg, "teacher"):
        raise ValueError("Config missing 'teacher' section.")
    if not hasattr(cfg, "student"):
        raise ValueError("Config missing 'student' section.")
        
    if hasattr(cfg.student, "alpha"):
        if not isinstance(cfg.student.alpha, (int, float)):
            raise TypeError("student.alpha must be float or int")
        if not (0.0 <= cfg.student.alpha <= 1.0):
            raise ValueError("student.alpha must be in range [0.0, 1.0]")

def save_config(cfg, filepath):
    """
    Saves a Config object to a YAML file.
    """
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    d = cfg.to_dict()
    with open(filepath, 'w') as f:
        yaml.safe_dump(d, f, default_flow_style=False)

def get_git_commit_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "Unknown"

def save_metadata(output_dir, seed, checkpoint_path=None, metrics=None, additional_fields=None):
    """
    Saves a metadata.json file to the specified directory for experiment traceability.
    """
    import json
    import datetime
    import torch
    
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
    }
    if checkpoint_path:
        metadata["checkpoint_path"] = os.path.abspath(checkpoint_path)
    if metrics:
        metadata["metrics"] = metrics
    if additional_fields:
        metadata.update(additional_fields)
        
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved experiment metadata to {metadata_path}")

