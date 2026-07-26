import os
import shutil

target_dir = "id-ood-analysis"
os.makedirs(target_dir, exist_ok=True)

files_to_move = [
    "m5_28day_timeline_summary.csv",
    "m5_temporal_profile.png",
    "candidate_protocols.csv",
    "protocol_analysis_summary.md",
    "event_window_candidates_all_history.csv",
    "event_window_candidates_future_eligible.csv",
    "event_window_summary.csv",
    "feasible_temporal_designs.csv",
    "selected_scenario_characteristics.csv",
    "protocol_summary.md"
]

moved = []
for f in files_to_move:
    if os.path.exists(f):
        dest = os.path.join(target_dir, f)
        shutil.move(f, dest)
        moved.append(f)
        print(f"Moved {f} -> {dest}")

print(f"Successfully moved {len(moved)} files to {target_dir}/")
