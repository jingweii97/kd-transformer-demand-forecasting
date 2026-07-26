import os

target_dir = "id-ood-analysis"

# Keep only the 4 essential deliverables requested for this phase
keep_files = {
    "m5_28day_timeline_summary.csv",
    "m5_temporal_profile.png",
    "candidate_protocols.csv",
    "protocol_analysis_summary.md"
}

if os.path.exists(target_dir):
    for fname in os.listdir(target_dir):
        if fname not in keep_files:
            filepath = os.path.join(target_dir, fname)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Removed irrelevant file: {filepath}")

print("\nCleanup complete. Files remaining in id-ood-analysis/:")
for fname in sorted(os.listdir(target_dir)):
    print(f" - {fname}")
