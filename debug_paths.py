
import glob
import os
import pickle as pkl
import sys
from pathlib import Path

repo_path = os.getcwd() # Since we run from root
data_path = 'data/simlingo'
bucket_path = 'data/simlingo'

print(f"Repo path: {repo_path}")

# Simulate glob from dataset_base.py
glob_pattern = f"{repo_path}/{data_path}/data/simlingo/*/*/*/Town*"
print(f"Glob pattern: {glob_pattern}")
route_dirs = glob.glob(glob_pattern)
print(f"Found {len(route_dirs)} route dirs on disk.")
if len(route_dirs) > 0:
    print(f"Sample route_dir: {route_dirs[0]}")
    # Sample measurement file path
    meas_dir = os.path.join(route_dirs[0], 'measurements')
    if os.path.exists(meas_dir):
        files = os.listdir(meas_dir)
        if files:
            print(f"Sample measurement file (Actual): {os.path.join(meas_dir, files[0])}")
            sample_actual_meas = Path(os.path.join(meas_dir, files[0]))
            print(f"Sample parent (Actual): {sample_actual_meas.parent}")

# Simulate pkl load
pkl_path = f"{bucket_path}/buckets_paths.pkl"
try:
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    print(f"Loaded pkl from {pkl_path}")
    
    # Simulate processing
    run_id_dict = {}
    sample_pkl_path = None
    
    # Just take the first bucket to test
    for b_name, paths in data.items():
        if len(paths) > 0:
            sample_pkl_path = paths[0]
            
            # Apply replacement logic
            run_id = sample_pkl_path.replace('database/simlingo_v2_2025_01_10', bucket_path)
            run_id_path = Path(run_id)
            run_id_parent = run_id_path.parent
            run_id_absolut = f"{repo_path}/{str(run_id_parent)}"
            
            print("\n--- Processing Sample --")
            print(f"Original PKL path: {sample_pkl_path}")
            print(f"After replace: {run_id}")
            print(f"Calculated Absolute Parent: {run_id_absolut}")
            
            run_id_dict[run_id_absolut] = True
            break
            
    # Check match
    if len(route_dirs) > 0:
        sample_route_dir = route_dirs[0]
        # We need to check if the route found on disk exists in the processed dict (assuming we processed all)
        # But this is hard since we only processed one. 
        # Instead, let's see if the CALCULATED absolute parent looks like the ACTUAL route dir + /measurements
        
        # Check normalization
        print(f"\nComparing paths...")
        # Note: dataset_base puts key as .../measurements usually?
        # run_id_parent of .../measurements/0xxx.json.gz is .../measurements
        
        calculated_path = Path(run_id_absolut)
        on_disk_path = Path(route_dirs[0]) / 'measurements'
        
        print(f"Calculated: {calculated_path}")
        print(f"On Disk (Example): {on_disk_path}")
        
except Exception as e:
    print(e)
