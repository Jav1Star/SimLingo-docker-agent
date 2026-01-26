import pickle
import os

pkl_path = "data/simlingo/buckets_paths.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    print(f"Keys: {list(data.keys())}")
    
    # Let's check a few specific keys if they exist
    target_keys = ['leading_vehicle', 'leading_object_vehicle', 'braking'] # guessing names
    
    for key in data.keys():
        print(f"Checking bucket: {key}")
        paths = data[key]
        if not paths:
            print("  Empty list in pickle.")
            continue
            
        print(f"  Total paths in pickle: {len(paths)}")
        
        # Check first path
        first_path = paths[0]
        # Apply transformation
        local_path = first_path.replace('database/simlingo_v2_2025_01_10', 'data/simlingo')
        
        print(f"  First raw path: {first_path}")
        print(f"  Mapped path:    {local_path}")
        
        if os.path.exists(local_path):
            print("  [SUCCESS] Mapped path exists on disk.")
        else:
            print("  [FAILURE] Mapped path DOES NOT exist on disk.")
