import pickle
import os

pkl_path = "data/simlingo/buckets_paths.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    
    # Check for a specific existing folder in the values
    search_term = "Town12_Rep0_4380_route0_01_10_22_15_50"
    found = False
    
    print(f"Searching for {search_term} in pickle values...")
    
    for key, values in data.items():
        if isinstance(values, list):
            for v in values:
                if search_term in str(v):
                    print(f"Found in key '{key}': {v}")
                    found = True
                    break
        elif isinstance(values, dict):
             for k, v in values.items():
                if search_term in str(v) or search_term in str(k):
                     print(f"Found in key '{key}': {v}")
                     found = True
                     break
        
        if found: break
    
    if not found:
        print("Not found.")

    # Print first few paths to see pattern again
    if 'all' in data:
        print("\nSample paths from 'all':")
        for i in range(5):
            if i < len(data['all']):
                print(data['all'][i])
