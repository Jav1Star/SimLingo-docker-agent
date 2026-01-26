import pickle as pkl
import sys

file_path = 'data/simlingo/buckets_paths.pkl'
try:
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    
    print(f"Loaded {file_path}")
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        print(f"Bucket {k}: {len(v)} items")
        if len(v) > 0:
            print(f"Sample item from {k}: {v[0]}")
except Exception as e:
    print(f"Error reading file: {e}")
