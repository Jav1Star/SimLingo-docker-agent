import json

def load_json(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError:
        d = None
    except json.JSONDecodeError as e:
        d = None
    return d