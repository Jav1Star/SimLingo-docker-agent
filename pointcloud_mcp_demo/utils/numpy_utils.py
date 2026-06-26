import numpy as np
import base64

def encode_array_to_dict(array: np.ndarray) -> dict:
    return {"data": base64.b64encode(array.tobytes()).decode("utf-8"), "shape": array.shape, "dtype": str(array.dtype)}

def decode_array_from_dict(array_dict: dict) -> np.ndarray:
    data = base64.b64decode(array_dict["data"])
    return np.frombuffer(data, dtype=array_dict["dtype"]).reshape(array_dict["shape"])