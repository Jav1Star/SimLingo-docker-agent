import base64
import json
import os
import unittest

import numpy as np

from simlingo_agents.common.nats_codec import (
    LEGACY_JSON_CODEC,
    MSGPACK_CODEC,
    NatsCodecError,
    decode_message,
    encode_message,
)


class NatsCodecTest(unittest.TestCase):
    def test_msgpack_round_trip_nested_numpy(self):
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::2]
        payload = {
            "frame_id": 7,
            "camera": source,
            "nested": [np.asarray([True, False]), np.float32(0.25)],
        }

        encoded = encode_message(payload, MSGPACK_CODEC)
        decoded = decode_message(encoded)

        self.assertEqual(decoded["frame_id"], 7)
        self.assertEqual(decoded["camera"].dtype, source.dtype)
        np.testing.assert_array_equal(decoded["camera"], source)
        np.testing.assert_array_equal(decoded["nested"][0], [True, False])
        self.assertAlmostEqual(decoded["nested"][1], 0.25)

    def test_legacy_json_base64_is_decoded(self):
        source = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
        legacy = {
            "encoded_payload": {
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "data": base64.b64encode(source.tobytes()).decode("ascii"),
            }
        }

        decoded = decode_message(json.dumps(legacy).encode())
        np.testing.assert_array_equal(decoded["encoded_payload"], source)

    def test_legacy_write_mode_round_trip(self):
        source = np.arange(3, dtype=np.float64)
        decoded = decode_message(encode_message({"value": source}, LEGACY_JSON_CODEC))
        np.testing.assert_array_equal(decoded["value"], source)

    def test_environment_selects_legacy_write_mode(self):
        previous = os.environ.get("SIMLINGO_NATS_CODEC")
        os.environ["SIMLINGO_NATS_CODEC"] = "legacy-json"
        try:
            encoded = encode_message({"value": np.asarray([5], dtype=np.uint8)})
        finally:
            if previous is None:
                os.environ.pop("SIMLINGO_NATS_CODEC", None)
            else:
                os.environ["SIMLINGO_NATS_CODEC"] = previous
        self.assertTrue(encoded.startswith(b"{"))

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(NatsCodecError, "unsupported.*version"):
            decode_message(b"SLNG\x02payload")


if __name__ == "__main__":
    unittest.main()
