"""The picture side of the dsv41 engine's own vision tower: geometry, marker expansion, DSI1."""
import struct
import tempfile
import unittest
from pathlib import Path

from serve import dsv41_vision as V

GEO = {"token": 129264, "patch": 14, "merge": 3, "max_tokens": 1024}


class Geometry(unittest.TestCase):
    def test_rows_match_the_engine(self):
        # measured with: strata-dsv41 --encode-image <rgb> --size WxH --dump-image out
        for (w, h), want in (((420, 420), 112), ((700, 700), 308), ((1008, 672), 402), ((1540, 700), 648)):
            self.assertEqual(V.rows(w, h, GEO), want, (w, h))

    def test_fit_keeps_the_patch_grid_and_the_budget(self):
        for w, h in ((1024, 701), (4000, 3000), (100, 5000), (37, 37)):
            fw, fh = V.fit(w, h, GEO)
            self.assertEqual(fw % 42, 0)
            self.assertEqual(fh % 42, 0)
            self.assertLessEqual(V.rows(fw, fh, GEO), GEO["max_tokens"], (w, h, fw, fh))


class Prompt(unittest.TestCase):
    def test_expand_puts_one_run_per_picture(self):
        prompt = "a" + V.MARKER + "b" + V.MARKER + "c"
        out = V.expand(prompt, [3, 2])
        self.assertEqual(out, "a" + V.IMAGE_TEXT * 3 + "b" + V.IMAGE_TEXT * 2 + "c")

    def test_expand_refuses_a_mismatch(self):
        with self.assertRaises(ValueError):
            V.expand("x" + V.MARKER, [3, 2])

    def test_starts_follow_the_prompt_order(self):
        ids = [5, 129264, 129264, 129264, 6, 129264, 129264]
        self.assertEqual(V.starts(ids, 129264, [3, 2]), [1, 5])
        with self.assertRaises(ValueError):
            V.starts(ids, 129264, [3, 2, 1])


class Dsi1(unittest.TestCase):
    def test_file_layout(self):
        pics = [(42, 42, bytes(42 * 42 * 3)), (84, 42, bytes(84 * 42 * 3))]
        with tempfile.TemporaryDirectory() as d:
            path = V.write_dsi1(pics, [7, 30], directory=d)
            raw = Path(path).read_bytes()
        self.assertEqual(raw[:4], b"DSI1")
        self.assertEqual(struct.unpack("<i", raw[4:8])[0], 2)
        start = struct.unpack("<q", raw[8:16])[0]
        w, h = struct.unpack("<ii", raw[16:24])
        self.assertEqual((start, w, h), (7, 42, 42))
        self.assertEqual(len(raw), 8 + (8 + 8 + 42 * 42 * 3) + (8 + 8 + 84 * 42 * 3))


if __name__ == "__main__":
    unittest.main()
