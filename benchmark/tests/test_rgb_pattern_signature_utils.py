from __future__ import annotations

import unittest

from benchmark.rgb_pattern_prediction.signature_utils import (
    canonical_cycle,
    edit_distance,
    make_signature,
    parse_signature,
)


class SignatureUtilitiesTest(unittest.TestCase):
    def test_rotation_is_equivalent(self) -> None:
        self.assertEqual(make_signature("LQCA"), make_signature("CALQ"))

    def test_reversal_is_equivalent(self) -> None:
        self.assertEqual(make_signature("LLQCA"), make_signature("ACQLL"))

    def test_canonical_signature_roundtrip(self) -> None:
        signature = make_signature("LALA")
        self.assertEqual(parse_signature(signature), canonical_cycle("LALA"))

    def test_declared_count_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            parse_signature("5:LLLL")

    def test_invalid_primitive_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_signature("LLX")

    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("LLLCC", "LLLC"), 1)
        self.assertEqual(edit_distance("LLQLLQQ", "LLQQLQQ"), 1)
        self.assertEqual(edit_distance("LLLL", "LLLL"), 0)


if __name__ == "__main__":
    unittest.main()
