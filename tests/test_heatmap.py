# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse
from matspy import to_spy_heatmap

numpy.random.seed(123)


class SpyHeatmapTests(unittest.TestCase):
    def test_buckets_1(self):
        density = 0.3
        for dims in [(501, 501), (10, 10)]:
            r = scipy.sparse.random(*dims, density=density)
            heatmap = to_spy_heatmap(r, buckets=1, shading="absolute")
            self.assertEqual(len(heatmap), 1)
            self.assertAlmostEqual(heatmap[0][0], density, places=2)

            heatmap = to_spy_heatmap(r, buckets=1, shading="binary")
            self.assertAlmostEqual(heatmap[0][0], 1.0, places=2)

    def test_aa_tweaks(self):
        from matspy.spy_renderer import _tweak_divisor

        # more pixels than matrix dimensions
        orig, display = 80, 100
        self.assertEqual(_tweak_divisor(orig, display), orig)

        orig, display = 100, 100
        self.assertEqual(_tweak_divisor(orig, display), orig)

        # matrix slightly bigger than screen
        orig, display = 101, 100
        self.assertEqual(_tweak_divisor(orig, display), orig)

        orig, display = 100, 99
        self.assertEqual(_tweak_divisor(orig, display), orig)

        # matrix bigger than screen
        orig, display = 415, 100
        buckets = _tweak_divisor(orig, display)
        self.assertLessEqual(orig % buckets, 2)

        orig, display = 4315, 873
        buckets = _tweak_divisor(orig, display)
        self.assertLessEqual(orig % buckets, 2)


if __name__ == '__main__':
    unittest.main()
