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


if __name__ == '__main__':
    unittest.main()
