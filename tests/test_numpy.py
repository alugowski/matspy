# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np

from matspy import spy_to_mpl, to_sparkline, to_spy_heatmap

np.random.seed(123)


class NumPyTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            np.array([[]]),
            np.random.random((10, 10)),
        ]

    def test_no_crash(self):
        import matplotlib.pyplot as plt
        for mat in self.mats:
            fig, ax = spy_to_mpl(mat)
            plt.close(fig)

            res = to_sparkline(mat)
            self.assertGreater(len(res), 5)

    def test_shape(self):
        arr = np.array([])
        with self.assertRaises(ValueError):
            spy_to_mpl(arr)

    def test_count(self):
        arrs = [
            (1, np.array([[1]])),
            (1, np.array([[1, 0], [0, 0]])),
            (1, np.array([[1, None], [None, None]])),
            (1, np.array([[1, 0], [None, None]])),
        ]

        for count, arr in arrs:
            area = np.prod(arr.shape)
            heatmap = to_spy_heatmap(arr, buckets=1, shading="absolute")
            self.assertEqual(len(heatmap), 1)
            self.assertAlmostEqual( count / area, heatmap[0][0], places=2)

    def test_precision(self):
        precision = 0.5
        arrs = [
            (1, np.array([[1]])),
            (1, np.array([[1, 0], [0, 0]])),
            (1, np.array([[1, None], [0.4, -0.2]])),
            (1, np.array([[1, 0], [None, None]])),
            (1, np.array([[1, 0.5], [0, -0.1]])),
        ]

        for count, arr in arrs:
            area = np.prod(arr.shape)
            heatmap = to_spy_heatmap(arr, buckets=1, shading="absolute", precision=precision)
            self.assertEqual(len(heatmap), 1)
            self.assertAlmostEqual( count / area, heatmap[0][0], places=2)


if __name__ == '__main__':
    unittest.main()
