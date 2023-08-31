# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

try:
    import sparse
except ImportError:
    sparse = None

import numpy as np
import scipy.sparse

from matspy import spy_to_mpl, to_sparkline, to_spy_heatmap

np.random.seed(123)


@unittest.skipIf(sparse is None, "pydata/sparse not installed")
class PyDataSparseTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            sparse.COO.from_scipy_sparse(scipy.sparse.random(10, 10, density=0.4)),
            sparse.COO.from_scipy_sparse(scipy.sparse.random(5, 10, density=0.4)),
            sparse.COO.from_scipy_sparse(scipy.sparse.random(5, 1, density=0.4)),
            sparse.COO.from_scipy_sparse(scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10))),
        ]

    def test_no_crash(self):
        import matplotlib.pyplot as plt
        for fmt in "coo", "gcxs", "dok", "csr", "csc":
            for source_mat in self.mats:
                mat = source_mat.asformat(fmt)

                fig, ax = spy_to_mpl(mat)
                plt.close(fig)

                res = to_sparkline(mat)
                self.assertGreater(len(res), 10)

    def test_count(self):
        arrs = [
            (0, sparse.COO(np.array([[0]]))),
            (1, sparse.COO(np.array([[1]]))),
            (0, sparse.COO(np.array([[0, 0], [0, 0]]))),
            (1, sparse.COO(np.array([[1, 0], [0, 0]]))),
        ]

        for count, arr in arrs:
            area = np.prod(arr.shape)
            heatmap = to_spy_heatmap(arr, buckets=1, shading="absolute")
            self.assertEqual(len(heatmap), 1)
            self.assertAlmostEqual( count / area, heatmap[0][0], places=2)


if __name__ == '__main__':
    unittest.main()
