# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse

from matspy import spy_to_mpl, to_sparkline

numpy.random.seed(123)


class SciPyTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            scipy.sparse.random(10, 10, density=0.4).tocoo(),
            scipy.sparse.random(5, 10, density=0.4).tocsr(),
            scipy.sparse.random(5, 1, density=0.4).tocsc(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocoo(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocsr(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocsc(),
        ]

    def test_no_crash(self):
        import matplotlib.pyplot as plt
        for mat in self.mats:
            fig, ax = spy_to_mpl(mat)
            plt.close(fig)

            res = to_sparkline(mat)
            self.assertGreater(len(res), 10)


if __name__ == '__main__':
    unittest.main()
