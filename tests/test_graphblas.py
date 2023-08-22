# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from matspy import spy_to_mpl, to_sparkline
import matspy

try:
    import graphblas as gb

    # Context initialization must happen before any other imports
    gb.init("suitesparse", blocking=True)

    have_gb = True
except ImportError:
    have_gb = False


@unittest.skipIf(not have_gb, "python-graphblas not installed")
class GraphBLASTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            gb.Matrix.from_coo([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], nrows=5, ncols=5),
        ]

    def test_no_crash(self):
        import matplotlib.pyplot as plt
        for mat in self.mats:
            fig, ax = spy_to_mpl(mat)
            plt.close(fig)

            res = to_sparkline(mat)
            self.assertGreater(len(res), 10)

    def test_shape(self):
        mat = gb.Matrix.from_coo([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4], nrows=5, ncols=1)
        adapter = matspy._get_spy_adapter(mat)
        self.assertEqual((5, 1), adapter.get_shape())


if __name__ == '__main__':
    unittest.main()
