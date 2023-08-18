# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse

from matspy import spy_to_mpl, to_sparkline, to_html, to_latex
from .test_html import generate_fixed_value

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
            generate_fixed_value(10, 10)
        ]

    def test_no_crash(self):
        import matplotlib.pyplot as plt
        for mat in self.mats:
            fig, ax = spy_to_mpl(mat)
            plt.close(fig)

            res = to_sparkline(mat)
            self.assertGreater(len(res), 10)

            res = to_html(mat)
            self.assertGreater(len(res), 10)

            res = to_latex(mat)
            self.assertGreater(len(res), 10)

    def test_formats(self):
        to_html_args = dict(notebook=False, max_rows=20, max_cols=20, title=False)

        expected = [to_html(m, **to_html_args) for m in self.mats]

        for fmt in ["coo", "csr", "csc", "dok", "lil"]:
            with self.subTest(f"{fmt}"):
                for i, source_mat in enumerate(self.mats):
                    mat = source_mat.asformat(fmt)
                    res = to_html(mat, **to_html_args)
                    self.assertEqual(expected[i], res)


if __name__ == '__main__':
    unittest.main()
