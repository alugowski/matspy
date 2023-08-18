# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse

from matspy import to_latex
from .test_html import generate_fixed_value

numpy.random.seed(123)


class ToLatexTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            scipy.sparse.random(10, 10, density=0.4),
            scipy.sparse.random(5, 10, density=0.4),
            scipy.sparse.random(5, 1, density=0.4),
            scipy.sparse.coo_array(([], ([], [])), shape=(10, 10)),
            generate_fixed_value(10, 10)
        ]

    def validate_latex(self, s):
        self.assertGreater(len(s), 0)

    def test_basic_validate(self):
        for mat in self.mats:
            for to_latex_args in [
                dict(max_rows=20, max_cols=20, title=False),
                dict(max_rows=20, max_cols=20, title=True),
                dict(max_rows=20, max_cols=20, title=False, indices=True),
                dict(max_rows=20, max_cols=20, title=True, indices=True),
                dict(max_rows=4, max_cols=4, title=True, indices=True),
            ]:
                res = to_latex(mat, **to_latex_args)
                self.validate_latex(res)

    def test_title(self):
        mat = generate_fixed_value(4, 4)
        off = to_latex(mat, title=False)
        self.assertNotIn("elements", off)

        on = to_latex(mat, title=True)
        self.assertIn(f"{mat.nnz} elements", on)

        title = "test title"
        custom = to_latex(mat, title=title)
        self.assertIn(title, custom)

    def test_precision(self):
        f = 0.123456789
        mat = scipy.sparse.coo_array(([f], ([0], [0])), shape=(1, 1))

        # default precision
        res = to_latex(mat)
        self.assertIn("0.1235", res)

        # explicit format string
        res = to_latex(mat, float_formatter=".2g")
        self.assertIn("0.12", res)

        # explicit precision number
        res = to_latex(mat, precision=6)
        self.assertIn("0.123457", res)


if __name__ == '__main__':
    unittest.main()
