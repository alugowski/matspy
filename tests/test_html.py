# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import html5lib
import numpy.random
import scipy.sparse

from matspy import to_html

numpy.random.seed(123)


def generate_fixed_value(m, n):
    row_factor = 10**(1+len(str(n)))
    nnz = m*n
    rows, cols, data = [1] * nnz, [1] * nnz, [1] * nnz
    for i in range(nnz):
        r = int(i / n)
        c = i % n
        rows[i] = r
        cols[i] = c
        data[i] = (r+1)*row_factor + c

    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(m, n))


class ToHTMLTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            scipy.sparse.random(10, 10, density=0.4),
            scipy.sparse.random(5, 10, density=0.4),
            scipy.sparse.random(5, 1, density=0.4),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)),
            generate_fixed_value(10, 10)
        ]

    def test_basic_validate(self):
        """
        A best-effort validation against silly mistakes.
        """
        html5parser = html5lib.HTMLParser(strict=True)

        for mat in self.mats:
            for to_html_args in [
                dict(notebook=False, max_rows=20, max_cols=20, title=False, indices=False),
                dict(notebook=True, max_rows=20, max_cols=20, title=True, indices=True),
                dict(notebook=True, max_rows=4, max_cols=4, title=True, indices=True),
            ]:
                res = to_html(mat, **to_html_args)
                html5parser.parseFragment(res)
                self.assertEqual(True, True)  # did not raise

    def test_contents(self):
        mat = generate_fixed_value(10, 10)
        res = to_html(mat, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        for value in mat.data:
            self.assertIn(f"<td>{value}</td>", res)

    def test_truncate(self):
        for fmt in ["csr", "csc"]:
            mat = generate_fixed_value(10, 10).asformat(fmt)

            for after_dots, expected_count in [
                (0, 25),  # 5*5
                (1, 25),  # 4*4 + 4 + 4 + 1
                (2, 25),  # 3*3 + 2*3 + 3*2 + 2*2
            ]:
                res = to_html(mat, notebook=False, max_rows=6, max_cols=6, num_after_dots=after_dots)
                count = 0
                for value in mat.data:
                    if f"<td>{value}</td>" in res:
                        count += 1
                print(res)
                self.assertEqual(expected_count, count)

    def test_precision(self):
        f = 0.123456789
        mat = scipy.sparse.coo_matrix(([f], ([0], [0])), shape=(1, 1))

        # default precision
        res = to_html(mat)
        self.assertIn("<td>0.1235</td>", res)

        # explicit format string
        res = to_html(mat, float_formatter=".2g")
        self.assertIn("<td>0.12</td>", res)

        # explicit precision number
        res = to_html(mat, precision=6)
        self.assertIn("<td>0.123457</td>", res)

    def test_title(self):
        mat = generate_fixed_value(4, 4)
        off = to_html(mat, title=False)
        self.assertNotIn("elements", off)

        on = to_html(mat, title=True)
        self.assertIn(f"{mat.nnz} elements", on)

        title = "test title"
        custom = to_html(mat, title=title)
        self.assertIn(title, custom)


if __name__ == '__main__':
    unittest.main()
