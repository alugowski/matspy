# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from matspy import to_html, to_latex
import matspy


class ListLikeTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            [[1, 2], [1003, 1004]],
            [[1, 2], [1003, 1004, 1005]],
        ]

    def test_no_crash(self):
        for mat in self.mats:
            res = to_html(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_latex(mat, title=True)
            self.assertGreater(len(res), 10)

    def test_shape(self):
        mat = (1, 2, 3, 4)
        adapter = matspy._get_adapter(mat)
        self.assertEqual((4, 1), adapter.get_shape())

        mat = [[1, 2], [1003, 1004, 1005]]
        adapter = matspy._get_adapter(mat)
        self.assertEqual((2, 3), adapter.get_shape())


if __name__ == '__main__':
    unittest.main()
