# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse

from matspy import to_sparkline

numpy.random.seed(123)


class SparklineTests(unittest.TestCase):
    def test_small_buckets(self):
        for dims in [(1001, 1001), (11, 11)]:
            with self.subTest(str(dims)):
                r = scipy.sparse.random(*dims, density=0.2)
                s = to_sparkline(r, buckets=10)
                self.assertGreater(len(s), 1)

    def test_buckets_1(self):
        for dims in [(1001, 1001), (1000, 1000), (501, 501), (500, 500), (10, 11), (11, 11)]:
            with self.subTest(str(dims)):
                r = scipy.sparse.random(*dims, density=0.2)
                s = to_sparkline(r, buckets=1)
                self.assertGreater(len(s), 1)


if __name__ == '__main__':
    unittest.main()
