# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest
import time

import numpy.random
import scipy.sparse

from matspy import to_spy_heatmap

numpy.random.seed(123)


class PerformanceTests(unittest.TestCase):
    def test_to_html_speed(self):
        # warmup, just in case
        r = scipy.sparse.random(100, 100, density=0.2, format="csr")
        to_spy_heatmap(r)

        # Verify that to_html() is fast.
        # Using somewhat small dimensions to keep the test snappy.
        a = time.time()
        randmat = scipy.sparse.random(1000, 1000, density=0.2, format="csr")
        b = time.time()
        to_spy_heatmap(randmat)
        c = time.time()

        self.assertGreater((b-a), (c-b))


if __name__ == '__main__':
    unittest.main()
