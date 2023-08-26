# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import matspy


class BasicTests(unittest.TestCase):
    def test_adaptation_errors(self):
        with self.assertRaises(AttributeError):
            matspy._get_spy_adapter(set())

    def test_argument_errors(self):
        with self.assertRaises(ValueError):
            matspy.to_spy_heatmap([], shading="foobar")


if __name__ == '__main__':
    unittest.main()
