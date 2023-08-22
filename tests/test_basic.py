# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest


class ImportTests(unittest.TestCase):
    def test_import(self):
        # noinspection PyUnresolvedReferences
        import matspy
        self.assertEqual(True, True)  # did not raise


if __name__ == '__main__':
    unittest.main()
