# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest


class ImportTests(unittest.TestCase):
    def test_import_not_in_jupyter(self):
        # noinspection PyUnresolvedReferences
        import matspy.jupyter
        self.assertEqual(True, True)  # did not raise

    def test_jupyter_html(self):
        # noinspection PyUnresolvedReferences
        import matspy.jupyter_html
        self.assertEqual(True, True)  # did not raise

    def test_jupyter_latex(self):
        # noinspection PyUnresolvedReferences
        import matspy.jupyter_latex
        self.assertEqual(True, True)  # did not raise


if __name__ == '__main__':
    unittest.main()
