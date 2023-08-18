# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

"""
Note: Only import this in Jupyter.
If not using Jupyter then just `import matspy`.

Importing this module registers formatters with Jupyter.

Also includes all public methods from the matspy module for convenient one-line imports.
"""

# noinspection PyUnresolvedReferences
from .jupyter_html import *
