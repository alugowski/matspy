# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

"""
Note: Only import this in Jupyter.
If not using Jupyter then just `import matspy`.

Importing this module registers Latex formatters with Jupyter.

Also includes all public methods from the matspy module for convenient one-line imports.
"""
from . import *
from . import _register_jupyter_formatter, _driver_registration_notify


def _repr_latex_(mat):
    return to_latex(mat)


def _register_latex_formatters_with_jupyter(_=None):
    _register_jupyter_formatter("text/latex", _repr_latex_)


_register_latex_formatters_with_jupyter()
_driver_registration_notify.append(_register_latex_formatters_with_jupyter)
