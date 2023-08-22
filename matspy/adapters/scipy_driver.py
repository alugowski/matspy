# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable

from . import Driver, MatrixSpyAdapter


class SciPyDriver(Driver):
    @staticmethod
    def get_supported_type_prefixes() -> Iterable[str]:
        return ["scipy.sparse"]

    @staticmethod
    def adapt_spy(mat: Any) -> MatrixSpyAdapter:
        from .scipy_impl import SciPySpy
        return SciPySpy(mat)
