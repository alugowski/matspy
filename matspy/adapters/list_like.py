# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Optional
from itertools import chain, islice, repeat
from . import describe, Driver, MatrixAdapterRow, MatrixSpyAdapter


def is_single(x):
    return not hasattr(x, '__len__') or isinstance(x, str)


class ListAdapter(MatrixAdapterRow):
    def __init__(self, mat: list):
        self.mat = mat
        self.row_lengths = [(1 if is_single(row) else len(row)) for row in mat]
        self.shape = (len(mat), max(self.row_lengths))

    def get_shape(self) -> tuple[int, int]:
        return self.shape

    def describe(self) -> str:
        return describe(shape=self.shape, nnz=sum(self.row_lengths), notes=None)

    def get_row(self, row_idx: int, col_range: tuple[int, int]) -> Iterable[Any]:
        num_desired = col_range[1] - col_range[0]
        row = self.mat[row_idx]

        if is_single(row):
            # single element
            if col_range[0] <= 0 < col_range[1]:
                return chain([row], repeat(None, num_desired - 1))
            else:
                return repeat(None, num_desired)

        return islice(chain(iter(row), repeat(None, num_desired - len(row))), col_range[0], col_range[1])


class ListDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[tuple[str, str, bool]]:
        return [
            ("builtins", "list", False),
            ("builtins", "tuple", False),
        ]

    @staticmethod
    def adapt(mat: Any):
        if isinstance(mat, (list, tuple)):
            return ListAdapter(mat)
        raise ValueError

    @staticmethod
    def adapt_spy(mat: Any) -> Optional[MatrixSpyAdapter]:
        return None
