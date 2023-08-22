# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Tuple

import numpy as np


def describe(shape: tuple = None, nnz: int = None, nz_type=None, notes: str = None) -> str:
    """
    Create a simple description string from potentially interesting pieces of metadata.
    """
    parts = []
    by = chr(215)  # ×
    if len(shape) == 1:
        parts.append(f"length={shape[0]}")
    elif len(shape) == 2:
        parts.append(f"{shape[0]}{by}{shape[1]}")
    else:
        parts.append(f"shape={by.join(shape)}")

    if nnz is not None:
        dtype_str = f" '{str(nz_type)}'" if nz_type else ""
        parts.append(f"{nnz}{dtype_str} elements")

    if notes:
        parts.append(notes)

    return ", ".join(parts)


class MatrixSpyAdapter(ABC):
    @abstractmethod
    def describe(self) -> str:
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        pass

    @abstractmethod
    def get_spy(self, spy_shape: tuple) -> np.array:
        pass


class Driver(ABC):
    @staticmethod
    @abstractmethod
    def get_supported_type_prefixes() -> Iterable[str]:
        pass

    @staticmethod
    @abstractmethod
    def adapt_spy(mat: Any) -> Optional[MatrixSpyAdapter]:
        pass


def generate_spy_triple_product(matrix_shape, spy_shape, uneven_to_end=True) ->\
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """
    Generate left and right matrices to create a matrix spy plot using two matrix multiplications.
    """
    left_shape = (spy_shape[0], matrix_shape[0])
    right_shape = (matrix_shape[1], spy_shape[1])

    left_nnz = max(left_shape)
    right_nnz = max(right_shape)

    def gen_even(stop, num):
        return np.linspace(0, stop, num=num, endpoint=False, dtype="int64")

    def gen(stop, num):
        remainder = num % stop
        if not uneven_to_end or num % stop == 0 or remainder > 4:
            return gen_even(stop, num)

        step = int(num / stop)
        a = np.repeat(np.arange(0, stop, dtype='int64'), step)
        b = np.full((num - len(a)), stop - 1, dtype="int64")
        return np.concatenate((a, b))

    left_rows = gen(left_shape[0], num=left_nnz)
    left_cols = gen_even(left_shape[1], num=left_nnz)

    right_rows = gen_even(right_shape[0], num=right_nnz)
    right_cols = gen(right_shape[1], num=right_nnz)

    return (left_shape, (left_rows, left_cols)), (right_shape, (right_rows, right_cols))
