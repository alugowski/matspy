# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Tuple

import numpy as np


def describe(shape: tuple = None, nnz: int = None, notes: str = None) -> str:
    """
    Create a simple description string from potentially interesting pieces of metadata.
    """
    parts = []
    by = chr(215)  # ×
    if len(shape) == 1:
        parts.append(f"length: {shape[0]}")
    elif len(shape) == 2:
        parts.append(f"{shape[0]}{by}{shape[1]}")
    else:
        parts.append(f"shape: {by.join(shape)}")

    if nnz is not None:
        parts.append(f"{nnz} elements")

    if notes:
        parts.append(notes)

    return ", ".join(parts)


class MatrixAdapter(ABC):
    @abstractmethod
    def describe(self) -> str:
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        pass

    def get_row_labels(self) -> Iterable[Optional[Any]]:
        return range(self.get_shape()[0])

    def get_col_labels(self) -> Iterable[Optional[Any]]:
        return range(self.get_shape()[1])


class MatrixAdapterRow(MatrixAdapter):
    @abstractmethod
    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        pass


class MatrixAdapterCol(MatrixAdapter):
    @abstractmethod
    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Any]:
        pass


class MatrixAdapterCoo(MatrixAdapter):
    @abstractmethod
    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        pass


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
    def get_supported_types() -> Iterable[Tuple[str, str, bool]]:
        pass

    @staticmethod
    @abstractmethod
    def adapt(mat: Any) -> Optional[MatrixAdapter]:
        pass

    @staticmethod
    @abstractmethod
    def adapt_spy(mat: Any) -> Optional[MatrixSpyAdapter]:
        pass


class Truncated2DMatrix(MatrixAdapterRow):
    def __init__(self, orig_shape: Tuple[int, int], display_shape: Tuple[int, int], num_after_dots=2,
                 description=None):
        self.orig_shape = orig_shape
        self.display_shape = (min(orig_shape[0], display_shape[0]), min(orig_shape[1], display_shape[1]))
        self.nrows, self.ncols = self.display_shape
        self.elements = [[None] * self.ncols for _ in range(self.nrows)]
        self.description = description

        self.dot_col = None
        self.dot_row = None

        if self.display_shape[0] < self.orig_shape[0]:
            # need to truncate rows
            self.dot_row = max(0, self.display_shape[0] - 1 - num_after_dots)

        if self.display_shape[1] < self.orig_shape[1]:
            # need to truncate columns
            self.dot_col = max(0, self.display_shape[1] - 1 - num_after_dots)

    def describe(self) -> str:
        return self.description

    def get_shape(self):
        return self.display_shape

    def get_row_labels(self) -> Iterable[Optional[int]]:
        if self.dot_row is None:
            return list(range(self.orig_shape[0]))
        else:
            pre_dot_end, post_dot_start = self.get_dot_indices_row()
            # noinspection PyTypeChecker
            return list(range(pre_dot_end)) + [None] + list(range(post_dot_start, self.orig_shape[0]))

    def get_col_labels(self) -> Iterable[Optional[int]]:
        if self.dot_col is None:
            return list(range(self.orig_shape[1]))
        else:
            pre_dot_end, post_dot_start = self.get_dot_indices_col()
            # noinspection PyTypeChecker
            return list(range(pre_dot_end)) + [None] + list(range(post_dot_start, self.orig_shape[1]))

    def get_row(self, row_idx: int, col_range: Tuple[int, int] = None):
        if col_range is None:
            return self.elements[row_idx]
        else:
            return self.elements[row_idx][col_range[0]:col_range[1]]

    def set(self, row_idx: int, col_idx: int, value: Any):
        pre_dot_end, post_dot_start = self.get_dot_indices_row()
        if self.dot_row is not None and row_idx >= pre_dot_end:
            row_idx = self.dot_row + 1 + (row_idx - post_dot_start)
            if row_idx < self.dot_row:
                # within the dots
                return

        pre_dot_end, post_dot_start = self.get_dot_indices_col()
        if self.dot_col is not None and col_idx >= pre_dot_end:
            col_idx = self.dot_col + 1 + (col_idx - post_dot_start)
            if col_idx < self.dot_col:
                # within the dots
                return

        self.elements[row_idx][col_idx] = value

    def apply_dots(self, dots):
        if self.dot_row is not None:
            for i in range(len(self.elements[self.dot_row])):
                self.elements[self.dot_row][i] = dots["v"]
        if self.dot_col is not None:
            for i in range(len(self.elements)):
                self.elements[i][self.dot_col] = dots["h"]
            if self.dot_row is not None:
                self.elements[self.dot_row][self.dot_col] = dots["d"]

    def get_dot_indices_row(self) -> Tuple[int, int]:
        if self.dot_row is None:
            return self.orig_shape[0], self.orig_shape[0]
        else:
            num_post_dot = self.display_shape[0] - 1 - self.dot_row
            return self.dot_row, self.orig_shape[0] - num_post_dot

    def get_dot_indices_col(self) -> Tuple[int, int]:
        if self.dot_col is None:
            return self.orig_shape[1], self.orig_shape[1]
        else:
            num_post_dot = self.display_shape[1] - 1 - self.dot_col
            return self.dot_col, self.orig_shape[1] - num_post_dot


def to_trunc(mat: MatrixAdapter, max_rows, max_cols, num_after_dots) -> Truncated2DMatrix:
    """
    Convert a
    :param mat:
    :param max_rows:
    :param max_cols:
    :param num_after_dots:
    :return:
    """
    if isinstance(mat, Truncated2DMatrix):
        return mat

    if isinstance(mat, MatrixAdapterCoo):
        trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                  display_shape=(max_rows, max_cols),
                                  num_after_dots=0,
                                  description=mat.describe())
        for row, col, val in mat.get_coo(row_range=(0, max_rows), col_range=(0, max_cols)):
            trunc.set(row, col, val)
        return trunc

    if isinstance(mat, MatrixAdapterRow):
        trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                  display_shape=(max_rows, max_cols),
                                  num_after_dots=num_after_dots,
                                  description=mat.describe())

        pre_dot_end, post_dot_start = trunc.get_dot_indices_col()
        _, ncols = mat.get_shape()

        # fetch the pre-dot rows
        for row_idx in trunc.get_row_labels():
            if row_idx is None:
                # dots
                continue

            for col_range in [(0, pre_dot_end), (post_dot_start, ncols)]:
                values = mat.get_row(row_idx, col_range=col_range)
                for col_idx, value in enumerate(values, start=col_range[0]):
                    trunc.set(row_idx, col_idx, value)
        return trunc

    if isinstance(mat, MatrixAdapterCol):
        trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                  display_shape=(max_rows, max_cols),
                                  num_after_dots=num_after_dots,
                                  description=mat.describe())

        pre_dot_end, post_dot_start = trunc.get_dot_indices_row()
        nrows, _ = mat.get_shape()

        # fetch the pre-dot rows
        for col_idx in trunc.get_col_labels():
            if col_idx is None:
                # dots
                continue

            for row_range in [(0, pre_dot_end), (post_dot_start, nrows)]:
                values = mat.get_col(col_idx, row_range=row_range)
                for row_idx, value in enumerate(values, start=row_range[0]):
                    trunc.set(row_idx, col_idx, value)
        return trunc

    raise NotImplementedError


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
