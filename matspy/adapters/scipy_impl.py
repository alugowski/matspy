# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import bisect
from typing import Any, Iterable

import numpy as np
import scipy.sparse

from . import describe, generate_spy_triple_product
from . import MatrixAdapterRow, MatrixAdapterCol, MatrixAdapterCoo, MatrixSpyAdapter


class CompressedSortedIterator:
    """A dense iterator over a CSR row or CSC column with sorted indices."""
    def __init__(self, indices_slice, data_slice, desired_index_range):
        self.indices_slice = indices_slice
        self.data_slice = data_slice
        self.index_iterator = iter(range(*desired_index_range))
        self.pos = bisect.bisect_left(indices_slice, desired_index_range[0])

    def __iter__(self):
        return self

    def __next__(self):
        # index of value being returned
        idx = next(self.index_iterator)

        if self.pos >= len(self.indices_slice):
            # reached past the end of this row/column
            return None

        if idx == self.indices_slice[self.pos]:
            # pointing at the next value in the row/column, so return it
            cur_pos = self.pos
            self.pos += 1
            return self.data_slice[cur_pos]

        # empty space
        return None


class SciPyAdapter:
    def __init__(self, mat):
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def describe(self) -> str:
        format_name = self.mat.getformat()

        return describe(shape=self.mat.shape, nnz=self.mat.nnz,
                        notes=f"'{str(self.mat.dtype)}', {format_name}")


class SciPyCSRAdapter(SciPyAdapter, MatrixAdapterRow):
    def __init__(self, mat: scipy.sparse.csr_array):
        super().__init__(mat)

    def get_row(self, row_idx: int, col_range: tuple[int, int]) -> Iterable[Any]:
        start = self.mat.indptr[row_idx]
        end = self.mat.indptr[row_idx + 1]

        index_slice = self.mat.indices[start:end]
        data_slice = self.mat.data[start:end]

        if not self.mat.has_sorted_indices:
            perm = np.argsort(index_slice)
            index_slice = index_slice[perm]
            data_slice = data_slice[perm]

        return CompressedSortedIterator(index_slice, data_slice, col_range)


class SciPyCSCAdapter(SciPyAdapter, MatrixAdapterCol):
    def __init__(self, mat: scipy.sparse.csc_array):
        super().__init__(mat)

    def get_col(self, col_idx: int, row_range: tuple[int, int]) -> Iterable[Any]:
        start = self.mat.indptr[col_idx]
        end = self.mat.indptr[col_idx + 1]

        index_slice = self.mat.indices[start:end]
        data_slice = self.mat.data[start:end]

        if not self.mat.has_sorted_indices:
            perm = np.argsort(index_slice)
            index_slice = index_slice[perm]
            data_slice = data_slice[perm]

        return CompressedSortedIterator(index_slice, data_slice, row_range)


class SciPyCOOAdapter(SciPyAdapter, MatrixAdapterCoo):
    def __init__(self, mat: scipy.sparse.coo_array):
        super().__init__(mat)

    def get_coo(self, row_range: tuple[int, int], col_range: tuple[int, int]) -> Iterable[tuple[int, int, Any]]:
        mask = (self.mat.row >= row_range[0]) & (self.mat.row < row_range[1]) & \
               (self.mat.col >= col_range[0]) & (self.mat.col < col_range[1])

        return zip(self.mat.row[mask], self.mat.col[mask], self.mat.data[mask])


def generate_spy_triple_product_coo(matrix_shape, spy_shape) -> tuple[scipy.sparse.coo_array, scipy.sparse.coo_array]:
    # construct a triple product that will scale the matrix
    left, right = generate_spy_triple_product(matrix_shape, spy_shape)

    left_shape, (left_rows, left_cols) = left
    right_shape, (right_rows, right_cols) = right
    left_mat = scipy.sparse.coo_array((np.ones(len(left_rows)), (left_rows, left_cols)), shape=left_shape)
    right_mat = scipy.sparse.coo_array((np.ones(len(right_rows)), (right_rows, right_cols)), shape=right_shape)

    return left_mat, right_mat


class SciPySpy(SciPyAdapter, MatrixSpyAdapter):
    def __init__(self, mat: scipy.sparse.coo_array):
        super().__init__(mat)

    def get_spy(self, spy_shape):
        # construct a triple product that will scale the matrix
        left, right = generate_spy_triple_product_coo(self.mat.shape, spy_shape)

        # save existing matrix data
        mat_data_save = self.mat.data

        # replace with all ones
        self.mat.data = np.ones(self.mat.data.shape)

        # triple product
        spy = left @ self.mat @ right

        # restore original matrix data
        self.mat.data = mat_data_save

        return spy.todense()
