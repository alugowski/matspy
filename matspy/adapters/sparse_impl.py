# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Tuple

import numpy as np
import sparse

from . import describe, generate_spy_triple_product, MatrixSpyAdapter


def generate_spy_triple_product_sparse(matrix_shape, spy_shape) -> Tuple[sparse.SparseArray, sparse.SparseArray]:
    # construct a triple product that will scale the matrix
    left, right = generate_spy_triple_product(matrix_shape, spy_shape)

    left_shape, (left_rows, left_cols) = left
    right_shape, (right_rows, right_cols) = right
    left_mat = sparse.COO(coords=(left_rows, left_cols), data=np.ones(len(left_rows)), shape=left_shape)
    right_mat = sparse.COO(coords=(right_rows, right_cols), data=np.ones(len(right_rows)), shape=right_shape)

    return left_mat, right_mat


class PyDataSparseSpy(MatrixSpyAdapter):
    def __init__(self, mat):
        super().__init__()
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def describe(self) -> str:
        parts = [
            self.mat.format,
        ]

        return describe(shape=self.mat.shape,
                        nnz=self.mat.nnz, nz_type=self.mat.dtype,
                        notes=", ".join(parts))

    def get_spy(self, spy_shape: tuple) -> np.array:
        if isinstance(self.mat, sparse.DOK):
            self.mat = self.mat.asformat("coo")

        # construct a triple product that will scale the matrix
        left, right = generate_spy_triple_product_sparse(self.mat.shape, spy_shape)

        # save existing matrix data
        mat_data_save = self.mat.data

        # replace with all ones
        self.mat.data = np.ones(self.mat.data.shape)

        # triple product
        try:
            spy = left @ self.mat @ right
        except ValueError:
            # broken matmul on some types
            temp = self.mat.asformat("coo")
            spy = left @ temp @ right

        # restore original matrix data
        self.mat.data = mat_data_save

        return np.array(spy.todense())
