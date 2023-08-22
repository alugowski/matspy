# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Tuple

import numpy as np
import scipy.sparse

from . import describe, generate_spy_triple_product, MatrixSpyAdapter


def generate_spy_triple_product_coo(matrix_shape, spy_shape) -> Tuple[scipy.sparse.coo_matrix, scipy.sparse.coo_matrix]:
    # construct a triple product that will scale the matrix
    left, right = generate_spy_triple_product(matrix_shape, spy_shape)

    left_shape, (left_rows, left_cols) = left
    right_shape, (right_rows, right_cols) = right
    left_mat = scipy.sparse.coo_matrix((np.ones(len(left_rows)), (left_rows, left_cols)), shape=left_shape)
    right_mat = scipy.sparse.coo_matrix((np.ones(len(right_rows)), (right_rows, right_cols)), shape=right_shape)

    return left_mat, right_mat


class SciPySpy(MatrixSpyAdapter):
    def __init__(self, mat):
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def describe(self) -> str:
        format_name = self.mat.getformat()

        return describe(shape=self.mat.shape, nnz=self.mat.nnz, nz_type=self.mat.dtype,
                        notes=f"{format_name}")

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

        return np.array(spy.todense())
