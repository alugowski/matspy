from typing import Tuple

import numpy as np
import graphblas as gb

from . import describe, generate_spy_triple_product
from . import MatrixSpyAdapter


def generate_spy_triple_product_gb(matrix_shape, spy_shape) -> Tuple[gb.Matrix, gb.Matrix]:
    # construct a triple product that will scale the matrix
    left, right = generate_spy_triple_product(matrix_shape, spy_shape)

    left_shape, (left_rows, left_cols) = left
    right_shape, (right_rows, right_cols) = right

    left_mat = gb.Matrix.from_coo(
        left_rows, left_cols, np.ones(len(left_rows)),
        nrows=left_shape[0], ncols=left_shape[1],
        dtype='int64'
    )

    right_mat = gb.Matrix.from_coo(
        right_rows, right_cols, np.ones(len(right_rows)),
        nrows=right_shape[0], ncols=right_shape[1],
        dtype='int64'
    )

    del left
    del right

    return left_mat, right_mat


class GraphBLASSpy(MatrixSpyAdapter):
    def __init__(self, mat):
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def get_format(self, is_transposed=False):
        x = self.mat
        try:
            # SS, SuiteSparse-specific: format (ends with "r" or "c"), and is_iso
            fmt = x.ss.format
            if is_transposed:
                fmt = fmt[:-1] + ("c" if fmt[-1] == "r" else "r")
            if x.ss.is_iso:
                return f"{fmt} (iso)"
            return fmt
        except AttributeError:
            return None

    def describe(self) -> str:
        parts = [f"gb.{type(self.mat).__name__}", f"'{self.mat.dtype}'"]

        fmt = self.get_format()
        if fmt:
            parts.append(fmt)

        return describe(shape=self.mat.shape,
                        nnz=self.mat.nvals,
                        notes=", ".join(parts))

    def get_spy(self, spy_shape):
        # construct a triple product that will scale the matrix
        left, right = generate_spy_triple_product_gb(self.mat.shape, spy_shape)

        # Construct a pattern view of the matrix
        pattern = gb.Matrix('int64', nrows=self.mat.nrows, ncols=self.mat.ncols)
        pattern(mask=self.mat.S) << 1

        # construct result
        spy = gb.Matrix(float, nrows=spy_shape[0], ncols=spy_shape[1])

        # triple product
        spy << gb.semiring.plus_times(left @ pattern @ right)

        return spy.to_dense(fill_value=0, dtype=spy.dtype)
