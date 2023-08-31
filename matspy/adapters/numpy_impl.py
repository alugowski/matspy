# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from scipy.sparse import csr_matrix

from . import describe, MatrixSpyAdapter
from .scipy_impl import SciPySpy


class NumPySpy(MatrixSpyAdapter):
    def __init__(self, arr):
        super().__init__()
        if len(arr.shape) != 2:
            raise ValueError("Only 2D arrays are supported")
        self.arr = arr

    def get_shape(self) -> tuple:
        return self.arr.shape

    def describe(self) -> str:
        format_name = "array"

        return describe(shape=self.arr.shape, nz_type=self.arr.dtype,
                        notes=f"{format_name}")

    def get_spy(self, spy_shape: tuple) -> np.array:
        precision = self.get_option("precision", None)

        if self.arr.dtype == 'object':
            not_none = (self.arr != np.array([None]))
            if precision:
                arr = self.arr
                if not np.all(not_none):
                    # avoid comparisons to None by making a copy and replacing None with 0
                    arr = arr.copy()
                    arr[arr == np.array([None])] = 0
                mask = (arr > precision) | (arr < -precision)
            else:
                mask = (self.arr != 0) & not_none
        else:
            if precision:
                mask = (self.arr > precision) | (self.arr < -precision)
            else:
                mask = (self.arr != 0)

        return SciPySpy(csr_matrix(mask)).get_spy(spy_shape)
