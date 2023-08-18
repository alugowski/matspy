# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver, MatrixSpyAdapter


class SciPyDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, str, bool]]:
        formats = ("coo", "csr", "csc", "lil", "dok", "dia", "bsr")
        ret = []
        for mod in ("scipy.sparse", "scipy.sparse._arrays"):
            for arr_mat in ("array", "matrix"):
                for fmt in formats:
                    ret.append((mod, f"{fmt}_{arr_mat}", True))
        for fmt in formats:
            for arr_mat in ("array", "matrix"):
                ret.append((f"scipy.sparse._{fmt}", f"{fmt}_{arr_mat}", True))
            ret.append((f"scipy.sparse.{fmt}", f"{fmt}_matrix", True))
        return ret

    @staticmethod
    def adapt(mat: Any):
        import scipy.sparse
        from .scipy_impl import SciPyCSRAdapter, SciPyCSCAdapter, SciPyCOOAdapter

        if scipy.sparse.isspmatrix_csr(mat):
            return SciPyCSRAdapter(mat)
        if scipy.sparse.isspmatrix_csc(mat):
            return SciPyCSCAdapter(mat)
        else:
            return SciPyCOOAdapter(mat.tocoo())

    @staticmethod
    def adapt_spy(mat: Any) -> MatrixSpyAdapter:
        from .scipy_impl import SciPySpy
        return SciPySpy(mat)
