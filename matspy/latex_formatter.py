# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import re

from .adapters import MatrixAdapter, MatrixAdapterRow, Truncated2DMatrix, to_trunc
from .base_formatter import BaseFormatter, unicode_dots


def python_scientific_to_latex_times10(s):
    return re.sub(r'e\+?(-?)0*([0-9]+)', ' \\\\times 10^{\\1\\2}', s)


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        " '": r' `',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


latex_dots = {
    "h": "\\dots",  # '⋯'
    "v": "\\vdots",  # '⋮'
    "d": "\\ddots"  # '⋱'
}

unicode_to_latex = {
    u: latex_dots[k] for k, u in unicode_dots.items()
}


class LatexFormatter(BaseFormatter):
    def __init__(self, max_rows, max_cols, num_after_dots, title_latex, latex_matrix_env,
                 float_formatter_latex=None, **_):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_after_dots = num_after_dots
        self.title = title_latex
        self.latex_matrix_env = latex_matrix_env
        self.float_formatter = float_formatter_latex
        if not self.float_formatter:
            self.float_formatter = lambda f: python_scientific_to_latex_times10(format(f))
        self.indent_width = 4

    def pprint(self, obj):
        if obj is None:
            return ""

        if isinstance(obj, float):
            return self.float_formatter(obj)

        if obj in unicode_to_latex:
            return unicode_to_latex[obj]

        return tex_escape(str(obj))

    def _write_matrix(self, mat: MatrixAdapterRow, indent: int = 0):
        if isinstance(mat, Truncated2DMatrix):
            mat.apply_dots(unicode_dots)

        nrows, ncols = mat.get_shape()
        self.write("\\begin{" + self.latex_matrix_env + "}", indent=indent)

        body_indent = indent + self.indent_width

        # values
        for row_idx in range(nrows):
            row_contents = []

            col_range = (0, ncols)
            for col_idx, cell in enumerate(mat.get_row(row_idx, col_range=col_range)):
                if cell is not None:
                    row_contents.append(self.pprint(cell))

                if col_idx == col_range[1] - 1:
                    if row_idx != nrows - 1:
                        row_contents.append("\\\\")
                else:
                    row_contents.append("&")

            self.write(" ".join(row_contents), body_indent)

        self.write("\\end{" + self.latex_matrix_env + "}", indent=indent)

    def format(self, mat: MatrixAdapter):
        if self.title:
            title = mat.describe() if self.title is True else self.title
            self.write("\\stackrel{\\textrm{" + tex_escape(title) + "}}{")

        if not isinstance(mat, MatrixAdapterRow) or \
                mat.get_shape()[0] > self.max_rows or \
                mat.get_shape()[1] > self.max_cols:
            mat = to_trunc(mat, self.max_rows, self.max_cols, self.num_after_dots)

        self._write_matrix(mat)

        if self.title:
            self.write("}")
        return self
