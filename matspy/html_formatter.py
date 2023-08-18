# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import html

from .adapters import MatrixAdapter, MatrixAdapterRow, Truncated2DMatrix, to_trunc
from .base_formatter import BaseFormatter, unicode_dots


html_dots = {
    "h": "&ctdot;",  # '⋯'
    "v": "&vellip;",  # '⋮'
    "d": "&dtdot;"  # '⋱'
}

unicode_to_html = {
    u: html_dots[k] for k, u in unicode_dots.items()
}


class HTMLTableFormatter(BaseFormatter):
    def __init__(self, max_rows, max_cols, num_after_dots, title, indices=False, cell_align="center",
                 float_formatter=None, **_):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_after_dots = num_after_dots
        self.title = title
        self.indices = indices
        self.cell_align = cell_align
        self.float_formatter = float_formatter if float_formatter else lambda f: format(f)
        self.indent_width = 4
        self.left_td_class = None
        self.right_td_class = None

    # noinspection PyMethodMayBeStatic
    def _attributes_to_string(self, attributes: dict) -> str:
        if attributes is None:
            attributes = {}

        return ' '.join([f'"{html.escape(str(k))}"={html.escape(str(v))}' for k, v in attributes])

    def pprint(self, obj):
        if obj is None:
            return ""

        if isinstance(obj, float):
            return self.float_formatter(obj)

        if obj in unicode_to_html:
            return unicode_to_html[obj]

        return html.escape(str(obj))

    def _write_matrix(self, mat: MatrixAdapterRow,
                      indent: int = 0,
                      table_attributes=None):
        if isinstance(mat, Truncated2DMatrix):
            mat.apply_dots(unicode_dots)

        nrows, ncols = mat.get_shape()
        self.write(f"<table {self._attributes_to_string(table_attributes)}>", indent=indent)

        body_indent = indent + self.indent_width
        cell_indent = body_indent + self.indent_width

        # Header
        if self.indices:
            self.write("<thead>", indent=body_indent)
            self.write("<tr>", indent=body_indent)
            self.write(f"<th></th>", indent=cell_indent)
            for col_label in mat.get_col_labels():
                self.write(f"<th>{self.pprint(col_label)}</th>", indent=cell_indent)
            self.write("</tr>", indent=body_indent)
            self.write("</thead>", indent=body_indent)

        # values
        self.write("<tbody>", indent=body_indent)
        row_labels = iter(mat.get_row_labels()) if self.indices else None
        for row_idx in range(nrows):
            self.write("<tr>", body_indent)
            if self.indices:
                self.write(f"<th>{self.pprint(next(row_labels))}</th>", cell_indent)

            col_range = (0, ncols)
            for col_idx, cell in enumerate(mat.get_row(row_idx, col_range=col_range)):
                td_classes = []
                if col_idx == col_range[0] and self.left_td_class:
                    td_classes.append(self.left_td_class)
                if col_idx == col_range[1] - 1 and self.right_td_class:
                    td_classes.append(self.right_td_class)
                td_class = f' class="{" ".join(td_classes)}"' if td_classes else ""
                self.write(f"<td{td_class}>{self.pprint(cell)}</td>", cell_indent)

            self.write("</tr>", body_indent)

        self.write("</tbody>", indent=body_indent)

        self.write("</table>", indent=indent)

    def format(self, mat: MatrixAdapter):
        if self.title:
            title = mat.describe() if self.title is True else self.title
            self.write(f"<p>{title}</p>")

        if not isinstance(mat, MatrixAdapterRow) or \
                mat.get_shape()[0] > self.max_rows or \
                mat.get_shape()[1] > self.max_cols:
            mat = to_trunc(mat, self.max_rows, self.max_cols, self.num_after_dots)

        self._write_matrix(mat)

        return self


class NotebookHTMLFormatter(HTMLTableFormatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _write_style(self):
        index_attributes = {"font-size": "smaller"}
        self.write("<style scoped>")
        for tags, attributes in [
            ("thead tr th", {**index_attributes, "vertical-align": "middle", "text-align": "center"}),  # column indices
            ("tbody tr th", {**index_attributes, "vertical-align": "middle", "text-align": "right"}),  # row indices
            ("tbody tr td", {"vertical-align": "middle", "text-align": self.cell_align}),
            ("tbody tr td.left_cell", {"border-left": "solid 2px"}),
            ("tbody tr td.right_cell", {"border-right": "solid 2px"}),
            ("tbody tr td:empty::after", {"content": "'&nbsp;'", "visibility": "hidden"}),  # fill empty cells
        ]:
            self.write(f"{tags} " + '{', indent=self.indent_width)
            for k, v in attributes.items():
                self.write(f"{k}: {v};", indent=2*self.indent_width)
            self.write("}", indent=self.indent_width)
        self.write("</style>")

    def format(self, mat: MatrixAdapter):
        self.write("<div>")
        self._write_style()
        self.left_td_class = "left_cell"
        self.right_td_class = "right_cell"
        super().format(mat)
        self.write("</div>")
        return self
