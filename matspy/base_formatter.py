# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

unicode_dots = {
    "h": "⋯",
    "v": "⋮",
    "d": "⋱",
}


class BaseFormatter:
    def __init__(self):
        self.lines = []

    def write(self, s, indent: int = 0):
        self.lines.append(" " * indent + s)

    def __str__(self):
        return "\n".join(self.lines)
