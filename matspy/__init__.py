# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import dataclasses
from dataclasses import dataclass, asdict
from typing import Type, Tuple, Dict, List, Union

from .adapters import Driver, MatrixSpyAdapter


@dataclass
class MatSpyParams:
    indices: bool = True
    """Whether to show row/column indices."""

    title: Union[bool, str] = True
    """Title of spy plot. If `True` then generate matrix description such as dimensions, nnz, datatype."""

    shading: str = "relative"
    """
    How to shade buckets:
     - `'binary'`: A nonempty bucket is considered full.
     - `'relative'`: A bucket is shaded relative to the fullness of the fullest bucket in the matrix.
     - `'absolute'`: A bucket is shaded according to how full it is.
    """

    shading_absolute_min: float = 0.2
    """
    if `shading == 'absolute'`: buckets with values less than this will be clipped to this value
    so they may be visible in the plot.
    """

    shading_relative_min: float = 0.4
    """
    if `shading == 'relative'`: the lightest non-zeros have this value.
    """

    shading_relative_max_percentile: float = 0.99
    """
    if `shading == 'relative'`: define what a 'full' bucket is as a percentile of the nonzero buckets.
    A simple max would allow one or two outliers to skew the entire range making the plot appear too light.
    """

    figsize: float = 3.5
    """Figure size for spy plots, of longest side, in default matplotlib units (inches)."""

    sparkline_size: float = 1
    """Figure size for sparklines, of longest side, in default matplotlib units (inches)."""

    dpi: float = None
    """Default spy image DPI. If None the matplotlib default Figure dpi is used."""

    buckets: int = None
    """Pixel count of longest side of spy image. If None then computed from size and DPI."""

    spy_aa_tweaks_enabled: bool = None
    """
    Whether to_sparkline() may tweak parameters like bucket count to prevent visible aliasing artifacts.
    If None then defaults to True if dpi and buckets are also None.
    """

    color_empty: Union[Tuple[float, float, float, float], str] = (1.0, 1.0, 1.0, 1.0)  # RGBA: empty space is white
    """Color for empty space. Can be anything matplotlib accepts, like RGB or RGBA tuples."""

    color_full: Union[Tuple[float, float, float, float], str] = (0.0, 0.0, 1.0, 1.0)  # RGBA: non-zeros are blue
    """Color for a full bucket. Can be anything matplotlib accepts, like RGB or RGBA tuples."""

    def _assert_one_of(self, var, choices):
        if getattr(self, var) not in choices:
            raise ValueError(f"{var} must be one of: " + ", ".join(choices))

    def get(self, **kwargs):
        ret = dataclasses.replace(self)

        # Allow some explicit overwrites for convenience
        if "title" in kwargs and "title_latex" not in kwargs:
            kwargs["title_latex"] = kwargs["title"]

        if "figsize" in kwargs and "sparkline_size" not in kwargs:
            kwargs["sparkline_size"] = kwargs["figsize"]

        # Update all parameters with the ones in kwargs
        for key, value in kwargs.items():
            if hasattr(ret, key):
                setattr(ret, key, value)

        # validate
        ret._assert_one_of("shading", ['relative', 'absolute', 'binary'])

        # Apply some default rules
        if ret.spy_aa_tweaks_enabled is None:
            ret.spy_aa_tweaks_enabled = ret.buckets is None and ret.dpi is None

        return ret

    def to_kwargs(self):
        return asdict(self)


params = MatSpyParams()
_drivers: List[Type[Driver]] = []
_driver_prefixes: Dict[str, Type[Driver]] = {}


def register_driver(driver: Type[Driver]):
    _drivers.append(driver)

    for prefix in driver.get_supported_type_prefixes():
        _driver_prefixes[prefix] = driver


def _register_bundled():
    """
    Register the built-in drivers.
    """
    from .adapters.scipy_driver import SciPyDriver
    register_driver(SciPyDriver)

    from .adapters.graphblas_driver import GraphBLASDriver
    register_driver(GraphBLASDriver)


_register_bundled()


def _get_driver(mat):
    type_str = ".".join((mat.__module__, mat.__class__.__name__))
    for prefix, driver in _driver_prefixes.items():
        if type_str.startswith(prefix):
            return driver

    raise AttributeError("Unsupported type: " + type_str)


def _get_spy_adapter(mat) -> MatrixSpyAdapter:
    if isinstance(mat, MatrixSpyAdapter):
        return mat

    adapter = _get_driver(mat).adapt_spy(mat)
    if not adapter:
        raise AttributeError("Unsupported matrix")

    return adapter


def to_spy_heatmap(mat, buckets=500, **kwargs):
    options = params.get(**kwargs)
    options.buckets = buckets
    adapter = _get_spy_adapter(mat)

    from .spy_renderer import get_spy_heatmap
    heatmap = get_spy_heatmap(adapter, **options.to_kwargs())
    return heatmap


from matspy.spy_renderer import spy, spy_to_mpl, to_sparkline


__all__ = ["to_sparkline", "to_spy_heatmap", "spy_to_mpl", "spy"]
