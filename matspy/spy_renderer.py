# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

from .adapters import MatrixSpyAdapter
# noinspection PyProtectedMember
from matspy import params, to_spy_heatmap, _get_spy_adapter


def _get_relative_max(heatmap, k):
    ret = None

    if min(heatmap.shape) > 3:
        # slice off the final row/column because those can be artificially high due to aliasing effects
        corner = heatmap[0:(heatmap.shape[0]-1), 0:(heatmap.shape[1]-1)]
        ret = _top_k(corner, k)

    if not ret:
        ret = _top_k(heatmap, k)

    return ret if ret else 1


def _top_k(arr, k):
    arr = arr.flatten()

    if arr.size == 0:
        return None

    if k == 1:
        ret = np.max(arr, initial=0)
        return ret if ret else None

    k = min(k, arr.size - 1)
    srt = np.sort(arr)

    for top in srt[-k:]:
        if top > 0:
            return top

    return None


def _rescale(arr, from_range, to_range):
    from_size = from_range[1] - from_range[0]
    to_size = to_range[1] - to_range[0]

    if from_size == 0:
        return np.full_like(arr, to_range[1])

    return (arr - from_range[0]) * (to_size / from_size) + to_range[0]


# noinspection PyUnusedLocal
def get_spy_heatmap(adapter: MatrixSpyAdapter, buckets, shading, shading_absolute_min,
                    shading_relative_min, shading_relative_max_percentile, **kwargs):
    # find spy matrix shape
    mat_shape = adapter.get_shape()
    ratio = buckets / max(mat_shape)
    spy_shape = tuple(max(1, int(ratio * x)) for x in mat_shape)

    dense = adapter.get_spy(spy_shape=spy_shape)

    dense[dense < 0] = 0

    # scale values
    if shading == "absolute":
        divisor = max(adapter.get_shape()) / buckets
        divisor *= divisor  # area
        dense /= divisor
        dense[(0 < dense) & (dense < shading_absolute_min)] = shading_absolute_min
        dense[dense > 1] = 1
    elif shading == "relative":
        mask = dense > 0

        small = np.min(dense[mask], initial=0)

        nnz = dense[mask].flatten().size
        k = max(1, nnz - int(nnz * shading_relative_max_percentile))
        big = _get_relative_max(dense, k)

        scaled = _rescale(dense, (small, big), (shading_relative_min, 1))
        dense[mask] = scaled[mask]
        dense[dense > 1] = 1
    elif shading == "binary":
        dense[dense != 0] = 1
    else:
        raise ValueError("shading must be one of 'absolute', 'relative', 'binary'")

    return dense


def _get_spy_cmap(options):
    return LinearSegmentedColormap.from_list("spy_cmap", [options.color_empty, options.color_full])


def _tweak_divisor(num, divisor, lower=0.2, higher=0.5):
    if num <= divisor:
        return num

    bucket_candidates = \
        list(range(divisor + 1, divisor + min(int(divisor * higher), 200))) + \
        list(range(divisor - 1, divisor - min(int(divisor * lower), 200), -1))

    best_remainder, best_candidate = (num % divisor, divisor)

    for candidate in bucket_candidates:
        if candidate < 1:
            continue

        extra = num % candidate
        if extra < best_remainder:
            best_remainder = extra
            best_candidate = candidate

    return best_candidate


def _resize_figure_to_match_dpi(fig, target_dpi):
    fig_width, fig_height = fig.get_size_inches()
    plot_frac_width = fig.subplotpars.right - fig.subplotpars.left
    plot_frac_height = fig.subplotpars.top - fig.subplotpars.bottom

    plot_width = fig_width * plot_frac_width
    plot_height = fig_height * plot_frac_height

    ratio = target_dpi / fig.dpi
    target_plot_width = plot_width * ratio
    target_plot_height = plot_height * ratio

    fig.set_size_inches(fig_width + (target_plot_width - plot_width),
                        fig_height + (target_plot_height - plot_height))


def spy_to_mpl(mat, **kwargs):
    """
    Create a spy plot and return as matplotlib figure without showing.
    """
    options = params.get(**kwargs)
    adapter = _get_spy_adapter(mat)

    fig, ax = plt.subplots()
    fig.set_size_inches(options.figsize, options.figsize)
    if options.indices:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=0, nbins='auto'))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=0, nbins='auto'))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_ylim(adapter.get_shape()[0], 0)
    ax.set_xlim(0, adapter.get_shape()[1])

    if options.title is True:
        options.title = adapter.describe()
    if options.title:
        plt.title(options.title)

    plt.tight_layout()

    max_dim = max(adapter.get_shape())
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_dim_max_pixels = max(bbox.width, bbox.height) * fig.dpi

    if options.buckets:
        # explicit bucket size from the user
        pass
    elif options.dpi:
        # explicit dpi from the user
        options.buckets = int(options.dpi * options.figsize)
    else:
        # from matplotlib figure dimensions
        options.buckets = int(fig_dim_max_pixels / 2)

    if options.spy_aa_tweaks_enabled:
        # tweak the bucket size to better fit the matrix
        options.buckets = _tweak_divisor(max_dim, options.buckets, lower=0.2, higher=0.2)

        # tweak the figure size to better fit the bucket count
        new_dpi = _tweak_divisor(options.buckets, int(fig.dpi), lower=0.1, higher=0.1)
        _resize_figure_to_match_dpi(fig, new_dpi)

    interpolation = "bilinear" if fig_dim_max_pixels / options.buckets < 1.2 else "nearest"

    ax.imshow(to_spy_heatmap(adapter, **options.to_kwargs()),
              cmap=_get_spy_cmap(options),
              interpolation=interpolation, interpolation_stage="rgba", aspect="equal", origin="upper", vmin=0, vmax=1,
              extent=[0, adapter.get_shape()[1], adapter.get_shape()[0], 0])

    return fig, ax


def spy(mat, **kwargs):
    fig, ax = spy_to_mpl(mat, **kwargs)
    plt.show()
    plt.close(fig)


def to_sparkline(mat, retscale=False, scale=None, html_border="1px solid black", **kwargs):
    options = params.get(**kwargs)
    adapter = _get_spy_adapter(mat)

    max_dim = max(adapter.get_shape())
    if scale is None:
        scale = options.sparkline_size / max_dim
    options.figsize = scale * max_dim
    sizing_dpi = plt.rcParams["figure.dpi"]

    img_height, img_width = tuple(int((dim / max_dim) * options.figsize * sizing_dpi) for dim in adapter.get_shape())

    if not options.dpi:
        # no explicit dpi from the user, use matplotlib default
        options.dpi = plt.rcParams["figure.dpi"]

    if options.buckets:
        # user-specified bucket size
        options.dpi = options.buckets / options.figsize
    else:
        # auto select bucket size
        options.buckets = int(options.dpi * options.figsize)

    # If the bucket size does not evenly divide the matrix dimensions then
    # there may be visible artifacts like banding in the spy image. These artifacts can
    # give the impression of structure that isn't there. Some tweaks to parameters may alleviate this.
    if options.spy_aa_tweaks_enabled:
        # tweak the bucket size to better fit the matrix
        options.buckets = _tweak_divisor(max_dim, options.buckets, lower=0.5, higher=0.5)

    repeat = max(img_height, img_width) / options.buckets
    repeat = int(repeat) if repeat >= 2 else 1

    heatmap = to_spy_heatmap(adapter, **options.to_kwargs())
    if repeat > 1:
        heatmap = heatmap.repeat(repeat, axis=0)
        heatmap = heatmap.repeat(repeat, axis=1)
    spy_cmap = _get_spy_cmap(options)
    image = spy_cmap(heatmap)

    from io import BytesIO
    import base64
    bio = BytesIO()
    plt.imsave(bio, image, format="png", origin="upper", vmin=0, vmax=1, dpi=(options.dpi*repeat))
    encoded = base64.b64encode(bio.getvalue()).decode()
    style = f' style="border: {html_border};"' if html_border else ''
    sparkline = f'<img src="data:image/png;base64,{encoded}"{style} width={img_width} height={img_height}/>'

    if retscale:
        return sparkline, scale
    else:
        return sparkline
