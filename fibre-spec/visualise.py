import numpy as np
from matplotlib import pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch
from ccdproc import trim_image


def view_slice(image,
               window,
               axis=0,
               fig_size=(20, 10),
               title=None,
               traces=None):
    trimmed_data = trim_image(image[window[1]:window[3]+1, window[0]:window[2]+1]).data

    if axis == 1:
        trimmed_data = np.rot90(trimmed_data, k=-1)
    elif axis != 0:
        raise ValueError("Axis must be 0 or 1, got {}.".format(axis))

    projection = np.ma.sum(trimmed_data, axis=0)

    figure_ar = fig_size[0] / fig_size[1]
    slice_ar = trimmed_data.shape[1] / trimmed_data.shape[0]
    f, (ax1, ax2) = plt.subplots(2,
                                 1,
                                 sharex=True,
                                 gridspec_kw={'height_ratios': (1.5, (slice_ar / figure_ar - 1.5))})
    ax1.imshow(trimmed_data, origin='lower')
    ax2.plot(projection)
    ax1.set_xlim(0, trimmed_data.shape[1] - 1)
    ax2.set_ylim(0, projection.max() * 1.05)

    if traces:
        if axis == 1:
            mean_pos = (window[2] + window[0]) / 2
        else:
            mean_pos = (window[3] + window[1]) / 2
        ax2.scatter(traces(mean_pos), np.zeros_like(traces(mean_pos)), marker=2, color='g')

    f.set_size_inches(fig_size)
    if title:
        ax1.set_title(title)
    plt.tight_layout()


def plot_tramlines(image_data, tramlines, tramlines_bg=None):
    """
    Displays image data with the tramline extraction regions using the viridis colour map, and
    the remainder in grey.
    """
    norm = ImageNormalize(image_data,
                          interval=PercentileInterval(99.5),
                          stretch=LinearStretch(),
                          clip=False)
    spectrum_data = np.ma.array(image_data, copy=True)
    tramline_mask = np.ones(spectrum_data.shape, dtype=np.bool)
    for tramline in tramlines:
        tramline_mask[tramline] = False
    spectrum_data[tramline_mask] = np.ma.masked

    fig = plt.figure(figsize=(15, 6), tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')

    if tramlines_bg:
        background_data = np.ma.array(image_data, copy=True)
        background_mask = np.ones(background_data.shape, dtype=np.bool)
        for tramline_bg in tramlines_bg:
            background_mask[tramline_bg] = False
        background_data[background_mask] = np.ma.masked
        ax1.imshow(background_data,
                   cmap='gray_r',
                   norm=norm,
                   origin='lower')
    else:
        ax1.imshow(image_data,
                   cmap='gray_r',
                   norm=norm,
                   origin='lower')

    spectrum_image = ax1.imshow(spectrum_data,
                                cmap='viridis_r',
                                norm=norm,
                                origin='lower')
    fig.colorbar(spectrum_image)
    plt.show()
