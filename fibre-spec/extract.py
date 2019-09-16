import numpy as np


def initialise_tramlines(tram_coef_file, window, width, background_width=None):
    """
    Used tramline fit from tram_coef to produce a list of tuples of arrays, each tuple contains
    the y and x coordinates of the pixels to include in the extraction for a given fibre. When
    formatted in this way each item in the list can be used directly to index the image data.

    Args:
        width (int): width, in pixels, of the spectral extraction region
        background_width (int): total width, in pixels, of the spectral extraction regions including
            the instrument background subtraction regions.

    Returns:
        tramlines (list of tuples of np.array): pixels coordinates for each fibre
        tramlines_bg (list of tuples of np.array): pixel coordinates for each fibre, including
            background regions.
    """
    try:
        tram_coef = np.loadtxt(tram_coef_file, delimiter=',')
    except Exception as err:
        warn("Couldn't read tramline coefficients file {}".format(tram_coef_file))
        raise err

    print("Initialising tramlines with coefficients from {}\n".format(tram_coef_file))

    xs = np.arange(0, window[1][1] - window[1][0])

    x_grid, y_grid = np.meshgrid(xs, np.arange(width))
    tramlines = []

    if background_width:
        x_grid_bg, y_grid_bg = np.meshgrid(xs, np.arange(background_width))
        tramlines_bg = []

    for (a, b, c) in tram_coef:
        if np.isnan(a):
            tramlines.append([])
            if background_width:
                tramlines_bg.append([])
            continue
        # Calculate curve
        ys = a + b * xs + c * xs**2
        # Calculate set of y shifted versions to get desired width
        ys_spectrum = ys.reshape((1, ys.shape[0]))
        ys_spectrum = ys_spectrum + np.arange(-(width - 1)/2, (width + 1)/2).reshape((width, 1))

        # Round to integers
        ys_spectrum = np.around(ys_spectrum, decimals=0).astype(np.int)

        # Reshape into (y coords, x coords) for numpy indexing
        tramline = (ys_spectrum.ravel(), x_grid.ravel())
        tramlines.append(tramline)

        if background_width:
            ys_bg = ys.reshape((1, ys.shape[0]))
            ys_bg = ys_bg + np.arange(-(background_width - 1)/2,
                                      (background_width + 1)/2).reshape((background_width, 1))
            ys_bg = np.around(ys_bg, decimals=0).astype(np.int)
            tramline_bg = (ys_bg.ravel(), x_grid_bg.ravel())
            tramlines_bg.append(tramline_bg)

    # Fibre number increases with decreasing y, but tram_coef is in order of increasing y.
    tramlines.reverse()

    if background_width:
        tramlines_bg.reverse()
        return tramlines, tramlines_bg

    return tramlines


def extract_spectrum(image_data, tramline, tramline_bg=None):
    """
    Crude, uncalibrated spectral extraction (just masks image & collapses in y direction)

    Args:
        image_data (np.array): 2D data
        tramline (tuples of np.array): y, x coordinates of the spectrum extraction region
        tramline_bg (tuples of np.array, optional):

    Returns:
        spectrum (np.array): 1D spectrum
    """
    spectrum_data = np.ma.array(image_data, copy=True)
    tramline_mask = np.ones(spectrum_data.shape, dtype=np.bool)
    tramline_mask[tramline] = False
    spectrum_data[tramline_mask] = np.ma.masked
    spectrum = spectrum_data.mean(axis=0)

    if tramline_bg:
        background_data = np.ma.array(image_data, copy=True)
        tramline_bg_mask = np.ones(background_data.shape, dtype=np.bool)
        tramline_bg_mask[tramline_bg] = False
        tramline_bg_mask[tramline] = True
        background_data[tramline_bg_mask] = np.ma.masked
        spectrum = spectrum - background_data.mean(axis=0)

    return spectrum.filled(fill_value=np.nan)
