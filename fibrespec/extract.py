"""Spectral extraction tools."""
import numpy as np


def extract_spectra(image_data, *, traces, width, axis, reverse_spectra=False):
    """Extract fibre spectra from image data.

    Parameters
    ----------
    image_data: ccdproc.CCDData
        Image containing the fibre spectra.
    traces: astropy.modeling.Model
        Model set that when evaluated at a given pixel coordinate along the spectral axis will
        return the pixel coordinates of the fibre spectra centres in the other axis, such as
        that returned by fibrespec.trace.trace_fibres().
    width: int
        Width, in pixels, of the spectral extraction regions.
    axis: int
        Axis of the image which corresponds to the spectral direction.

    Returns
    -------
    spectra: np.ndarray
        Spectra.

    """
    spectral_length = image_data.data.shape[axis]
    n_spectra = len(traces)

    xs = np.arange(0, spectral_length)

    ys = traces(np.broadcast_to(xs, (n_spectra, spectral_length)))
    ys = np.expand_dims(ys, axis=2)
    ys = ys + np.arange(-(width - 1)/2, (width + 1)/2).reshape((1, 1, width))
    ys = np.around(ys, decimals=0).astype(np.int)

    xs = xs.reshape((1, spectral_length, 1))
    xs = np.broadcast_to(xs, (n_spectra, spectral_length, width)).astype(np.int)

    if axis == 0:
        pixel_values = image_data.data[(xs.ravel(), ys.ravel())]
    elif axis == 1:
        pixel_values = image_data.data[(ys.ravel(), xs.ravel())]
    else:
        raise ValueError("axis must be 0 or 1, got {}.".format(axis))

    pixel_values = pixel_values.reshape((n_spectra, spectral_length, width))
    spectra = pixel_values.sum(axis=2)
    if reverse_spectra:
        spectra = np.flip(spectra, axis=1)

    return spectra
