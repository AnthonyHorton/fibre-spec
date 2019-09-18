from warnings import warn

import numpy as np
from scipy import signal, ndimage
from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter


def get_peaks(data, peak_order=3, centroid_order=1, min_peak_height=None):
    peak_indices = signal.argrelmax(data, order=peak_order)[0]
    if min_peak_height:
        peak_indices = peak_indices[data[peak_indices] >= data.max() * min_peak_height]
    if centroid_order:
        peaks = np.zeros_like(peak_indices, dtype=np.float)
        for i, peak_index in enumerate(peak_indices):
            masked_data = np.ma.array(data, mask=True)
            masked_data.mask[peak_index - centroid_order:peak_index + centroid_order + 1] = False
            peaks[i] = ndimage.measurements.center_of_mass(masked_data)[0]
        return peaks
    else:
        return peak_indices


def fill_gaps(peaks,
              number_of_peaks,
              sep_tol=0.5,
              first_peak_position=None):

    filled_peaks = np.ma.zeros(number_of_peaks)
    filled_peaks.mask = True
    separations = np.diff(peaks)
    median_spacing = np.median(separations)

    i = 0
    for peak in peaks:
        if i >= filled_peaks.size:
            # Already got all the peaks we were expecting, but have some left over?
            warn("Reached expected number of peaks with detected peaks left over.")
            break

        if i == 0:
            if not first_peak_position or ((peak - first_peak_position) <
                                           (sep_tol * median_spacing)):
                # Either don't know where first peak should be, or it's at (or before)
                # the expected position.
                filled_peaks[0] = peak
                i = 1
                continue
            else:
                # One or more peaks missing at the start.
                # Put the first one in, leave the rest (if any) to the gap filling code below.
                n_spaces = int(np.around((peak - first_peak_position) / median_spacing))
                filled_peaks[0] = peak - (n_spaces * median_spacing)
                filled_peaks[0] = np.ma.masked
                i = 1

        # Use .data attribute to avoid trouble if filled_peaks[i - 1] is masked.
        gap = peak - filled_peaks.data[i - 1]
        if gap < (1 + sep_tol) * median_spacing:
            # Gap 'normal' (or small)
            filled_peaks[i] = peak
            i += 1
        else:
            # Gap bigger than normal, some number of missing fibres
            n_spaces = int(np.around(gap / median_spacing))
            mean_sep = gap / n_spaces

            # Fill in missing peak positions based on assumed number of missing fibres
            # and size of gap.
            filled_peaks[i:i + n_spaces - 1] = np.arange(mean_sep,
                                                         mean_sep * n_spaces * (1 - 1e-12),
                                                         mean_sep) + filled_peaks[i - 1]
            # Mask those entries to show that they are assumed not detected peaks
            filled_peaks[i:i + n_spaces - 1] = np.ma.masked

            filled_peaks[i + n_spaces - 1] = peak
            i += n_spaces

    # If there are peaks missing on the end need to add them.
    if i < filled_peaks.size:
        filled_peaks[i:] = np.arange(median_spacing,
                                     median_spacing * (filled_peaks.size - i + 1),
                                     median_spacing) + filled_peaks[i - 1]
        filled_peaks[i:] = np.ma.masked

    return filled_peaks


def find_taipan_peaks(data,
                      n_blocks=2,
                      n_fibres_per_block=75,
                      peak_order=3,
                      centroid_order=1,
                      min_peak_height=0.05,
                      sep_tol=0.5,
                      first_peak_position=None):
    """
    TAIPAN currently had two slit blocks with a gap between them (will be 4 in TAIPAN 300),
    and some number of Starbugs are absent for repairs. Consequently need to split the data
    up into slit blocks to find the fibres seperately, and detect and fill in the gaps where
    Starbugs are missing.
    """
    block_width = data.size // n_blocks
    peak_blocks = []

    for block_number in range(n_blocks):
        data_chunk = data[block_number * block_width:(block_number + 1) * block_width]
        peaks = get_peaks(data_chunk,
                          peak_order=peak_order,
                          centroid_order=centroid_order,
                          min_peak_height=min_peak_height)
        filled_peaks = fill_gaps(peaks,
                                 number_of_peaks=n_fibres_per_block,
                                 sep_tol=sep_tol,
                                 first_peak_position=first_peak_position)
        # Trickery required to add to masked elements.
        aligned_peaks = np.ma.array(filled_peaks.data + (block_number * block_width),
                                    mask=filled_peaks.mask)
        peak_blocks.append(aligned_peaks)

    return np.ma.concatenate(peak_blocks)


def trace_fibres(data,
                 slice_positions,
                 slice_width,
                 polynomial_order=2,
                 spectral_axis=0,
                 first_peak_position=None,
                 peak_kwargs={}):

    if polynomial_order >= len(slice_positions):
        warn("Polynomial order ({}) >= number of slices ({}), under-constrained.".format(
            polynomial_order, len(slice_positions)))

    try:
        # Convert CCDData or similar to masked array
        data = np.ma.array(data.data, mask=data.mask)
    except AttributeError:
        data = np.ma.array(data)

    slice_half_width = slice_width // 2
    slice_peaks = []

    for i, slice_position in enumerate(slice_positions):
        start = slice_position - slice_half_width
        stop = slice_position + slice_half_width + 1
        if spectral_axis == 0:
            data_slice = data[start:stop, :]
        else:
            data_slice = data[:, start:stop]
        projection = data_slice.sum(axis=spectral_axis)
        if first_peak_position:
            peak_kwargs.update({'first_peak_position': first_peak_position})
        peaks = find_taipan_peaks(projection, **peak_kwargs)
        # Store first peak from peak finder to use for next slice
        first_peak_position = peaks.data[0]
        slice_peaks.append(peaks)

    slice_peaks = np.array(slice_peaks)
    n_fibres = slice_peaks.shape[1]

    traces_init = Polynomial1D(degree=polynomial_order, n_models=n_fibres, c0=slice_peaks[0])
    trace_fitter = LinearLSQFitter()
    traces = trace_fitter(traces_init, slice_positions, slice_peaks.T)

    return traces
