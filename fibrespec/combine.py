"""Basic processing and combining of image data."""
from warnings import warn
from pathlib import Path

import numpy as np
from astropy import units as u
from ccdproc import CCDData, ImageFileCollection, Combiner, subtract_dark, trim_image


def process_fits(fitspath, *,
                 obstype=None,
                 object=None,
                 exposure_times=None,
                 percentile=None,
                 percentile_min=None,
                 percentile_max=None,
                 window=None,
                 darks=None,
                 normalise=False,
                 normalise_func=np.ma.average,
                 combine_type=None,
                 sigma_clip=False,
                 low_thresh=3,
                 high_thresh=3):
    """Combine all FITS images of a given type and exposure time from a given directory.

    Parameters
    ----------
    fitspath: str
        Path to the FITS images to process. Can be a path to a single file, or a path to a
        directory. If the latter the directory will be searched for FITS files and checked
        against criteria from obstype, object, exposure_times critera.
    obstype: str, optional
        Observation type, an 'OBSTYPE' FITS header value e.g. 'DARK', 'OBJ'. If given only files
        with matching OBSTYPE will be processed.
    object: str, optional
        Object name, i.e. 'OBJECT' FITS header value. If given only files with matching OBJECT
        will be processed.
    exposure_times: float or sequence, optional
        Exposure time(s), i.e 'TOTALEXP' FITS header value(s). If given only files with matching
        TOTALEXP will be processed.
    percentile: float, optional
        If given will only images whose percentile value fall between percentile_min and
        percentile_max will be processed, e.g. set to 50.0 to select images by median value,
        set to 99.5 to select images by their 99.5th percentile value.
    percentile_min: float, optional
        Minimum percentile value.
    percentile_max: float, optional
        Maximum percentil value.
    window: (int, int, int, int), optional
        If given will trim images to the window defined as (x0, y0, x1, y1), where (x0, y0)
        and (x1, y1) are the coordinates of the bottom left and top right corners.
    darks: str or sequence, optional
        Filename(s) of dark frame(s) to subtract from the image(s). If given a dark frame with
        matching TOTALEXP will be subtracted from each image during processing.
    normalise: bool, optional
        If True each image will be normalised. Default False.
    normalise_func: callable, optional
        Function to use for normalisation. Each image will be divided by normalise_func(image).
        Default np.ma.average.
    combine_type: str, optional
        Type of image combination to use, 'MEAN' or 'MEDIAN'. If None the individual
        images will be processed but not combined and the return value will be a list of
        CCDData objects. Default None.
    sigma_clip: bool, optional
        If True will perform sigma clipping on the image stack before combining, default=False.
    low_thresh: float, optional
        Lower threshold to use for sigma clipping, in standard deviations. Default is 3.0.
    high_thresh: float, optional
        Upper threshold to use for sigma clipping, in standard deviations. Default is 3.0.


    Returns
    -------
    master: ccdproc.CCDData
        Combined image.

    """
    if exposure_times:
        try:
            # Should work for any sequence or iterable type
            exposure_times = set(exposure_times)
        except TypeError:
            # Not a sequence or iterable, try using as a single value.
            exposure_times = {float(exposure_times), }

    if darks:
        try:
            dark_filenames = set(darks)
        except TypeError:
            dark_filenames = {darks, }
        dark_dict = {}
        for filename in dark_filenames:
            dark_data = CCDData.read(filename, unit='adu')
            dark_dict[dark_data.header['totalexp']] = dark_data

    if combine_type and combine_type not in ('MEAN', 'MEDIAN'):
        raise ValueError("combine_type must be 'MEAN' or 'MEDIAN', got '{}''".format(combine_type))

    fitspath = Path(fitspath)
    if fitspath.is_file():
        # FITS path points to a single file, turn into a list.
        filenames = [fitspath, ]
    elif fitspath.is_dir():
        # FITS path is a directory. Find FITS file and collect values of selected FITS headers
        ifc = ImageFileCollection(fitspath, keywords='*')
        if len(ifc.files) == 0:
            raise RuntimeError("No FITS files found in {}".format(fitspath))
        # Filter by observation type.
        if obstype:
            try:
                ifc = ifc.filter(obstype=obstype)
            except FileNotFoundError:
                raise RuntimeError("No FITS files with OBSTYPE={}.".format(obstype))
        # Filter by object name.
        if object:
            try:
                ifc = ifc.filter(object=object)
            except FileNotFoundError:
                raise RuntimeError("No FITS files with OBJECT={}.".format(object))
        filenames = ifc.files
    else:
        raise ValueError("fitspath '{}' is not an accessible file or directory.".format(fitspath))

    # Load image(s) and process them.
    images = []
    for filename in filenames:
        ccddata = CCDData.read(filename, unit='adu')
        # Filtering by exposure times here because it's hard filter ImageFileCollection
        # with an indeterminate number of possible values.
        if not exposure_times or ccddata.header['totalexp'] in exposure_times:
            if window:
                ccddata = trim_image(ccddata[window[1]:window[3]+1, window[0]:window[2]+1])

            if percentile:
                # Check percentile value is within specified range, otherwise skip to next image.
                percentile_value = np.percentile(ccddata.data, percentile)
                if percentile_value < percentile_min or percentile_value > percentile_max:
                    continue

            if darks:
                try:
                    ccddata = subtract_dark(ccddata,
                                            dark_dict[ccddata.header['totalexp']],
                                            exposure_time='totalexp',
                                            exposure_unit=u.second)
                except KeyError:
                    raise RuntimeError("No dark with matching totalexp for {}.".format(filename))

            if normalise:
                ccddata = ccddata.divide(normalise_func(ccddata.data))

            images.append(ccddata)

    n_images = len(images)
    if n_images == 0:
        msg = "No FITS files match exposure time criteria"
        raise RuntimeError(msg)

    if n_images == 1 and combine_type:
        warn("Combine type '{}' selected but only 1 matching image, skipping image combination.'")
        combine_type = None

    if combine_type:
        combiner = Combiner(images)

        # Sigma clip data
        if sigma_clip:
            if combine_type == 'MEAN':
                central_func = np.ma.average
            else:
                # If not MEAN has to be MEDIAN, checked earlier that it was one or the other.
                central_func = np.ma.median
            combiner.sigma_clipping(low_thresh=low_thresh,
                                    high_thresh=high_thresh,
                                    func=central_func)

        # Stack images.
        if combine_type == 'MEAN':
            master = combiner.average_combine()
        else:
            master = combiner.median_combine()

        # Populate header of combined image with metadata about the processing.
        master.header['fitspath'] = str(fitspath)
        if obstype:
            master.header['obstype'] = obstype
        if exposure_times:
            if len(exposure_times) == 1:
                master.header['totalexp'] = float(exposure_times.pop())
            else:
                master.header['totalexp'] = tuple(exposure_times)
        master.header['nimages'] = n_images
        master.header['combtype'] = combine_type
        master.header['sigclip'] = sigma_clip
        if sigma_clip:
            master.header['lowclip'] = low_thresh
            master.header['highclip'] = high_thresh

    else:
        # No image combination, just processing indivudal image(s)
        if n_images == 1:
            master = images[0]
        else:
            master = images

    return master


def make_flat(directory,
              master_darks,
              window,
              max_value=30000,
              min_value=10000,
              low_thresh=3,
              high_thresh=3):
    exposure_times = list(master_darks.keys())
    ifc = ImageFileCollection(directory, keywords=['obstype', 'totalexp'])
    images = []
    for ccddata in ifc.ccds(obstype='SFLAT', ccd_kwargs={'unit': 'adu'}):
        if ccddata.header['totalexp'] in exposure_times:
            ccddata = trim_image(ccddata[window[1]:window[3]+1, window[0]:window[2]+1])
            nnpf = np.percentile(ccddata.data, 99.5)
            if nnpf < max_value and nnpf > min_value:
                ccddata = subtract_dark(ccddata,
                                        master_darks[ccddata.header['totalexp']],
                                        exposure_time='totalexp',
                                        exposure_unit=u.second)
                ccddata = ccddata.divide(np.ma.average(ccddata.data))
                images.append(ccddata)
    s = "Found {} files with exposures in {} & 99.5 percentiles in {}-{}.".format(len(images),
                                                                                  exposure_times,
                                                                                  min_value,
                                                                                  max_value)
    print(s)
    combiner = Combiner(images)
    combiner.sigma_clipping(low_thresh=low_thresh, high_thresh=high_thresh, func=np.ma.average)
    master_flat = combiner.average_combine()
    return master_flat


def mask_bad_pixels(image_data, pixel_mask):
    try:
        bad_pixels = np.loadtxt(pixel_mask, dtype=np.int)
    except Exception as err:
        warn("Couldn't read bad pixel mask from {}".format(pixel_mask))
        raise err
    print("Applying bad pixel mask from {}\n".format(pixel_mask))
    bad_pixels = (bad_pixels[0], bad_pixels[1])
    image_data = np.ma.array(image_data)
    image_data[bad_pixels] = np.ma.masked
    return image_data
