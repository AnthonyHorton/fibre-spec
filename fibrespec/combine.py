"""Basic processing and combining of image data."""
from warnings import warn
from pathlib import Path

import numpy as np
from astropy import units as u
import ccdproc
from ccdproc import CCDData, ImageFileCollection, Combiner


def process_fits(fitspath, *,
                 obstype=None,
                 object=None,
                 exposure_times=None,
                 percentile=None,
                 percentile_min=None,
                 percentile_max=None,
                 window=None,
                 darks=None,
                 cosmic_ray=False,
                 cosmic_ray_kwargs={},
                 gain=None,
                 readnoise=None,
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
    cosmic_ray: bool, optional
        Whether to perform single image cosmic ray removal, using the lacosmic algorithm,
        default False. Requires both gain and readnoise to be set.
    cosmic_ray_kwargs: dict, optional
        Additional keyword arguments to pass to the ccdproc.cosmicray_lacosmic function.
    gain: str or astropy.units.Quantity, optional
        Either a string indicating the FITS keyword corresponding to the (inverse gain), or
        a Quantity containing the gain value to use. If both gain and read noise are given
        an uncertainty frame will be created.
    readnoise: str or astropy.units.Quantity, optional
        Either a string indicating the FITS keyword corresponding to read noise, or a Quantity
        containing the read noise value to use. If both read noise and gain are given then an
        uncertainty frame will be created.
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
            try:
                dark_data = CCDData.read(filename)
            except ValueError:
                # Might be no units in FITS header. Assume ADU.
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
        filenames = [Path(ifc.location).joinpath(filename) for filename in ifc.files]
    else:
        raise ValueError("fitspath '{}' is not an accessible file or directory.".format(fitspath))

    # Load image(s) and process them.
    images = []
    for filename in filenames:
        try:
            ccddata = CCDData.read(filename)
        except ValueError:
            # Might be no units in FITS header. Assume ADU.
            ccddata = CCDData.read(filename, unit='adu')
        # Filtering by exposure times here because it's hard filter ImageFileCollection
        # with an indeterminate number of possible values.
        if not exposure_times or ccddata.header['totalexp'] in exposure_times:
            if window:
                ccddata = ccdproc.trim_image(ccddata[window[1]:window[3]+1, window[0]:window[2]+1])

            if percentile:
                # Check percentile value is within specified range, otherwise skip to next image.
                percentile_value = np.percentile(ccddata.data, percentile)
                if percentile_value < percentile_min or percentile_value > percentile_max:
                    continue

            if darks:
                try:
                    ccddata = ccdproc.subtract_dark(ccddata,
                                                    dark_dict[ccddata.header['totalexp']],
                                                    exposure_time='totalexp',
                                                    exposure_unit=u.second)
                except KeyError:
                    raise RuntimeError("No dark with matching totalexp for {}.".format(filename))

            if gain:
                if isinstance(gain, str):
                    egain = ccddata.header[gain]
                    egain = egain * u.electron / u.adu
                elif isinstance(gain, u.Quantity):
                    try:
                        egain = gain.to(u.electron / u.adu)
                    except u.UnitsError:
                        egain = (1 / gain).to(u.electron / u.adu)
                else:
                    raise ValueError(f"gain must be a string or Quantity, got {gain}.")

            if readnoise:
                if isinstance(readnoise, str):
                    rn = ccddata.header[readnoise]
                    rn = rn * u.electron
                elif isinstance(readnoise, u.Quantity):
                    try:
                        rn = readnoise.to(u.electron / u.pixel)
                    except u.UnitsError:
                        rn = (readnoise * u.pixel).to(u.electron)
                else:
                    raise ValueError(f"readnoise must be a string or Quantity, got {readnoise}.")

            if gain and readnoise:
                ccddata = ccdproc.create_deviation(ccddata,
                                                   gain=egain,
                                                   readnoise=rn,
                                                   disregard_nan=True)

            if gain:
                ccddata = ccdproc.gain_correct(ccddata, gain=egain)

            if cosmic_ray:
                if not gain and readnoise:
                    raise ValueError("Cosmic ray removal required both gain & readnoise.")

                ccddata = ccdproc.cosmicray_lacosmic(ccddata,
                                                     gain=1.0,  # ccddata already gain corrected
                                                     readnoise=rn,
                                                     **cosmic_ray_kwargs)

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
