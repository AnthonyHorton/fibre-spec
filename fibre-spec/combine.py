from warnings import warn

import numpy as np
from astropy import units as u
from ccdproc import ImageFileCollection, Combiner, subtract_dark, trim_image


def make_master(directory, obstype, exposure_time, window, low_thresh=3, high_thresh=3):
    ifc = ImageFileCollection(directory, keywords=['obstype', 'totalexp'])
    images = [trim_image(ccddata[window[1]:window[3]+1, window[0]:window[2]+1]) for ccddata in
              ifc.ccds(obstype=obstype, totalexp=exposure_time, ccd_kwargs={'unit': 'adu'})]
    print("Found {} {} files with {} exposure time.".format(len(images), obstype, exposure_time))
    combiner = Combiner(images)
    combiner.sigma_clipping(low_thresh=low_thresh, high_thresh=high_thresh, func=np.ma.average)
    master = combiner.average_combine()
    master.header['totalexp'] = exposure_time
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
    s = "Found {} files with exposurs in {} & 99.5 percentiles in {}-{}.".format(len(images),
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
