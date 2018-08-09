#! /usr/bin/env python

# does not work on multi-ext data!!!

import numpy as np
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

def crop_nans(fitsfile, write=True, inplace=False):
    with fits.open(fitsfile) as f:
        data = f[0].data
        print('Original shape: {}'.format(data.shape))
    nans = np.isnan(data)
    nancols = np.all(nans, axis=0)
    nanrows = np.all(nans, axis=1)

    firstcol = nancols.argmin()
    firstrow = nanrows.argmin()

    lastcol = len(nancols) - nancols[::-1].argmin()
    lastrow = len(nanrows) - nanrows[::-1].argmin()
    if (firstcol + firstrow == 0) & (lastrow * lastcol == data.size):
        print('No cropping needed. Exiting.')
        return
    print('Cropping to {}:{},{}:{}'.format(firstrow,lastrow,firstcol,lastcol))
    x_sum = firstcol + lastcol
    y_sum = firstrow + lastrow
    if x_sum % 2 != 0:
        x_sum -= 1
    if y_sum % 2 != 0:
        y_sum -= 1
    cutout = Cutout2D(data, position=(x_sum/2., y_sum/2.),
                      size=(lastrow-firstrow, lastcol-firstcol),
                      wcs=WCS(fitsfile), mode='trim')
    if write:
        if inplace:
            mode = 'update'
        else:
            mode = 'readonly'
        with fits.open(fitsfile, mode=mode) as f:
            for k,v in cutout.wcs.to_header().items():
                f[0].header[k] = v
            f[0].data = cutout.data
            if inplace:
                f.flush()
            else:
                outfile = fitsfile.replace('.fits','_crop.fits')
                print('Writing cropped image to {}'.format(outfile))
                f.writeto(outfile, overwrite=True)
    else:
        return cutout

if __name__ == '__main__':
    fitsfile = sys.argv[1]
    crop_nans(fitsfile)