#! /usr/bin/env python

import aplpy
import glob
import os
import numpy as np
import pandas as pd
import shutil
import sys

from astropy.io import fits
from astropy.wcs import WCS
from drizzlepac import astrodrizzle

def calc_center_and_shape(fitslist):
    '''
    Calculates optimal central RA and Dec of drizzled output
    based on input image footprints, and number of pixels required
    at native resolution for drizzled image
    '''
    ra, dec = [], []
    xs, ys = [], []
    for fitsfile in fitslist: # parallelize eventually
        with fits.open(fitsfile) as f:
            w = WCS(f[1].header,fobj=f)
            foot = w.calc_footprint()
            y, x = f[1].shape
            if f[4].name == 'SCI': # ACS and UVIS
                foot2 = WCS(f[4].header,fobj=f).calc_footprint()
                y += f[4].shape[0]
                foot = np.vstack([foot, foot2])
        ra.append(foot[:,0].mean())
        dec.append(foot[:,1].mean())
        xs.append(x)
        ys.append(y)
    mean_ra, mean_dec = np.mean(ra), np.mean(dec)
    coords = w.all_world2pix(ra,dec,0)
    dx = coords[0].max() - coords[0].min()
    dy = coords[1].max() - coords[1].min()
    nx, ny = 1.1*(max(xs) + dx), 1.1*(max(ys) + dy)
    return mean_ra, mean_dec, nx, ny

def drizzle_singlefilt(df_driz, mean_ra, mean_dec, nx, ny,
    scale=0.0498, outfilebase='drizzle'):
    '''
    do the thing
    '''
    detector = df_driz.DETECTOR.unique()[0]
    filt = df_driz.FILTER.unique()[0]
    print('Drizzling images for {} {}'.format(detector,filt))

    wht1  = 'IVM'
    if detector == 'WFC':
        native_scale = 0.049
        wht2 = 'ERR'
        bits = '~128'
        final_kernel='lanczos3'
    elif detector == 'UVIS':
        native_scale = 0.04
        wht2 = 'EXP'
        bits = None
        final_kernel='lanczos3'
    elif detector == 'IR':
        native_scale = 0.13
        wht2 = 'EXP'
        bits = None
        final_kernel='square'
    else:
        print('Detector not recognized!')

    final_nx = np.ceil(nx * (native_scale / scale)).astype(int)
    final_ny = np.ceil(ny * (native_scale / scale)).astype(int)
    print('Final image size: {} x {} pix'.format(final_nx, final_ny))

    flist = (df_driz.index+'.fits').values.tolist()
    print('Input images:', flist)
    outfile1 = '_'.join([outfilebase, filt])
    print('Drizzle output base:', outfile1)

    for wht in [wht1, wht2]:
        teal.unlearn('astrodrizzle')
        if wht == 'IVM':
            driz_cr_snr = '2 1.5'
        else:
            driz_cr_snr = '4 3'
        astrodrizzle.AstroDrizzle(flist, output=outfile1 + '_{}'.format(wht), 
                                  combine_type='median', driz_cr_snr=driz_cr_snr, 
                                  final_pixfrac=1.0, final_scale=scale, final_bits=bits,
                                  final_wht_type=wht, final_kernel=final_kernel,
                                  final_outnx=final_nx, final_outny=final_ny,
                                  final_ra=mean_ra, final_dec=mean_dec,
                                  skysub=False, clean=True)
    return outfile1

def combine_outfiles(expfile, keep=True):
    '''
    why is there no single weighting scheme that doesn't suck
    '''
    if not os.path.isfile(expfile):
        expfile = expfile.replace('EXP','ERR')
    ivmfile = expfile.replace('EXP','IVM').replace('ERR','IVM')
    combfile = expfile.replace('EXP','comb').replace('ERR','comb')
    shutil.copyfile(expfile, combfile)
    d1 = fits.getdata(expfile,0)
    d2 = fits.getdata(ivmfile,0)
    fill_cond = (d1 < 0) & (d2 > d1)
    d1[fill_cond] = d2[fill_cond]
    with fits.open(combfile, mode='update') as f:
        f[0].data = d1
    if not keep:
        os.remove(ivmfile)
        os.remove(expfile)
    return combfile

def make_color_image(b, r, blue_filter, red_filter):
    print(b, r, blue_filter, red_filter)
    g = b.replace(blue_filter, 'GREEN')
    print('Making pseudo-green image {}'.format(g))
    shutil.copyfile(b, g)
    with fits.open(g, mode='update') as f:
        f[0].data = np.nanmean([fits.getdata(b), fits.getdata(r)], axis=0)
    outfile = g.split('GREEN')[0] + blue_filter + '_' + red_filter + '.png'
    aplpy.make_rgb_image([r,g,b], outfile, 
                         pmin_r=3, stretch_r='arcsinh', # pmax=99.75
                         pmin_g=3, stretch_g='arcsinh',
                         pmin_b=3, stretch_b='arcsinh')


def make_header_table(fitsdir, search_string='*fl?.fits'):
    """Construct a table of key-value pairs from FITS headers of images

    Inputs
    ------
    fitsdir : path 
        directory of FITS files
    search_string : string or regex patter, optional
        string to search for FITS images with. Default is
        '*fl?.fits'

    Returns
    -------
    df : DataFrame
        A table of header key-value pairs indexed by image name.
    """
    headers = {}
    fitslist = list(glob.glob(os.path.join(fitsdir, search_string)))
    if len(fitslist) == 0: 
        raise Exception('No fits files found in {}!'.format(fitsdir))
    # get headers from each image
    for fitsfile in fitslist:
        fitsname = fitsfile.split('/')[-1]
        head = dict(fits.getheader(fitsfile, 0, ignore_missing_end=True).items())
        try:
            photplam = fits.getval(fitsfile, 'PHOTPLAM', ext=0)
        except KeyError:
            photplam = fits.getval(fitsfile, 'PHOTPLAM', ext=1)
        head['PHOTPLAM'] = float(photplam)
        headers.update({fitsname:head})
    # construct dataframe
    df = pd.DataFrame(columns=['DETECTOR','FILTER','FILTER1','FILTER2','PHOTPLAM'])
    for fitsname, head in headers.items():
        row = pd.Series(dict(head.items()))
        df.loc[fitsname.split('.fits')[0]] = row.T
    lamfunc = lambda x: ''.join(x[~(x.str.startswith('CLEAR')|x.str.startswith('nan'))])
    filters = df.filter(regex='FILTER').astype(str).apply(lamfunc, axis=1)
    df.loc[:,'FILTER'] = filters
    df.drop(['FILTER1','FILTER2'], axis=1, inplace=True)
    df.sort_values(by='PHOTPLAM', inplace=True)
    return fitslist, df

if __name__ == '__main__':
    fitsdir = sys.argv[1]
    fitslist, df = make_header_table(fitsdir)
    mean_ra, mean_dec, nx, ny = calc_center_and_shape(fitslist)
    print('Drizzle center: {}, {}'.format(mean_ra, mean_dec))
    outfiles = []
    filters = []
    if not os.getcwd().endswith(fitsdir):
        os.chdir(fitsdir)
        print('Moving to {} to run drizzle'.format(fitsdir))
    for photplam in sorted(df.PHOTPLAM.unique()[:2]):
        df_to_driz = df[np.isclose(df.PHOTPLAM, [photplam])]
        filt = df_to_driz.FILTER.unique()[0]
        filters.append(filt)
        fname = drizzle_singlefilt(df_to_driz, mean_ra, mean_dec, nx, ny)
        expfile = glob.glob(fname + '_E??_dr?_sci.fits')[0]
        outfile = combine_outfiles(expfile)
        outfiles.append(outfile)
    print('Making color images')
    print('Blue filter: {}'.format(filters[0]))
    print('Red filter: {}'.format(filters[1]))
    make_color_image(*outfiles, *filters)
