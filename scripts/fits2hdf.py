#!usr/bin/env/python

import pandas as pd
from astropy.table import Table
import sys
import glob

def fits2hdf(fitsfile):
    hdffile = fitsfile.split('.fits')[0] + '.hdf5'
    t = Table.read(fitsfile).to_pandas()
    t.to_hdf(hdffile, key='data', mode='w', format='table',
        complevel=0)
    print('Converted {} to hdf5'.format(fitsfile))

def hdf2fits(hdffile):
    # direct reverse of fits2hdf
    fitsfile = hdffile.split('.hdf5')[0] + '.fits'
    df = pd.read_hdf(hdffile, key='data')
    t = Table.from_pandas(df)
    t.write(fitsfile, overwrite=True)
    print('Converted {} to fits'.format(hdffile))

if __name__ == '__main__':
    fitslist = glob.glob('{}'.format(sys.argv[1]))
    for f in fitslist:
        if '.fits' in f:
            fits2hdf(f)
        elif '.hdf5' in f:
            hdf2fits(f)
