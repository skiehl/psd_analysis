#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for estimating the power spectral density (PSD) of a blazar light
curve.

This script provides an easy to use wrapper. The actual analysis scripts are
implemented in powerspectraldensity.py.

Usage: Edit the configuration parameters under CONFIG below as you see fit.
Provide the the input data files for the PSD analysis in a directory. Provide
a list of source names (which must equal the file names without the file
extensions) in a text file. Then run this script, which will iterate through
all given light curve files and determine the best fit simple power-law
rednoise PSD slope.

Note: Under MAIN are two lines that read in the source list and the data files.
Those assume a certain formatting of the files. These lines may have to be
adjusted to run correctly.
"""

import numpy as np
import os

import powerspectraldensity as psd

#==============================================================================
# CONFIG
#==============================================================================

# PSD analysis parameters:
ntime = 10
sim_sampling = 0.1
split = False
resample_rate = 'median'
resample_n = 10
resample_split_n = 10
bins_per_order = 5
split_limit = 10
scaling = 'std'

iterations = 1000
spectral_indices = np.linspace(0., 5., 51)

# directories:
dir_data = 'example/data/'        # directory where light curves are located
dir_results = 'example/analysis/' # analysis and results directory

# file name:
file_sources = 'sources.dat' # file that lists the source names
file_suffix = '.csv'         # extension of the light curve files

skip_if_exists = False       # skip analysis if analysis directory exists

#==============================================================================
# MAIN
#==============================================================================

with open(file_sources, mode='r') as f:
    sources = [line.strip() for line in f.readlines()]
    n_sources = len(sources)

# iterate through data files:
for i, source in enumerate(sources, start=1):
    source = source.strip()
    print(f'Source {i:d} of {n_sources:d}: {source:s}')
    db_file = os.path.join(dir_results, f'{source:s}/psd/{source:s}.h5')

    if skip_if_exists and os.path.isfile(db_file):
        print('Done.\n')
        continue

    # load data:
    try:
        dtype = [('jd', float), ('flux', float), ('flux_err', float)]
        data_file = os.path.join(dir_data, '{0:s}{1:s}'.format(
                source, file_suffix))
        data = np.loadtxt(data_file, delimiter=',', dtype=dtype, skiprows=1)
    except Exception as e:
        print(e, end='\n\n')
        continue

    # create PSD simulation data base:
    psd.create_psd_db(
            db_file, data['jd'], data['flux'], data['flux_err'],
            ntime, sim_sampling, spec_shape='powerlaw',
            split=split, split_limit=split_limit,
            resample_rate=resample_rate, resample_n=resample_n,
            resample_split_n=resample_split_n,
            bins_per_order=bins_per_order, scaling=scaling)

    # run simulations:
    for index in spectral_indices:
        print(f'\nSpectral index: {index:.2f}')
        psd.run_sim(db_file, iterations, index=index)

    # evaluate simulations:
    dir_out = os.path.join(dir_results, f'{source:s}/psd/')
    psd.evaluate_sim(db_file, output_dir=dir_out, smooth_confint=5)
    psd.test_reliability(db_file, output_dir=dir_out, smooth_confint=5)
    print('')

#==============================================================================
