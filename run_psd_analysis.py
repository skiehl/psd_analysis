#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import glob
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
dir_data = 'data/'           # directory where light curves are located
dir_results = 'analysis/'    # directory where analysis and results are stored

# file name:
file_sources = 'sources.dat' # file that lists the source names
file_suffix = '.csv'         # extension of the light curve files

#==============================================================================
# MAIN
#==============================================================================

sources = np.loadtxt(
        file_sources, dtype='str', delimiter=',', usecols=(0,), skiprows=1)
n_sources = len(sources)

# iterate through data files:
for i, source in enumerate(sources, start=1):
    source = source.strip()
    print 'Source {0:d} of {1:d}: {2:s}'.format(i, n_sources, source)
    db_file = os.path.join(dir_results, '{0:s}/psd/{0:s}.h5'.format(source))

    if os.path.isfile(db_file):
        print 'Done.\n'
        continue

    # load data:
    try:
        dtype = [('jd', float), ('flux', float), ('flux_err', float)]
        data_file = os.path.join(dir_data, '{0:s}{1:s}'.format(
                source, file_suffix))
        data = np.loadtxt(data_file, delimiter=',', dtype=dtype, skiprows=1)
    except Exception as e:
        print e
        print ''
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
        print '\nSpectral index: %.2f' % index
        psd.run_sim(db_file, iterations, index=index)

    # evaluate simulations:
    dir_out = os.path.join(dir_results, '{0:s}/psd/'.format(source))
    psd.evaluate_sim(db_file, output_dir=dir_out, smooth_confint=5)
    psd.test_reliability(db_file, output_dir=dir_out, smooth_confint=5)
    print ''
