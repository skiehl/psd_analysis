# psd_analysis
Scripts to determine the power spectral density (PSD) of blazar light curves.

## Requirements

This script uses the following standard python packages:
+ datetime
+ math
+ os
+ sys

This script uses the following python packages:
+ numpy
+ scipy
+ statsmodels
+ matplotlib
+ pytables

## Getting Started

Get the python script:

    $ git clone https://github.com/skiehl/psd_analysis.git

## Usage

`powerspectraldensity.py` contains the relevant scripts. `run_psd_analysis.py`
provides a useful "wrapper" script for easily running the PSD analysis for
many light curves. In the CONFIG part of the script all parameters for the
analsis can be set. In the MAIN part the "load data" lines may be edited if
the light curve files follow a different file structure than assumed here.

## Example

An example of a light curve, the analysis, and the resulting plots and data
files is given in the directory `example/`.

## Notes

+ This PSD analysis code is an extension of the algorithm introduced by [1].
  The detailed explanation is given in [2].
+ Different PSD models are implemented in the script (power-law,
  broken power-law, knee-model, explained in [1]). However, the optimization is
  only implemented for the power-law model, which has a single free parameter.
  Any other models do not work with the current code.
+ In principle this code allows us to use the light curve simulation algorithm
  by [4]. Everything necessary is implemented and could be chosen by setting
  `scaling = 'pdf'` in the CONFIG part of `run_psd_analysis.py`. However, in
  [1] Sec. 4.2.5 I showed that the simulation algorithm by [4] is not feasible
  for estimating the PSD, because it will lead to strong biases. While this
  implementation still allows its usage, the user should be strongly
  discouraged from doing so.
+ Though first uploaded to GitHub in 2023, the script was originally written in
  2015 for python 2.7, with some modifications made over the years. This is now
  a port to python 3 with updated string formatting, updated docstrings, and
  with some new formatting of the script files. My programming style has
  strongly changed (evolved, I would claim) over these years. This code now may
  be inconsistently formatted. Generally, I find it not well structured. And as
  explained in the points above, the code includes functions that cannot or
  should not be used. I would favor a complete overhaul of this code, but
  currently I do not have the time to do it. Maybe it is useful to someone
  in its current form nethertheless.

## References

[1] [Uttley et al., 2002](https://ui.adsabs.harvard.edu/abs/2002MNRAS.332..231U/abstract)
[2] [Kiehlmann, 2015](https://kups.ub.uni-koeln.de/6231/)
[2] [Timmer&Koenig, 1995](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..707T/abstract).
[3] [Emmanoulopoulos et al., 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433..907E/abstract).

## License

psd_analysis is licensed under the BSD 3-Clause License - see the
[LICENSE](https://github.com/skiehl/psd_analysis/blob/main/LICENSE) file.

## Alternatives

At least one other python implementation of this method is available on GitHub:

+ [PSRESP](https://github.com/joy-228/PSRESP)
