import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from colour import (
    STANDARD_OBSERVER_CMFS, SpectralShape, SpectralDistribution,
    COLOURCHECKER_SDS, ILLUMINANT_SDS, ILLUMINANTS, sd_to_XYZ, XYZ_to_xy,
    XYZ_to_Lab)
from colour.difference import delta_E_CIE1976
from colour.utilities import as_float_array
from colour.plotting import plot_chromaticity_diagram_CIE1931

from otsu2018 import load_Otsu2018_spectra
from datasets.otsu2018 import *


# Copied from https://github.com/enneract/colour/tree/feature/otsu2018
def XYZ_to_sd_Otsu2018(
        XYZ,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(OTSU_2018_SPECTRAL_SHAPE),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            OTSU_2018_SPECTRAL_SHAPE),
        clip=True):
    XYZ = as_float_array(XYZ)
    xy = XYZ_to_xy(XYZ)
    cluster = select_cluster_Otsu2018(xy)

    basis_functions = OTSU_2018_BASIS_FUNCTIONS[cluster]
    mean = OTSU_2018_MEANS[cluster]

    M = np.empty((3, 3))
    for i in range(3):
        sd = SpectralDistribution(
            basis_functions[i, :],
            OTSU_2018_SPECTRAL_SHAPE.range(),
        )
        M[:, i] = sd_to_XYZ(sd, illuminant=illuminant) / 100
    M_inverse = np.linalg.inv(M)

    sd = SpectralDistribution(mean, OTSU_2018_SPECTRAL_SHAPE.range())
    XYZ_mu = sd_to_XYZ(sd, illuminant=illuminant) / 100

    weights = np.dot(M_inverse, XYZ - XYZ_mu)
    recovered_sd = np.dot(weights, basis_functions) + mean

    if clip:
        recovered_sd = np.clip(recovered_sd, 0, 1)

    return SpectralDistribution(recovered_sd, OTSU_2018_SPECTRAL_SHAPE.range())



if __name__ == '__main__':
    print('Loading spectral data...')
    data = load_Otsu2018_spectra('CommonData/spectrum_m.csv')
    shape = SpectralShape(380, 730, 10)
    sds = [SpectralDistribution(data[i, :], shape.range())
           for i in range(data.shape[0])]


    for name, colourchecker in COLOURCHECKER_SDS.items():
        print('Adding %s...' % name)
        sds += colourchecker.values()

    D65 = ILLUMINANT_SDS['D65']
    xy_w = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

    x = []
    y = []
    errors = []
    above_JND = 0
    for i, sd in tqdm.tqdm(enumerate(sds), total=len(sds)):
        XYZ = sd_to_XYZ(sd, illuminant=D65) / 100
        xy = XYZ_to_xy(XYZ)
        x.append(xy[0])
        y.append(xy[1])
        Lab = XYZ_to_Lab(XYZ, xy_w)

        recovered_sd = XYZ_to_sd_Otsu2018(XYZ)
        recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=D65) / 100
        recovered_Lab = XYZ_to_Lab(recovered_XYZ, xy_w)

        error = delta_E_CIE1976(Lab, recovered_Lab)
        errors.append(error)
        if error > 2.4:
            above_JND += 1

    print('Min. error: %g' % min(errors))
    print('Max. error: %g' % max(errors))
    print('Avg. error: %g' % np.mean(errors))
    print('Errors above JND: %d (%.1f%%)'
          % (above_JND, 100 * above_JND / len(sds)))

    bins = [int((max(y) - min(y)) / 0.01), int((max(x) - min(x)) / 0.01)]
    histogram, _, _ = np.histogram2d(x, y, bins=bins, weights=errors)

    plot_chromaticity_diagram_CIE1931(standalone=False)
    plt.imshow(histogram, extent=(min(x), max(x), min(y), max(y)),
               interpolation='bicubic')
    plt.colorbar()
    plt.show()
