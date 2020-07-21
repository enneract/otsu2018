import os
import matplotlib.pyplot as plt
from colour import (
    SpectralShape, COLOURCHECKER_SDS, ILLUMINANT_SDS, sd_to_XYZ)

from otsu2018 import load_Otsu2018_spectra, Clustering


if __name__ == '__main__':
    print('Loading spectral data...')
    sds = load_Otsu2018_spectra('CommonData/spectrum_m.csv', every_nth=50)
    shape = SpectralShape(380, 730, 10)

    print('Initializing the clustering...')
    clustering = Clustering(sds, shape)

    print('Clustering...')
    before = clustering.root.total_reconstruction_error()
    clustering.optimise(2)
    after = clustering.root.total_reconstruction_error()

    print('Error before: %g' % before)
    print('Error after:  %g' % after)

    print('Saving the dataset...')
    os.makedirs('datasets', exist_ok=True)
    clustering.write_python_dataset('datasets/otsu2018.py')

    print('Plotting...')
    clustering.root.visualise()

    plt.figure()

    examples = COLOURCHECKER_SDS['ColorChecker N Ohta'].items()
    for i, (name, sd) in enumerate(examples):
        plt.subplot(2, 3, 1 + i)
        plt.title(name)

        plt.plot(sd.wavelengths, sd.values, label='Original')

        XYZ = sd_to_XYZ(sd, illuminant=ILLUMINANT_SDS['D65']) / 100
        recovered_sd = clustering.reconstruct(XYZ)
        plt.plot(recovered_sd.wavelengths, recovered_sd.values,
                 label='Recovered')

        plt.legend()

        if i + 1 == 6:
            break

    plt.show()
