import numpy as np
import matplotlib.pyplot as plt

from otsu2018 import load_Otsu2018_spectra, Clustering


if __name__ == '__main__':
    sds = load_Otsu2018_spectra('CommonData/spectrum_m.csv', every_nth=1)
    wl = np.arange(380, 731, 10)

    mean = np.mean(sds, axis=0)
    data_matrix = sds - mean
    covariance_matrix = np.dot(data_matrix.T, data_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    for i, v in enumerate(eigenvalues):
        w = eigenvectors[:, i]
        color = 'C%d' % (i % 10)

        scale = 0.5 / np.max(np.abs(w))
        baseline = i + 1

        plt.plot((min(wl), max(wl)), (baseline, baseline), color + ':')
        plt.plot(wl, baseline + scale * w, color + '-')

        plt.annotate('λ = %g' % v, color=color, xy=(max(wl), baseline),
                     xytext=(max(wl), baseline + 0.05), ha='right')

    plt.title('PCA decomposition')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Eigenvector number')
    plt.xlim(370, 740)
    plt.ylim(30.5, 36.5)
    plt.show()
