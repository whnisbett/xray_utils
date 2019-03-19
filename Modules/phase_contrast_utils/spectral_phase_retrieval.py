import numpy as np
import stat_utils as su


# import spectral_utils to get the spectral_datum class


def get_coeff_mat(k_2_1d, energies, magnification, r2):
    coeff_mat = np.zeros((len(k_2_1d), len(energies), 2))

    C = 299792458  # speed of light in [m/s]
    H = 6.62607004e-34
    R_E = 2.8179403227e-15
    Gamma_term = (magnification * R_E * r2 * (C * H)**2) / (2.0 * np.pi)

    for i in range(len(k_2_1d)):
        for j in range(len(energies)):
            coeff_mat[i, j, 0] = (energies[j] * 1.6021766208e-16)**(-3) * 1.0E-79
            coeff_mat[i, j, 1] = klein_nishina(energies[j]) + Gamma_term * k_2_1d[i] / (energies[j] *
                                                                                        1.6021766208e-16)**2
    return coeff_mat


def main(experiment):
    r2 = experiment.r2
    magnification = experiment.magnification
    data = experiment.data  # this is spectral data
    imgs = data.get_ff_corr_img_list()
    thresholds = data.get_threshold_list()  # do not correspond to energies yet
    energies = (experiment.kvp + np.array(thresholds)) / 2
    k_2 = experiment.k_2
    r, c = k_2.shape
    k_2_1d = np.reshape(k_2, r * c)

    coeff_mat = get_coeff_mat(k_2_1d, energies, magnification, r2)

    scaled_imgs = -1. * np.log(np.array(imgs))
    ffts = np.zeros((len(k_2_1d), len(energies)), dtype = np.complex_)

    for i in range(len(energies)):
        fft = np.fft.fft2(scaled_imgs[i])
        fft = np.fft.fftshift(fft)
        ffts[:, i] = np.reshape(fft, r * c)

    # how do you get the energies from the thresholds though? Should be the mean energy of the bin

    # get a list of energies for each datum object

    w_mat = np.linspace(200, 1, num = len(imgs))
    w_mat = np.diag(w_mat)
    w_mat = w_mat / np.sum(w_mat)

    solution = np.zeros((2, len(k_2_1d)), dtype = np.complex_)

    for i in range(len(k_2_1d)):
        matBeta = coeff_mat[i, :, :]

        matGamma = ffts[i, :]  # , 0]

        matBetaTW = np.matmul(np.transpose(matBeta), w_mat)

        B = np.matmul(matBetaTW, matBeta)

        C = np.matmul(matBetaTW, matGamma)

        x = np.linalg.lstsq(B, C,rcond = None)[0]

        solution[:, i] = x
    map1 = np.fft.ifft2(
        np.fft.ifftshift(np.reshape(solution[0, :], (r, c))))  # should this be an fftshift or an ifftshift??
    map2 = np.fft.ifft2(np.fft.ifftshift(np.reshape(solution[1, :], (r, c))))

    return map1, map2


def klein_nishina(energy):
    alp = energy / 511.0
    R_E = 2.8179403227e-15
    term1 = (1 + alp) / alp**2
    term2 = 2 * (1 + alp) / (1 + 2 * alp) - np.log(1 + 2 * alp) / alp
    term3 = np.log(1 + 2 * alp) / (2 * alp) - (1 + 3 * alp) / (1 + 2 * alp)**2

    kn = (2 * np.pi * R_E**2) * (term1 * term2 + term3)
    return kn
