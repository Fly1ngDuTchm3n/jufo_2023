import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


def plot_fft_result(fft_res, len_orig_input):
    # sampling interval
    ts = 1.0 / len_orig_input
    t = np.arange(0, 1, ts)

    fft_len = len(fft_res)
    n = np.arange(fft_len)
    T = fft_len / len_orig_input
    freq = n / T

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(fft_res), "b", markerfmt=" ", basefmt="-b")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0, np.floor(fft_len / 2))

    plt.subplot(122)
    plt.plot(t, ifft(fft_res), "r")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def hand_picked_fft_algorithm(fft_mat, isAsp):
    wrong_counter = 0
    for arr in fft_mat:
        if fft_check_for_asp(arr) != isAsp:
            wrong_counter += 1
    return wrong_counter


def fft_check_for_asp(fft_res):
    len_fft = len(fft_res)

    greater_than = np.abs(fft_res[int(len_fft / 3) : int(len_fft / 2) :]) > len_fft / 2
    return sum(greater_than) < len_fft / 25


def replace_with_fft(input_mat):
    res_mat = np.empty_like(input_mat, dtype=np.float16)
    for i, arr in enumerate(input_mat):
        res_mat[i] = np.abs(fft(arr))
    return res_mat
