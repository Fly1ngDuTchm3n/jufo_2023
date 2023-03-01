import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

CLUSTER_SIZE = 100


def pack_in_clusters(diff_list: list):
    diff_list += [0] * (100 - (len(diff_list) % 100))
    np_diff_list = np.array(diff_list, dtype=np.int16)
    np_diff_list = np_diff_list.reshape(round(len(np_diff_list) / 100), 100)

    print(np_diff_list.shape)
    return 0


def plot_fft_result(fft_input):
    # sampling rate
    sr = len(fft_input)
    # sampling interval
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)

    fft_res = fft(fft_input)
    N = len(fft_res)
    n = np.arange(N)
    T = N / sr
    freq = n / T

    greater_than = np.abs(fft_res) > N / 5
    print(sum(greater_than))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(fft_res), "b", markerfmt=" ", basefmt="-b")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0, np.floor(N / 2))
    plt.ylim(0, N)

    plt.subplot(122)
    plt.plot(t, ifft(fft_res), "r")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def fft_check_for_asp(fft_input, is_asp):
    fft_res = fft(fft_input)
    len_fft = len(fft_res)

    greater_than = np.abs(fft_res) > len_fft
    if not sum(greater_than) < len_fft/2 == is_asp:
        if is_asp:
            print('asp mistake')
        else:
            print('gra mistake')
        plot_fft_result(fft_input)
        return False
    else:
        return True
