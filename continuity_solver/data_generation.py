import numpy as np
from numpy.random import uniform, random
from tqdm.auto import trange
import matplotlib.pyplot as plt
from .helpers import Peak
import pickle


class Generator:
    @staticmethod
    def w_sampling(U_max, dw):
        Nw = int(U_max / dw)
        ws = np.linspace(-U_max, U_max, 2 * Nw + 1)
        return ws

    @staticmethod
    def spectral_density(ws, U_max, seed=None, shape='quadratic'):
        """ Returns an array of A[i] containing the values of A at ws[i] """
        if seed is not None:
            np.random.seed(seed)
        peaks = []
        # Interaction strength
        U = uniform(0, U_max)
        # Metallic or insulator
        metallic = random() < (U_max - U) / U_max
        if metallic:
            peaks.append(Peak(
                0,
                0.5,
                min(2, 4 / U**2),
                shape
            ))
        # Generating first peak
        peaks.append(Peak(
            U / 2,
            uniform(0.1, 0.4),
            uniform(0.5, 2),
            shape
        ))
        # And its symmetric
        peaks.append(Peak(
            -1 * U / 2,
            peaks[-1].height,
            peaks[-1].width,
            shape
        ))

        # Aggregate all the peaks
        return Peak.aggregate(peaks, ws), {
            'U': U,
            'peaks': peaks
        }

    @staticmethod
    def data(ws, U_max, N_samples, seed=42, shape='quadratic'):
        np.random.seed(seed)  # for reproductibility
        data = []
        for _ in trange(N_samples, desc='Generating input (A and G)'):
            A, params = Generator.spectral_density(ws, U_max, shape=shape)
            # print(np.sum(A) * dw)
            data.append({
                'params': params,
                'greens': Green.compute_greens(A, ws)
            })
        return data

    @staticmethod
    def display(datum, ws):
        values = Peak.aggregate(datum['params']['peaks'], ws)
        print(Peak.integral(values, ws))
        plt.figure()
        plt.plot(ws, values)
        plt.show()

    @staticmethod
    def store(data, filename='data.pkl'):
        with open(filename, 'wb+') as out:
            pickle.dump(data, out)


class Green:
    @staticmethod
    def compute_green(A, ws, n):
        """ Computes Integral(A(w) / (iw_n - w)) for a given n"""
        return complex(Peak.integral(A / (1j * (2 * n + 1) - ws), ws))

    @staticmethod
    def compute_greens(A, ws, nw_cutoff=300):
        """ Computes Integral(A(w) / (iw_n - w)) for all n below nw_cutoff"""
        ReG, ImG = [], []
        for n in range(nw_cutoff):
            z = Green.compute_green(A, ws, n)
            ReG.append(z.real)
            ImG.append(z.imag)
        return ReG + ImG


if __name__ == '__main__':
    # Full generation of data
    U_max = 5
    dw = 0.1
    N_samples = 100000
    ws = Generator.w_sampling(U_max, dw)
    data = Generator.data(ws, U_max, N_samples, shape='gaussian')
    Generator.store(data, filename='data.pkl')
    # Investigating one example
    datum_index = 5
    Generator.display(data[datum_index], ws)
    print(data[datum_index])
