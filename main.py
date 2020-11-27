from typing import Callable
from scipy import integrate
from numpy import exp, pi, sqrt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


BETA = 1     # Temperature
MU = 1       # ???
K_MAX = 100  # Spatial frequency cutoff
z = 10000    # Coordination (should be infinite)
t = sqrt(z)  # Nearest-neighbor hopping: t_ij = t / sqrt(z) in Bethe lattice case


def freq(n: int) -> float:
    """
    :param n: Frequency index
    :return: The corresponding frequency
    """
    return (2 * n + 1) * pi / BETA


def fourier_transform(f: Callable[[float], complex]) -> Callable[[float], complex]:
    return f


def g0_init(n: int) -> float:
    """
    Non-interacting Green function of impurity model
    :param n: index of the frequency
    :return: G0(i w_n)
    """
    return n / 3


def impurity_solver(U: float, g0: Callable[[int], float], method='IPTA') -> Callable[[int], float]:
    """
    Solves the interacting impurity problem we mapped our lattice problem to.
    Using the provided method (default is IPTA : Iterated Perturbation Theory Approximation -> Eq 157).
    :param U: Impurity model interaction term
    :param g0: Impurity model non-interacting Green function
    :return: S(n) returning the self-energy for i w_n
    """
    g0_ft = fourier_transform(g0)
    y = lambda n: lambda t: exp(1j * freq(n) * t) * g0_ft(t)**3
    return lambda n: U / 2 + U**2 * integrate.quad(y(n), 0, BETA)[0]


def e(k: int) -> float:
    return k


def green_fct(k: int, sigma_imp: Callable[[int], float]) -> Callable[[int], complex]:
    """
    :return: The lattice Green function approximation (Eq 12)
    """
    return lambda n: 1 / (1j * freq(n) + MU - e(k) - sigma_imp(n))


def green_fct_loc(sigma_imp: Callable[[int], float]) -> Callable[[int], float]:
    """
    Computes G_loc based on the resulting self-energy for our impurity model sigma_imp.
    Sums all the local k parts: G(w) = sum_k G(k, w).
    :return: G_loc based on sigma_imp
    """
    return lambda n: sum([green_fct(k, sigma_imp)(n) for k in range(K_MAX)])


def self_consistency(sigma_imp: Callable[[int], float]) -> Callable[[int], float]:
    """
    Computes the new G0 using Dyson equation and G_loc
    :return: The updated G0
    """
    return lambda n: 1 / (sigma_imp(n) + 1 / green_fct_loc(sigma_imp)(n))
    # ToDo: This simplifies to 1j * freq(n) + MU - t**2 * G(n) for the Bethe lattice (Eq 23)


def diff(f1: Callable[[int], float], f2: Callable[[int], float]) -> float:
    """
    :return: The difference between the 2 input functions
    """
    _diff = 0
    for n in range(100):
        _diff += abs(f1(n) - f2(1))
    return _diff


def rebuild_g_t(g_aux_t: Callable[[float], complex]) -> Callable[[float], complex]:
    """
    Rebuilds the full G(T) from the G_aux(T) by imposing a 1 jump in 0 and anti-periodicity of period BETA
    """
    return lambda t: g_aux_t(t) - 1 / (2 * BETA)


if __name__ == '__main__':
    diffs = []
    g0, U = g0_init, 1  # we set our initial guess (could be metallic or insulator)
    for i in tqdm(range(K_MAX)):
        sigma_imp = impurity_solver(U, g0)
        new_g0 = self_consistency(sigma_imp)
        diffs.append(diff(g0, new_g0))
        g0 = new_g0

    # Basic plots
    plt.figure()
    plt.plot(diffs)
    plt.title("G0 update over iterations")
    plt.show()
    plt.figure()
    plt.plot([green_fct_loc(sigma_imp)(n) for n in trange(100)])
    plt.title("Green function G(i w_n)")
    plt.show()

    # go from G(iw) to G(T)
    g_loc = green_fct_loc(sigma_imp)
    g_aux_w = lambda n: g_loc(n) - 1 / (1j * freq(n))
    g_aux_t = fourier_transform(g_aux_w)
    g_t = rebuild_g_t(g_aux_t)

    # ML test: should get back to Bethe lattice's semi-circular density of states
    # density = predict(g_t)
