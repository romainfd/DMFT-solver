# DMFT solver

A LISA DMFT solver for the single-orbital Hubbard model at half-filling, based on Iterated Perturabation Theory. Reconstruction of the spectral function machine-learning-based (or based on Padé approximants).

Basically 100 lines on Python, just a modest introductory DMFT / Green function formalism project.

Shipped with detailed review / lecture notes on Green functions and DMFT : [Lecture notes](https://raw.githubusercontent.com/romainfd/DMFT-solver/main/notes/dmft.pdf)
[Papier-like report](https://raw.githubusercontent.com/romainfd/DMFT-solver/main/report/main.pdf)

## Resources
- [GKKR Review](https://www.physics.rutgers.edu/~gkguest/papers/rmp63_1996_p13_Kotliar.pdf)
- [Quantum toolbox in python/C++](https://triqs.github.io/triqs/latest/)
- Analytic continuation
   - [2019 paper](https://arxiv.org/abs/1806.03841) -> [citations](https://scholar.google.co.il/scholar?oi=bibs&hl=en&cites=18149676228975098363)
   - [2020 paper](https://actu.epfl.ch/news/le-machine-learning-pour-les-problemes-de-prolonge/) by [Romain Fournier](https://www.linkedin.com/in/romain-fournier-08466895)
- Some DMFT solvers :
   - [Solver possibilities](https://www.theorie.physik.uni-muenchen.de/activities/schools/archiv/asc_school_17/extramaterial/parcollet_slides_3.pdf): page 35 for Iterated Perturbation Theory
   - [cometscome's code](https://github.com/cometscome/DMFT_withJulia) (CTAUX QMC as impurity solver, Bethe lattice)
