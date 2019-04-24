[![Build Status](https://travis-ci.com/crstnbr/dqmc.svg?token=ZhpR15dDPdpyVFTzrPfp&branch=master)](https://travis-ci.com/crstnbr/dqmc)
[![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master)
<!-- [![codecov](https://codecov.io/gh/crstnbr/dqmc/branch/master/graph/badge.svg)](https://codecov.io/gh/crstnbr/dqmc) !-->
<!-- [![travis][travis-img]](https://travis-ci.org/crstnbr/dqmc) !-->

<!-- [travis-img]: https://img.shields.io/travis/crstnbr/dqmc/master.svg?label=linux !-->
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/dqmc/master.svg?label=codecov


**Determinant quantum Monte Carlo (DQMC)** code for simulating quantum critical metals in two dimensions.

### Settings

**Environmental variables:**

* `LATTICES`: folder with ALPS XML lattice files (mandatory unless you're me)
* `WALLTIME`: Set a walltime limit for the algorithm. (optional)
* `JULIA_DQMC`: path to the root of this repo (optional, currently only used in `live.jl/ipynb` and `test_live.ipynb` to activate the environment)

### Modes
Special modes as indicated by fields in `dqmc.in.xml`:

* `EDRUN`: if set to true, temporal gradients and quartic term in bosonic action are turned off.
* `TIMING`: stop after thermalization phase and report speed and allocations of all major functions.
