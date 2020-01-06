[![Build Status](https://travis-ci.com/crstnbr/dqmc.svg?token=ZhpR15dDPdpyVFTzrPfp&branch=master)](https://travis-ci.com/crstnbr/dqmc)
[![codecov](https://codecov.io/gh/crstnbr/dqmc/branch/master/graph/badge.svg?token=jTD6HWrHVh)](https://codecov.io/gh/crstnbr/dqmc)
<!-- [![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master) !-->
<!-- [![travis][travis-img]](https://travis-ci.org/crstnbr/dqmc) !-->

<!-- [travis-img]: https://img.shields.io/travis/crstnbr/dqmc/master.svg?label=linux !-->
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/dqmc/master.svg?label=codecov

| **Documentation**                                                               | **Build Status**                                                                                |  **License**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-dev-img]][docs-dev-url] | ![][lifecycle-img] [![][github-ci-img]][github-ci-url] [![](https://codecov.io/gh/crstnbr/dqmc/branch/master/graph/badge.svg?token=jTD6HWrHVh)][codecov-url] | [![][license-img]][license-url] |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://crstnbr.github.io/dqmc/dev
[github-ci-img]: https://github.com/crstnbr/dqmc/workflows/Run%20tests/badge.svg
[github-ci-url]: https://github.com/crstnbr/dqmc/actions?query=workflow%3A%22Run+tests%22
[codecov-img]: https://codecov.io/gh/crstnbr/dqmc/branch/master/graph/badge.svg?token=jTD6HWrHVh
[codecov-url]: https://codecov.io/gh/crstnbr/dqmc

[slack-url]: https://slackinvite.julialang.org/
[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[license-img]: https://img.shields.io/badge/License-MIT-red.svg
[license-url]: https://opensource.org/licenses/MIT

[lifecycle-img]: https://img.shields.io/badge/lifecycle-stable-blue.svg


**Determinant quantum Monte Carlo (DQMC)** code for simulating a quantum critical metal, a Fermi sea coupled to antiferromagnetic bosonic fluctuations, in two spatial dimensions.

This code has been used to produce (most of) the results in the following paper:

> [*Hierarchy of energy scales in an O(3) symmetric antiferromagnetic quantum critical metal: a Monte Carlo study*<br>Carsten Bauer, Yoni Schattner, Simon Trebst, Erez Berg](https://arxiv.org/abs/2001.00586)



### Settings

**Environmental variables:**

* `LATTICES`: folder with ALPS XML lattice files (mandatory unless you're me)
* `WALLTIME`: Set a walltime limit for the algorithm. (optional)
* `JULIA_DQMC`: path to the root of this repo (optional, currently only used in `live.jl/ipynb` and `test_live.ipynb` to activate the environment)

### Modes
Special modes as indicated by fields in `dqmc.in.xml`:

* `EDRUN`: if set to true, temporal gradients and quartic term in bosonic action are turned off.
* `TIMING`: stop after thermalization phase and report speed and allocations of all major functions.
