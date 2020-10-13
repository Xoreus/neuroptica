# Neuroptica: Towards a Practical Implementation of Photonic Neural Networks [![Documentation Status](https://readthedocs.org/projects/neuroptica/badge/?version=latest)](https://neuroptica.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.com/fancompute/neuroptica.svg?token=CSoUuvqmixfJpdwkLqet&branch=master)](https://travis-ci.com/fancompute/neuroptica)

`Neuroptica` is a flexible chip-level simulation platform for [nanophotonic neural networks](https://arxiv.org/abs/1903.04579) written in Python/NumPy. It provides a wide range of abstracton levels for simulating optical NN's: the lowest-level functionality allows you to manipulate the arrangement and properties of individual phase shifters on a simulated chip, and the highest-level features provide a Keras-like API for designing optical NN by stacking network layers.

`Neuroptica: Towards a Practical Implementation of Photonic Neural Networks` was used to create and study Optical Neural Networks for [The diamond mesh, a phase-error- and loss-tolerant field-programmable MZI-based optical processor for optical neural networks](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-28-16-23495) and many conference papers. The code itself was cloned from the original `Neuroptica` repository and subsequently modified in order to create the different meshes and figures seen in the papers. 


```
You can clone this modified Neuroptica repository:
```
git clone https://gitlab.com/simongg/neuroptica.git
```

and in your program or notebook, add

```python
import sys
sys.path.append('path/to/neuroptica')
``` 

`neuroptica` requires Python >=3.6.


## Getting started

For an overview of `neuroptica`, read the [documentation](https://neuroptica.readthedocs.io). Example notebooks of the original code are included in the [`neuroptica-notebooks`](https://github.com/fancompute/neuroptica-notebooks) repository:

- [Planar data classification using electro-optic activation functions](https://github.com/fancompute/neuroptica-notebooks/blob/master/neuroptica_demo.ipynb)

![Phi Theta error Simulation of a trained 96x96 Diamond Mesh](img/DiamondMesh.png  "Diamond mesh accuracy plot")
![Insertion Loss + Phase error Simulation of a trained 96x96 Reck Mesh](img/LPU_ACC_R_P_N=96.pdf   "Diamond mesh accuracy plot")
![Phi Theta error Simulation of a trained 96x96 Clements Mesh](img/PT_ACC_E_P_N=96.pdf   "Reck mesh accuracy plot")
![Insertion Loss + Phase error Simulation of a trained 96x96 Reck Mesh](img/LPU_ACC_R_P_N=96.pdf  "Reck mesh accuracy plot")


## Authors
`Neuroptica: Towards a Practical Implementation of Photonic Neural Networks` was written by [Simon Geoffroy-Gagnon](https://s-g-gagnon.research.mcgill.ca/), with help from Farhad Shorkaneh.

The original `neuroptica` was written by [Ben Bartlett](https://github.com/bencbartlett), [Momchil Minkov](https://github.com/momchilmm), [Tyler Hughes](https://github.com/twhughes), and  [Ian Williamson](https://github.com/ianwilliamson).
