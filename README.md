# Spectral Methods for Hyperbolic Problems

This code implements spectral filtering and Gibbs complimentary basis re-expansion, as outlined in the paper [_Spectral Methods for Hyperbolic Problems_](https://doi.org/10.1016/S0377-0427(00)00510-0) by Gottlieb and Hesthaven.

It is part of an end-semester project, which you can read more about [here](https://aadi.ink/spectral4hyp).

## Install
Download or clone this repository and then activate your environment 
before running
```bash
pip install -r requirements.txt
```
to install the required packages, in case they are not already present.

## Run
```bash
python main.py -h
```
will show the list of options.


| Flag | Explanation | 
| --- | --- | 
| `-N` | Number of grid points and Fourier modes (can be used multiple times) |
| `--pde` | Whether to solve the linear advection or Burgers' equation |
| `--ggb` | Plot the re-expansion of Fourier coefficients in the basis of Gegenbauer polynomials |
| '--Lambda' | The value of Lambda (sometimes referred to with alpha) for the Gegenbauer polynomials | 
| '--exact' | Path to the file containing the exact solution for the specified combination of PDE, initial condition, and final time. |

## Known issues
The Burgers' equation computes the right solution, but at a wrong speed.

