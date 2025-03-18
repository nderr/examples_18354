### 18.354 Example Code

This repo contains example scripts illustrating topics for the spring 2025 edition of 18.354: Nonlinear Dynamics II.

#### Setup
I suggest using [mamba](https://github.com/mamba-org/mamba) for package management, but the code will work as long as you have all of the packages listed in `environment.yml` installed.

If you are using mamba, you can just enter the directory and run `mamba env create --name examples_18354 --file environment.yml` to install the requirements into an environment named `examples_18354`. Afterwards, you can run `conda activate examples_18354` to enter the environment.

#### Instability and pattern formation
Run any of the scripts using `python <script_name>.py`.

- `lotka_volterra.py` simulates the [prey/predator](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) equations, $\dot{u} = au-buv, \ \dot{v} = -cv + euv$
and plots the results vs. time and then in phase space. You'll notice the phase space plot very slightly spirals inward, which is an example of numerical error.

- `swift_hohenberg.py` simulates the Swift-Hohenberg equation, $\dot{u} = ru - (1+\nabla^2)^2 u + g u^2 - u^3.$ The formulation is slightly different than in the lecture notes. See [this example from Chebfun](https://www.chebfun.org/examples/pde/SwiftHohenberg.html) for some example patterns you can try to recreate. This implementation uses finite difference methods rather than the spectral approach of `chebfun`, but the idea is the same: we numerically integrate the non-linear/reaction part of the equation using a [high-order explicit solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) and use the semi-implicit [Crank-Nicolson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) for the spatial portion.

- `turing_instability.py` will open an interactive window so that you can look at the dispersion relation $\sigma(k)$ of perturbations to the fixed point of $\dot{u} = u(uv-1), \ \dot{v} = s(1-u^2 v)$
for different values of $d$ and $s$. For any dispersion relation shown, you can press a button to simulate the corresponding non-linear system and see if stable patterns emerge or not. Notice that finite-wavelength instabilities occur for $d < 1$, $s > 1$ and unbounded homogeneous growth occurs for $s < 1$. The dispersion relation is calculated symbolically using `sympy`, so this also serves as an example of free and open source symbolic computation e.g. as an alternvative to Wolfram Mathematica (although typically `sympy`'s capabilities are a subset of Mathetmatica's.)