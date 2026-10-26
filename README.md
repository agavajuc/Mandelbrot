# Escape-time fractal engine

Project focused on developing an escape-time fractal renderer, with the purpose of making it fractal arbitrary. The project was inspired from [NeuralNine's YouTube video](https://youtu.be/xjjmkg9J7Gg?si=fouHKw4ElbWYwFYn) and evolved into a Numba parallelized engine for visualizing different fractals and interacting with them. The desired fractal can be rendered by describing it in a small TOML file, as explained below.

```bash
pip install numba matplotlib pytest

python fractal.py render  fractals/mandelbrot.toml --size 3000x3000

python fractal.py explore fractals/julia.toml

python fractal.py code    fractals/burning_ship.toml

python -m pytest tests -q
```

## Project Strucutre

1. `fractals/mandelbrot.toml` and `fractals/julia.toml`: same formula, different `[setup]`.
   `p` is the pixel; whatever you assign it to decides which "plane" you are looking at.
2. `engine/spec.py`: loads a TOML file into a `FractalSpec` dataclass. Pure data.
3. `engine/compiler.py`: the core, in five numbered sections:
   check (whitelist) → optimise (`z**2` → `pow_2(z)`) → generate source → compile → run.
   Run `python fractal.py code <spec>` next to it to see what it produces.
4. `engine/colouring.py`: (n, z) arrays → RGB. Independent of the formula.
5. `fractal.py` (CLI) and `engine/viewer.py` (interactive window): thin layers on top.
6. `tests/test_engine.py`: the engine against plain Python, periodicity on vs off, unsafe formulas.

## Spec format

```toml
name = "Mandelbrot"
step = "z**2 + c"            # the recurrence; may use z, p and anything from [setup]

[setup]                      # evaluated once per pixel, in order; must define z
z = "0"
c = "p"

[stop]
mode = "diverge"             # or "converge" (Newton-type: stop when z stops moving)
radius = 256                 # bailout radius
degree = 2                   # for smooth colouring; 0 = integer counts
periodicity = true           # Brent periodicity checking
# tolerance = 1e-9           # converge mode only

[view]
center = [-0.5, 0.0]
width = 3.0
max_iter = 500

[colour]
cmap = "twilight_shifted"    # any matplotlib colormap
```

Functions available in formulas: `re im conj abs exp log sqrt sin cos sinh cosh`.

## Known limitations, next steps

- Each new formula compiles in ~2 s (Numba can't disk-cache `exec`-generated code).
- Burning Ship appears upside-down compared to most pictures online, which flip the imaginary axis.
- Smooth coloring needs `degree`; formulas like `exp(z) + c` should use `degree = 0`.
<!-- - Ideas: parameters as kernel arguments (live sliders, Mandelbrot→Julia link), resumable iterations,
  Mariani–Silver subdivision, a GPU backend generating Metal/CUDA source from the same syntax tree. -->

## Use of AI

This project has made use of generative AI for two purposes: 
- Building the well-structured TOML spec architecture, for significant improvement of user experience, and
- Testing the source code.