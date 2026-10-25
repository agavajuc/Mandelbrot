"""
Turns a FractalSpec into a fast parallel Numba kernel.

The pipeline is a tiny compiler:

    formula text --parse--> syntax tree --check--> safe tree --optimise--> tree --unparse--> text
                                                                                              |
    Python source of a whole kernel  <--- pasted into a fixed loop TEMPLATE  <----------------+
    (plus small generated helper functions such as pow_2)
                |
                +--exec--> Python function --numba.njit(parallel=True)--> machine code

Run `python fractal.py code fractals/mandelbrot.toml` to see the generated source.
"""
import ast
import cmath
import math
import textwrap

import numpy as np
from numba import njit, prange

from .spec import FractalSpec


class FormulaError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# 1. Functions a formula may call. Each is a tiny jitted helper, so they work on both real
#    and complex arguments (adding 0j forces complex, which cmath requires).
# --------------------------------------------------------------------------------------
@njit(inline="always")
def re(x):
    return (x + 0j).real

@njit(inline="always")
def im(x):
    return (x + 0j).imag

@njit(inline="always")
def conj(x):
    return (x + 0j).conjugate()

@njit(inline="always")
def exp(x):
    return cmath.exp(x + 0j)

@njit(inline="always")
def log(x):
    return cmath.log(x + 0j)

@njit(inline="always")
def sqrt(x):
    return cmath.sqrt(x + 0j)

@njit(inline="always")
def sin(x):
    return cmath.sin(x + 0j)

@njit(inline="always")
def cos(x):
    return cmath.cos(x + 0j)

@njit(inline="always")
def sinh(x):
    return cmath.sinh(x + 0j)

@njit(inline="always")
def cosh(x):
    return cmath.cosh(x + 0j)

FUNCTIONS = {"re": re, "im": im, "conj": conj, "abs": abs, "exp": exp, "log": log,
             "sqrt": sqrt, "sin": sin, "cos": cos, "sinh": sinh, "cosh": cosh}

# Names used by the kernel template itself; a spec may not define variables with these names.
RESERVED = {"p", "i", "j", "k", "n", "r2", "d", "z_new", "saved", "check_at",
            "cx", "cy", "step", "width", "height", "max_iter", "out_n", "out_z", "math",
            "prange", "njit", "kernel", "use_periodicity"} | set(FUNCTIONS)


# --------------------------------------------------------------------------------------
# 2. Checking: walk the syntax tree and allow ONLY arithmetic on numbers and known names.
#    This is a whitelist, which is what makes it safe to exec the generated code later:
#    anything like  __import__('os')  or  z.__class__  is rejected here.
# --------------------------------------------------------------------------------------
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
                 ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


def check_expression(text, variables):
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"cannot parse {text!r}: {e.msg}") from None

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise FormulaError(f"{text!r}: '{type(node).__name__}' is not allowed in a formula")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float, complex)):
            raise FormulaError(f"{text!r}: only numeric constants are allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS) or node.keywords:
                raise FormulaError(f"{text!r}: unknown function call (available: {sorted(FUNCTIONS)})")
            if len(node.args) != 1:
                raise FormulaError(f"{text!r}: {node.func.id}() takes exactly one argument")
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in FUNCTIONS:
            raise FormulaError(f"{text!r}: unknown name '{node.id}' (known here: {sorted(variables)})")
    return tree


# --------------------------------------------------------------------------------------
# 3. Optimising: rewrite  x ** <small positive integer>  into  pow_n(x).
#    Complex `**` in general goes through exp(n * log(x)): slow and less accurate.
#    An ast.NodeTransformer visits every node of the tree and may return a replacement.
# --------------------------------------------------------------------------------------
class IntegerPowers(ast.NodeTransformer):
    def __init__(self):
        self.used = set()                      # which pow_n functions we need to generate

    def visit_BinOp(self, node):
        self.generic_visit(node)               # optimise the children first
        exponent = node.right
        if (isinstance(node.op, ast.Pow) and isinstance(exponent, ast.Constant)
                and type(exponent.value) is int and 1 <= exponent.value <= 64):
            self.used.add(exponent.value)
            return ast.Call(func=ast.Name(f"pow_{exponent.value}", ast.Load()), args=[node.left], keywords=[])
        return node


def power_function_source(n):
    """Source of an unrolled x**n by repeated squaring, e.g. n = 5 (binary 101):
         r = x;  x = x*x;  x = x*x;  r = r*x      -> 3 multiplications instead of 4.
    Unrolled at generation time, because Numba does not unroll a loop over n by itself."""
    lines = ['@njit(inline="always")', f"def pow_{n}(x):"]
    have_result = False
    while True:
        if n & 1:
            lines.append("    r = r * x" if have_result else "    r = x")
            have_result = True
        n >>= 1
        if n == 0:
            break
        lines.append("    x = x * x")
    lines.append("    return r")
    return "\n".join(lines)


def to_code(text, variables, powers):
    """Formula text -> checked, optimised Python expression text (records the powers it needs)."""
    optimiser = IntegerPowers()
    tree = optimiser.visit(check_expression(text, variables))
    powers |= optimiser.used
    return ast.unparse(ast.fix_missing_locations(tree))


# --------------------------------------------------------------------------------------
# 4. Code generation: the parts that are the same for every fractal live in these templates.
#    Only {setup}, {step} and a few constants change.
# --------------------------------------------------------------------------------------
KERNEL_TEMPLATE = """\
def kernel(cx, cy, step, width, height, max_iter, use_periodicity, out_n, out_z):
    # rows are spread over the CPU threads (like '#pragma omp parallel for')
    for i in prange(height):
        for j in range(width):
            # p = this pixel's point in the complex plane
            p = complex(cx + (j - (width - 1) / 2) * step,
                        cy + (i - (height - 1) / 2) * step)
            # setup (from the spec); '+ 0j' makes every variable complex, so Numba sees one type
{setup}
            # n = -1 means "never stopped" (inside the set / did not converge)
            n = -1.0
{loop}
            out_n[i, j] = n
            out_z[i, j] = z
"""

# Divergence: stop when |z| > radius. Includes Brent's periodicity check.
DIVERGE_LOOP = """\
saved = z
check_at = 1
for k in range(max_iter):
    z = {step}
    r2 = z.real * z.real + z.imag * z.imag
    # escaped? (written as 'not <=' so that NaN also counts as escaped)
    if not (r2 <= {bail2!r}):
        n = {escape_value}
        break
    if use_periodicity:
        # back at the saved point -> the orbit is periodic and will never escape
        d = z - saved
        if d.real * d.real + d.imag * d.imag < {tol2!r}:
            break
        # Brent: move the saved point forward at k = 1, 2, 4, 8, ...
        if k == check_at:
            saved = z
            check_at *= 2
"""

# Convergence (Newton-type fractals): stop when z stops moving.
CONVERGE_LOOP = """\
for k in range(max_iter):
    z_new = {step}
    d = z_new - z
    z = z_new
    if d.real * d.real + d.imag * d.imag < {tol2!r}:
        n = float(k)
        break
    # blew up (e.g. division by ~0): give up on this pixel
    if not (z.real * z.real + z.imag * z.imag < 1e300):
        break
"""


def generate_source(spec: FractalSpec) -> str:
    if "z" not in spec.setup:
        raise FormulaError("[setup] must define z (the starting value of the orbit)")

    # Setup lines, in file order: each may use p and the names defined before it.
    known = {"p"}
    powers = set()
    setup_lines = []
    for name, text in spec.setup.items():
        if not name.isidentifier() or name in RESERVED:
            raise FormulaError(f"'{name}' cannot be used as a variable name")
        setup_lines.append(f"{name} = ({to_code(text, known, powers)}) + 0j")
        known.add(name)

    step_code = to_code(spec.step, known, powers)

    if spec.stop_mode == "diverge":
        if spec.degree and spec.degree > 1:
            # Smooth (continuous) iteration count, in [k, k+1): removes colour banding.
            escape_value = (f"k + 1 - math.log(0.5 * math.log(r2) / {math.log(spec.radius)!r}) "
                            f"/ {math.log(spec.degree)!r} if r2 < math.inf else float(k)")
        else:
            escape_value = "float(k)"
        loop = DIVERGE_LOOP.format(step=step_code, bail2=spec.radius ** 2,
                                   escape_value=escape_value, tol2=1e-24)
    elif spec.stop_mode == "converge":
        loop = CONVERGE_LOOP.format(step=step_code, tol2=spec.tolerance ** 2)
    else:
        raise FormulaError(f"unknown stop mode {spec.stop_mode!r} (use 'diverge' or 'converge')")

    indent = " " * 12
    helpers = "".join(power_function_source(n) + "\n\n" for n in sorted(powers))
    return helpers + KERNEL_TEMPLATE.format(setup=textwrap.indent("\n".join(setup_lines), indent),
                                            loop=textwrap.indent(loop.rstrip(), indent))


# --------------------------------------------------------------------------------------
# 5. Compilation, cached by source text: the same formula is only compiled once per run.
# --------------------------------------------------------------------------------------
_cache = {}


def compile_kernel(spec: FractalSpec):
    source = generate_source(spec)
    if source not in _cache:
        namespace = {"prange": prange, "math": math, "njit": njit, **FUNCTIONS}
        exec(source, namespace)        # safe: every formula fragment passed the whitelist above
        _cache[source] = njit(parallel=True)(namespace["kernel"])
    return _cache[source]


def compute(spec: FractalSpec, width_px, height_px, center=None, view_width=None,
            max_iter=None, periodicity=None):
    """Run the fractal over a pixel grid. Returns (n, z):
         n : float array, stop iteration (smooth in diverge mode), -1 where it never stopped
         z : complex array, the last z of every orbit (used e.g. to colour Newton basins)"""
    cx, cy = center if center is not None else spec.center
    view_width = view_width if view_width is not None else spec.width
    max_iter = max_iter if max_iter is not None else spec.max_iter
    periodicity = spec.periodicity if periodicity is None else periodicity

    kernel = compile_kernel(spec)
    out_n = np.empty((height_px, width_px))
    out_z = np.empty((height_px, width_px), dtype=np.complex128)
    pixel_step = view_width / (width_px - 1)
    kernel(float(cx), float(cy), pixel_step, width_px, height_px, int(max_iter),
           bool(periodicity), out_n, out_z)
    return out_n, out_z
