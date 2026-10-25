"""
Command-line entry point.

    python fractal.py render  fractals/mandelbrot.toml --size 3000x3000 --out mandel.png
    python fractal.py explore fractals/julia.toml
    python fractal.py code    fractals/burning_ship.toml     # print the generated Numba kernel
"""
import argparse
import time

import matplotlib.pyplot as plt

from engine import colour, compute, generate_source, load_spec


def cmd_render(args):
    spec = load_spec(args.spec)
    w, h = (int(v) for v in args.size.lower().split("x"))
    periodicity = False if args.no_periodicity else None

    t0 = time.perf_counter()
    compute(spec, 8, 8)      # tiny first call: Numba compiles the kernel here, so we can time it separately
    t1 = time.perf_counter()
    n, z = compute(spec, w, h, max_iter=args.max_iter, periodicity=periodicity)
    t2 = time.perf_counter()

    out = args.out or args.spec.rsplit("/", 1)[-1].replace(".toml", ".png")
    plt.imsave(out, colour(spec, n, z), origin="lower")
    print(f"{spec.name}: {w}x{h}, compile {t1 - t0:.2f}s, render {t2 - t1:.3f}s, "
          f"{(n < 0).mean() * 100:.1f}% never stopped -> {out}")


def cmd_explore(args):
    from engine.viewer import explore
    explore(load_spec(args.spec), size=args.size)


def cmd_code(args):
    print(generate_source(load_spec(args.spec)))


def main():
    parser = argparse.ArgumentParser(description="General escape-time fractal engine")
    sub = parser.add_subparsers(required=True)

    r = sub.add_parser("render", help="render a spec to a PNG")
    r.add_argument("spec")
    r.add_argument("--size", default="1600x1600", help="WIDTHxHEIGHT in pixels")
    r.add_argument("--out", help="output file (default: <spec name>.png)")
    r.add_argument("--max-iter", type=int, help="override the spec's max_iter")
    r.add_argument("--no-periodicity", action="store_true", help="turn periodicity checking off")
    r.set_defaults(func=cmd_render)

    e = sub.add_parser("explore", help="open the interactive viewer")
    e.add_argument("spec")
    e.add_argument("--size", type=int, default=800, help="viewer resolution (square)")
    e.set_defaults(func=cmd_explore)

    c = sub.add_parser("code", help="print the generated kernel source")
    c.add_argument("spec")
    c.set_defaults(func=cmd_code)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
