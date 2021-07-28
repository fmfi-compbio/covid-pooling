import argparse
import math
import sys
import time
from copy import deepcopy
from pprint import pprint

import pathos.multiprocessing as mp

import numpy as np
import yaml
from scipy.optimize import minimize, Bounds
from scipy.special import softmax

from helpers import load_observed_counts, load_clades_raw, remove_n, \
    normalize_clades, n_to_deletions
from models import ModelS_jitted


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("counts")
    parser.add_argument("clades")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("-n", "--tries", type=int, default=10)
    parser.add_argument("--no-jacobian", action="store_true",
                        help="Disable symbolic Jacobian in optimisation")
    parser.add_argument("--clip-start", type=int, default=0,
                        help="Ignore first <N> positions of reference")
    return parser.parse_args(argv)


def estimate_ratios(counts, clades, clades_names, rng=None, disable_jacobian=False, clip_start=0):
    if not disable_jacobian:
        raise NotImplementedError("Jacobian is not yet implemented! Add flag '--no-jacobian'")

    rng = rng or np.random.default_rng()  # to assure different seeds in multiprocessing mode

    # counts_np = np.array(counts)
    # clades_np = np.array([clades.values()])

    model = ModelS_jitted(counts, clades, clip_start=clip_start)

    x0 = [math.log(rng.uniform(1, 10)) for _ in clades] + [rng.uniform(1e-4, 0.05) for _ in range(1)]

    bounds = Bounds(
        lb=[0.0 for _ in clades] + [1e-8],
        ub=[50.0 for _ in clades] + [0.25]
    )

    values_during_run = []

    def logger(x):
        nonlocal values_during_run
        weights = softmax(x[:-1])
        subst_rate = x[-1]
        row = {"weights": {k: v for k, v in zip(clades_names, map(float, weights))},
               "subst_rate": float(subst_rate),}
        values_during_run.append(row)
        print(row)

    minimizer_args = {
        "fun": model.opt_fun,
        "x0": np.array(x0),
        #"jac": None if disable_jacobian else model.opt_jac,
        "bounds": bounds,
        "options": {"disp": True},
        "callback": logger
    }

    optimisation_result = minimize(**minimizer_args)

    result_weights = softmax(optimisation_result['x'][:-1])
    result = {
        "ratios": {c: float(r) for c, r in zip(clades_names, result_weights)},
        "subst_rate": float(optimisation_result['x'][-1]),
        "fun": float(optimisation_result['fun']),
        "raw_output": str(optimisation_result),
        "values_during_run": values_during_run
    }

    return result


def estimate_ratios_ntries(counts, clades, clade_names, threads=4, tries=10, disable_jacobian=False, clip_start=0, **kwargs):
    f = estimate_ratios

    with mp.Pool(threads) as p:
        results = p.map(lambda t: f(counts=counts, clades=clades, clades_names=clade_names,
                                    disable_jacobian=disable_jacobian, clip_start=clip_start), range(tries))

    best_result = deepcopy(list(sorted(results, key=lambda r: r['fun']))[0])
    best_result["all_runs"] = [
        result["values_during_run"]
        for result in results
    ]

    return best_result


def estimate_ratios_from_cpp(counts_filename, clades_filename, output_filename, options):
    with open(counts_filename) as f:
        counts = load_observed_counts(f)
    with open(clades_filename) as f:
        clades_raw = load_clades_raw(f)

    clades_reduced = {name: [p[:4] for p in table] for name, table in clades_raw.items()}
    counts_reduced = [p[:4] for p in counts]

    clades = normalize_clades(clades_reduced)
    clade_names = list(clades.keys())
    counts_np = np.array(counts_reduced, dtype=np.float64)
    clades_np = np.array(list(clades.values()), dtype=np.float64)

    result = estimate_ratios_ntries(counts_np, clades_np, clade_names, **options)
    with open(output_filename, "w") as f:
        yaml.dump(result, f)


def main():
    args = parse_args(sys.argv[1:])
    estimate_ratios_from_cpp(counts_filename=args.counts,
                             clades_filename=args.clades,
                             output_filename=args.output,
                             options={"threads": args.threads,
                                      "tries": args.tries,
                                      "disable_jacobian": args.no_jacobian,
                                      "clip_start": args.clip_start,
                                      }
                             )


if __name__ == "__main__":
    main()
