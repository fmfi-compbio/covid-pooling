import argparse
import itertools
import math
import sys
import numpy as np
import yaml

from helpers import load_posteriors_for_reads, entropy


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("posteriors")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-c", "--cutoff", type=float, default=0.5)
    parser.add_argument("--cutoff-positions", type=int, default=1)
    parser.add_argument("--cutoff-method", choices=("probsum", "entropy"), default="probsum")
    parser.add_argument("-m", "--method", choices=("max", "sum"), default="max")
    return parser.parse_args(argv)


def cutoff_uncertain_reads(posteriors, cutoff, cutoff_positions, cutoff_method):
    if cutoff_method == "probsum":
        eval_certainty = lambda probs: sum(itertools.islice(sorted(probs, reverse=True), cutoff_positions))
    elif cutoff_method == "entropy":
        eval_certainty = lambda probs: 1 - entropy(probs)/math.log(len(probs))
    else:
        raise NotImplementedError(f"Cutoff method {cutoff_method} is not implemented!")

    for read_id, probs in posteriors:
        certainty = eval_certainty(probs)
        if certainty > cutoff:
            yield read_id, probs


def estimate_ratios_from_read_posteriors(posteriors_filename, output_filename, cutoff,
                                         cutoff_positions, cutoff_method, method):
    with open(posteriors_filename) as f:
        clade_names, posteriors = load_posteriors_for_reads(f)
    filtered_posteriors = list(cutoff_uncertain_reads(posteriors, cutoff, cutoff_positions, cutoff_method))

    if method == "max":
        result = count_maxima(clade_names, filtered_posteriors)
    elif method == "sum":
        result = count_sums(clade_names, filtered_posteriors)
    else:
        raise NotImplementedError(f"Method {method} is not implemented!")

    with open(output_filename, "w") as f:
        yaml.dump(result, f)


def count_sums(clade_names, posteriors):
    total_probs = [0 for _ in clade_names]
    for read_id, probs in posteriors:
        for cnum in range(len(clade_names)):
            total_probs[cnum] += probs[cnum]
    result = {clade_names[n]: total_probs[n] / len(posteriors) for n in range(len(clade_names))}
    return result


def count_maxima(clade_names, posteriors):
    wins = [0 for _ in clade_names]
    for read_id, probs in posteriors:
        sol_num = int(np.argmax(probs))
        wins[sol_num] += 1
    result = {clade_names[n]: wins[n] / sum(wins) for n in range(len(clade_names))}
    return result


def main():
    args = parse_args(sys.argv[1:])
    estimate_ratios_from_read_posteriors(posteriors_filename=args.posteriors,
                                         output_filename=args.output,
                                         cutoff=args.cutoff,
                                         cutoff_positions=args.cutoff_positions,
                                         cutoff_method=args.cutoff_method,
                                         method=args.method)


if __name__ == "__main__":
    main()