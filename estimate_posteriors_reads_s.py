import argparse
import itertools
import math
import sys
import pathos.multiprocessing as mp

from scipy.special import softmax

from count_observed_counts import filter_alignment_length, filter_with_query_sequence
from helpers import apply_to_cigartuples, load_alignments_from_bam, L2N, \
    load_clades_normalised, load_clades_raw, normalize_clades


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("clades")
    parser.add_argument("reads")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--alignment_low_cutoff", type=int, default=50)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--subst_rate", type=float, default=0.02)
    return parser.parse_args(argv)


class PosteriorCounter:
    EPS = 1e-300

    def __init__(self, clades, subst_rate):
        # values in `clades` are assumed to be normalised
        self.clades = clades
        self.subst_rate = subst_rate
        self.ces = [0 for _ in self.clades.keys()]

    def _count_alongside_cigar(self, op, l, r, q, al):
        # this model ignores N symbols in both clades and alignments
        query = al.query_sequence
        if op == 0 or op == 7 or op == 8:
            for k in range(l):
                base = query[q + k]
                if base == "N":
                    continue
                qln = L2N[base]
                ref_pos = r + k
                for cnum, (clade_name, table) in enumerate(self.clades.items()):
                    r_a = table[ref_pos][qln]
                    r_a_prime = r_a
                    q_a = (1 - self.subst_rate) * r_a_prime + self.subst_rate / 3 * (1 - r_a_prime)

                    ce = -math.log(self.EPS + q_a)
                    self.ces[cnum] += ce

    def dump_cross_entropies(self):
        return self.ces

    def dump_posteriors(self):
        return list(softmax([-x for x in self.ces]))


def eval_posteriors(al, clades, subst_rate):
    counter = PosteriorCounter(clades, subst_rate)
    apply_to_cigartuples(counter._count_alongside_cigar, al)
    result = [al.query_name] + list(counter.dump_posteriors())
    #print(result)
    return result


def eval_cross_entropies(al, clades, subst_rate):
    counter = PosteriorCounter(clades, subst_rate)
    apply_to_cigartuples(counter._count_alongside_cigar, al)
    result = [al.query_name] + list(counter.dump_cross_entropies())
    #print(result)
    return result


class SimplifiedAlignment:  # is needed to use pathos.multiprocessing library
    def __init__(self, alignment):
        self.query_name = alignment.query_name
        self.reference_start = alignment.reference_start
        self.cigartuples = alignment.cigartuples
        self.query_sequence = alignment.query_sequence


def merge_paired_reads_ces(posteriors):
    data = dict()
    for line in posteriors:
        id, nums = line[0], line[1:]
        if id not in data:
            data[id] = nums
        else:
            data[id] = [o + n for o, n in zip(data[id], nums)]  # summing log-posteriors for paired reads
    result = [[id] + nums for id, nums in data.items()]
    return result


def estimate_posterior_reads(clades_filename, reads_filename,
                             output_filename, alignment_length_low_cutoff, threads_count,
                             subst_rate):
    with open(clades_filename) as f:
        clades_raw = load_clades_raw(f)
    clades_reduced = {k: [p[:4] for p in t] for k, t in clades_raw.items()}
    clades = normalize_clades(clades_reduced)

    alignments = load_alignments_from_bam(reads_filename)

    def is_good(alignment):
        return (not alignment.is_secondary) and (not alignment.is_supplementary) and (not alignment.is_unmapped) and (
            not alignment.mate_is_unmapped)

    good_alignments = list(filter(is_good, alignments))
    alignments_long = filter_alignment_length(good_alignments, threshold=alignment_length_low_cutoff)

    alignments_long = itertools.islice(alignments_long, 500)  # @DEBUG REMOVE ME!!

    ces = evaluate_posteriors_multiprocessing(alignments_long, threads_count, clades, subst_rate)

    merged_ces = merge_paired_reads_ces(ces)

    merged_posteriors = map(lambda x: [x[0]] + list(softmax([-a for a in x[1:]])), merged_ces)

    with open(output_filename, "w") as f:
        dump_posteriors(clades, merged_posteriors, f)


def filter_with_cigartuples(alignments):
    return filter(lambda al: al.cigartuples is not None, alignments)


def dump_posteriors(clades, posteriors, f):
    header = "read_id," + ",".join(clades.keys())
    print(header, file=f)
    for posterior in posteriors:
        print(",".join(map(str, posterior)), file=f)


def evaluate_posteriors_multiprocessing(alignments, threads_count, clades, subst_rate):
    als = map(lambda al: SimplifiedAlignment(al), alignments)
    with mp.Pool(threads_count) as p:
        posteriors = p.map(lambda al: eval_cross_entropies(al, clades, subst_rate), als)
    return posteriors


def main():
    args = parse_args(sys.argv[1:])
    estimate_posterior_reads(clades_filename=args.clades,
                             reads_filename=args.reads,
                             output_filename=args.output,
                             alignment_length_low_cutoff=args.alignment_low_cutoff,
                             subst_rate=args.subst_rate,
                             threads_count=args.threads)


if __name__ == "__main__":
    main()
