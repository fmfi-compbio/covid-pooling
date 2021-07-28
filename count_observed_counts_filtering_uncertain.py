import argparse
import sys
import pysam

from count_observed_counts import dump_observed_counts, filter_alignment_length, filter_with_query_sequence, BaseCounter
from estimate_ratios_from_read_posteriors import cutoff_uncertain_reads
from helpers import load_fasta, load_posteriors_for_reads, load_alignments_from_bam


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("alignments")
    parser.add_argument("posteriors")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--alignment_low_cutoff", type=int, default=50)
    parser.add_argument("-c", "--cutoff", type=float, default=0.5)
    parser.add_argument("--cutoff-positions", type=int, default=1)
    parser.add_argument("--cutoff-method", choices=("probsum", "entropy"), default="probsum")
    return parser.parse_args(argv)


def count_observed_counts_filtering_uncertain(reference_filename, alignments_filename, posteriors_filename,
                                              output_filename, aligment_low_length_cutoff,
                                              cutoff, cutoff_positions, cutoff_method):
    reference = list(load_fasta(reference_filename))[0][1]
    alignments = load_alignments_from_bam(alignments_filename)

    with open(posteriors_filename) as f:
        clade_names, posteriors = load_posteriors_for_reads(f)

    filtered_posteriors = {read_id: probs for read_id, probs in
                           cutoff_uncertain_reads(posteriors, cutoff, cutoff_positions, cutoff_method)}

    alignments_with_queries = filter_with_query_sequence(alignments)
    alignments_long = filter_alignment_length(alignments_with_queries,
                                              threshold=aligment_low_length_cutoff)
    alignments_certain = filter(lambda al: al.query_name in filtered_posteriors.keys(), alignments_long)

    base_counter = BaseCounter(reference)
    counts_by_pos = base_counter(alignments_certain)

    with open(output_filename, "w") as f:
        dump_observed_counts(counts_by_pos, f)


def main():
    args = parse_args(sys.argv[1:])
    count_observed_counts_filtering_uncertain(reference_filename=args.reference,
                                              alignments_filename=args.alignments,
                                              posteriors_filename=args.posteriors,
                                              output_filename=args.output,
                                              aligment_low_length_cutoff=args.alignment_low_cutoff,
                                              cutoff=args.cutoff,
                                              cutoff_positions=args.cutoff_positions,
                                              cutoff_method=args.cutoff_method)


if __name__ == "__main__":
    main()
