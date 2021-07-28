import argparse
import sys

from helpers import load_fasta, apply_to_cigartuples, ALPHABET, L2N, load_alignments_from_bam, load_alignments_from_sam


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=str)
    parser.add_argument("alignment", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--alignment_low_cutoff", type=int, default=50)
    parser.add_argument("--type", default="bam", choices=["sam", "bam"])
    return parser.parse_args(argv)


def dump_observed_counts(counts_by_pos, f):
    print("position,letter,count", file=f)
    for position, counts in enumerate(counts_by_pos):
        for letter, count in zip(ALPHABET, counts):
            print(f"{position+1},{letter},{count}", file=f)


class BaseCounter:
    def __init__(self, reference):
        self.reference = reference
        self.counts_by_pos = [[0 for _ in ALPHABET] for _ in self.reference]

    def _count_alongside_cigar(self, op, l, r, q, al):
        query = al.query_sequence
        if op == 0 or op == 7 or op == 8:  # matching (with or without error)
            for k in range(l):
                base = query[q + k]
                if base == "N":
                    continue
                self.counts_by_pos[r + k][L2N[base]] += 1
        if op == 2 or op == 3:  # deletion in read
            for k in range(l):
                self.counts_by_pos[r + k][L2N["-"]] += 1

    def __call__(self, alignments):
        empty_cigartuples_count = 0
        for alignment in alignments:
            if alignment.cigartuples is not None:
                apply_to_cigartuples(self._count_alongside_cigar, alignment)
            else:
                empty_cigartuples_count += 1
                if empty_cigartuples_count % 100 == 0:
                    print(f"{empty_cigartuples_count} empty cigar tuples so far!")
        return self.counts_by_pos


def filter_with_query_sequence(alignments):
    return filter(lambda al: al.query_sequence is not None, alignments)


def filter_alignment_length(alignments, threshold):
    return filter(lambda al: al.query_alignment_length >= threshold, alignments)


def count_observed_counts(reference_filename,
                          alignments_filename,
                          output_filename,
                          alignment_length_low_cutoff,
                          alignment_type):
    reference = list(load_fasta(reference_filename))[0][1]
    if alignment_type == "bam":
        alignments = load_alignments_from_bam(alignments_filename)
    elif alignment_type == "sam":
        alignments = load_alignments_from_sam(alignments_filename)

    filtered_alignments = filter_alignment_length(filter_with_query_sequence(alignments),
                                                  threshold=alignment_length_low_cutoff)

    base_counter = BaseCounter(reference)
    counts_by_pos = base_counter(filtered_alignments)

    with open(output_filename, "w") as f:
        dump_observed_counts(counts_by_pos, f)


def main():
    args = parse_args(sys.argv[1:])
    count_observed_counts(reference_filename=args.reference,
                          alignments_filename=args.alignment,
                          output_filename=args.output,
                          alignment_length_low_cutoff=args.alignment_low_cutoff,
                          alignment_type=args.type)


if __name__ == "__main__":
    main()
