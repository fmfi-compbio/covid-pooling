import math
from collections import defaultdict

import pysam


def invert_alphabet(alphabet):
    return {letter: num for num, letter in enumerate(alphabet)}


SHORT_ALPHABET = "ACGT"
SHORT_L2N = invert_alphabet(SHORT_ALPHABET)
ALPHABET = "ACGT-"
EXTENDED_ALPHABET = ALPHABET+"N"
L2N = invert_alphabet(ALPHABET)
EXTENDED_L2N = invert_alphabet(EXTENDED_ALPHABET)


def load_fasta(filename: str):
    with open(filename) as f:
        yield from load_fasta_fd(f)


def load_fasta_fd(f):
    label, buffer = "", []
    for line in f:
        if len(line) > 0 and line[0] == ">":
            # new label
            if len(buffer) > 0:
                yield label, "".join(buffer)
            label = line.strip()[1:]
            buffer = []
        else:
            buffer.append(line.strip())
    if len(buffer) > 0:
        yield label, "".join(buffer)


def load_observed_counts(fd):
    global ALPHABET
    header = fd.readline().strip()
    assert header == "position,letter,count", f"Expected 'position,letter,count' header, got '{header}' instead"
    result_raw = defaultdict(dict)
    for line in fd:
        row = line.strip().split(",")
        pos, letter, count = int(row[0])-1, row[1], int(row[2])
        result_raw[pos][letter] = count
    result = [[result_raw.get(pos, {}).get(letter, 0) for letter in ALPHABET]
              for pos in range(1 + max(result_raw.keys(), default=-1))]
    return result


def load_clades_raw(fd, header=False, zero_based=True):
    global EXTENDED_L2N

    if header:
        header = fd.readline().strip()
        assert header == "clade,position,letter,probability"

    result = defaultdict(list)
    for line in fd:
        row = line.strip().split(",")
        clade, position, lnum, count = \
            row[0], int(row[1]), EXTENDED_L2N.get(row[2], None), float(row[3])

        assert lnum is not None, f"Letter {row[2]} is not in the extended " \
                                 f"alphabet {EXTENDED_ALPHABET}!"

        if not zero_based:
            position -= 1

        while position >= len(result[clade]):
            result[clade].append([0 for _ in EXTENDED_L2N])
        if row[2] is not None:
            result[clade][position][lnum] = count
    clade_names = list(result.keys())

    # control that all clades has the same number of positions
    for clade in clade_names:
        assert len(result[clade]) == len(result[clade_names[0]])

    return dict(result)


def remove_n(clades):
    global EXTENDED_L2N, L2N
    result = {}
    for clade_name, table in clades.items():
        result[clade_name] = []
        for position, counts in enumerate(table):
            new_counts = [counts[EXTENDED_L2N[letter]] for letter, lnum in L2N.items()]
            result[clade_name].append(new_counts)
    return result


def normalize_clades(clades, pseudocount=0.0000000001):
    result = {}
    for clade_name, table in clades.items():
        result[clade_name] = []
        for counts in table:
            total = sum(c+pseudocount for c in counts)
            new_counts = [(c + pseudocount)/total for c in counts]
            result[clade_name].append(new_counts)
    return result


def entropy(p):
    return -sum(x * math.log(x) if x > 0 else 0 for x in p)


def cross_entropy(p, q):
    assert len(p) == len(q), f"Different lengths! p={p} q={q}"
    return -sum(sorted(p[i] * math.log(max(1e-300, q[i])) for i in range(len(p))))


def add_noise(v, eps=0.02):
    res = [(1-eps) * x + eps/(len(v)-1) * (1 - x) for x in v]
    return res


def apply_to_cigartuples(fun, alignment, *args, **kwargs):
    """
    M	BAM_CMATCH	0
    I	BAM_CINS	1
    D	BAM_CDEL	2
    N	BAM_CREF_SKIP	3
    S	BAM_CSOFT_CLIP	4
    H	BAM_CHARD_CLIP	5
    P	BAM_CPAD	6
    =	BAM_CEQUAL	7
    X	BAM_CDIFF	8
    B	BAM_CBACK	9 (????!)
    """
    query_pos = 0
    reference_pos = alignment.reference_start
    for op, length in alignment.cigartuples:
        fun(op, length, reference_pos, query_pos, alignment, *args, **kwargs)
        if op == 0 or op == 7 or op == 8:
            reference_pos += length
            query_pos += length
        elif op == 1 or op == 4:
            query_pos += length
        elif op == 2 or op == 3:
            reference_pos += length
        elif op == 5 or op == 6:
            pass
        else:
            raise Exception(f"Operation code of cigar tuple is outside of range [0-8]: "
                            f"op={op}, length={length}")


def load_posteriors_for_reads(f):
    header = f.readline().strip().split(",")
    clade_names = header[1:]
    probs = []
    for line in f:
        row = line.strip().split(",")
        read_id = row[0]
        probs.append((read_id, tuple(map(float, row[1:]))))
    return clade_names, probs


def load_alignments_from_bam(alignment_filename):
    return pysam.AlignmentFile(alignment_filename, "rb").fetch()


def load_alignments_from_sam(alignments_filename):
    return pysam.AlignmentFile(alignments_filename).fetch()


def n_to_deletions(clades):
    result = {}
    n_pos = EXTENDED_L2N['N']
    del_pos = L2N['-']

    for clade_name, table in clades.items():
        result[clade_name] = []
        for counts in table:
            cc = counts[:n_pos] + counts[n_pos+1:]
            cc[del_pos] += counts[n_pos]
            result[clade_name].append(cc)
    return result


def load_clades_normalised(f, n_as_deletions):
    clades_raw = load_clades_raw(f)
    if n_as_deletions:
        clades_reduced = n_to_deletions(clades_raw)
    else:
        clades_reduced = remove_n(clades_raw)
    clades = normalize_clades(clades_reduced)
    return clades
