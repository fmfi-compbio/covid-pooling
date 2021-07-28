# COVID-Pooling

## Install

```bash
conda env create -n covid-pooling python=3.7
conda activate covid-pooling
conda config --add channels r
conda config --add channels bioconda
conda install pysam numpy matplotlib scipy pathos
```

## Usage

### 1. Estimate posterior probabilities for the individual reads

```bash
python estimate_posteriors_reads_s.py.py variants.csv reads.bam -o posteriors.csv -t 10 --subst_rate 0.05 
```

Output: `posteriors.csv`

### 2. Count the reads profile with filtering uncertain reads

```bash
python count_observed_counts_filtering_uncertain.py ref.fa reads.bam posteriors.csv -c 0.75 -o observed.csv
```

Output: `observed.csv`

### 3. Estimate the variant proportions

```bash
python fastend_cpp_s.py observed.csv variants.csv --no-jacobian -t 4 -n 10 -o calculations.yaml
```\

Output: `calculations.yaml`

### 4. Extract the final result

```bash
python extract_weights.py calculations.yaml result.yaml
```

Output: `result.yaml`