#!/usr/bin/python3
  
import yaml
import sys
from pprint import pprint

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename) as f:
    data = yaml.safe_load(f)


ratios = data['ratios']
omitted_pvalues = None

if 'omitted_pvalues' in data:
    omitted_pvalues = data['omitted_pvalues']

simple_omit = None
if 'simple_omit' in data:
    simple_omit = data['simple_omit']['ratios']

result = {
    "total": ratios,
    "omitted_pvalues": omitted_pvalues,
    "simple_omit": simple_omit,
}

with open(output_filename, "w") as f:
    yaml.dump(result, f)

