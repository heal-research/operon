#!/nix/store/wadc2pdvj6zr3fvqvwgvlszksms46mi1-python3-3.11.9-env/bin/python
# -*- coding: utf-8 -*-

import argparse
import builtins
import itertools
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--bin', help='Path to algorithm executable', type=str)
parser.add_argument('--data', help='Data path (can be either a directory or a .csv file', type=str)
parser.add_argument('--reps', help='The number of repetitions for each configuration', type=int)
parser.add_argument('--prefix', help='Prefix to add to output filenames', type=str)
parser.add_argument('--out', help='Location where the produced result files should be saved', type=str)

args = parser.parse_args()

operon_bin   = args.bin
data_path    = args.data
base_path    = os.path.dirname(data_path)
reps         = args.reps
prefix       = args.prefix
out_path     = args.out

# prepare input files
all_files  = list(os.listdir(data_path)) if os.path.isdir(data_path) else [ os.path.basename(data_path) ]
data_files = sorted([ f for f in all_files if f.endswith('.json') ])
data_count = len(data_files)
reps_range = list(range(reps))

# run experiment
devnull = open(os.devnull, 'w')

logging.basicConfig(level=1)
logger = logging.getLogger("operon-gp")

# some parameters
iterations = 0
enabled_symbols = ['div', 'exp', 'log', 'sin', 'cos', 'sqrt', 'tanh']
disabled_symbols = ['aq']
header = ['problem', 'symbols', 'generation', 'r2_train', 'r2_test', 'mae_train', 'mae_test', 'nmse_train', 'nmse_test', 'avg_fit', 'avg_len', 'eval', 'res_eval', 'jac_eval', 'opt_time', 'seed', 'elapsed', 'energy', 'expression']

results = {k:[] for k in header}

def parse_float(s):
    try:
        return float(s)
    except:
        return np.nan

for i, f in enumerate(data_files):
    with builtins.open(os.path.join(base_path, f), 'r') as h:
        info           = json.load(h)
        metadata       = info['metadata']
        target         = metadata['target']
        training_rows  = metadata['training_rows']
        training_start = training_rows['start']
        training_end   = training_rows['end']
        test_rows      = metadata['test_rows']
        test_start     = test_rows['start']
        test_end       = test_rows['end']
        problem_name   = metadata['name']
        problem_csv    = metadata['filename']

    cmd = [ "perf", "stat", "-e", "power/energy-pkg/", operon_bin,
            "--dataset", os.path.join(base_path, problem_csv),
            "--train", f'{training_start}:{training_end}',
            "--test", f'{test_start}:{test_end}',
            "--target", f'{target}',
            "--maxdepth", '8',
            "--maxlength", '50',
            "--enable-symbols", ','.join(enabled_symbols),
            "--disable-symbols", ','.join(disabled_symbols),
            "--iterations", f'3',
            "--generations", f'1000',
            "--evaluations", f'1000001',
            "--threads", '16'
        ]

    # logger.info('operon command:')
    # logger.info(' '.join(cmd))
    # logger.info(problem_name)
    for j in reps_range:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        lines = list(filter(lambda x: x, output.split(b'\n')))

        for stats in lines[1:-4]:
            results['problem'].append(problem_name)
            results['expression'].append('n/a')
            results['symbols'].append('_'.join(enabled_symbols))
            results['energy'].append('n/a')

            svals = [ parse_float(x) for x in (v.decode('ascii') for v in stats.split(b' ')) if x != '' ]

            for k, h in enumerate(header[2:-2]):
                results[h].append(parse_float(svals[k]))

        expr  = lines[-4]
        watts = lines[-2]
        svals = [ parse_float(x) for x in (v.decode('ascii') for v in watts.split(b' ')) if x != '' ]
        w = svals[0]

        results['expression'][-1] = expr.decode('ascii')
        results['energy'][-1] = w

        #logger.info(svals)
        #logger.info(expr)

        logger.info(f'[{i+1}/{len(data_files)}] {operon_bin} {problem_name}: {j+1}/{len(reps_range)}')

df = pd.DataFrame.from_dict(results)
df.to_csv(sys.stdout, index=None)
