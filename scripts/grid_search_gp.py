#!/bin/python

from colorama import Back as bg
from colorama import Fore as fg
from colorama import Style as st
from colorama import init
import argparse
import coloredlogs
import itertools
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import subprocess
import sys

init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument('--bin', help='Path to algorithm executable', type=str)
parser.add_argument('--data', help='Data path (can be either a directory or a .csv file', type=str)
parser.add_argument('--reps', help='The number of repetitions for each configuration', type=int)
parser.add_argument('--prefix', help='Prefix to add to output filenames', type=str)
parser.add_argument('--out', help='Location where the produced result files should be saved', type=str)

args = parser.parse_args()

bin_path=args.bin
data_path=args.data
base_path = os.path.dirname(data_path)
reps = args.reps
prefix = args.prefix
results_path = args.out

population_size = [ 1000, 5000, 10000, 50000 ]
iteration_count = [ 0 ]
evaluation_budget = [ 1000000 ]
recombinators = ['basic', 'plus', 'os:100', 'brood:10:10']
selectors = ['random', 'proportional', 'tournament:5', 'tournament:10']

meta_header = ['Problem', 
        'Pop size',
        'Iter count',
        'Eval count',
        'Selector',
        'Recombinator']

output_header = ['Elapsed',
        'Generation', 
        'R2 (train)',
        'R2 (test)',
        'NMSE (train)',
        'NMSE (test)',
        'Avg fit',
        'Avg len',
        'Fitness eval',
        'Local eval',
        'Total eval',
        'Total memory']

header = meta_header + output_header

parameter_space = list(itertools.product(population_size, iteration_count, evaluation_budget, recombinators, selectors))
total_configurations = len(parameter_space)
all_files = list(os.listdir(data_path)) if os.path.isdir(data_path) else [ os.path.basename(data_path) ]
data_files = [ f for f in all_files if f.endswith('.json') ]
data_count = len(data_files)
reps_range = list(range(reps))

coloredlogs.install(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("operon-gp")

idx = 0

total_idx = reps * len(parameter_space) * data_count

problem_results = []

for pop_size, iter_count, eval_count, recombinator, selector in parameter_space:
    idx = idx+1

    gen_count = int(math.ceil(eval_count / pop_size))
    
    for i,f in enumerate(data_files):
        with open(os.path.join(base_path, f), 'r') as h:
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
            config_str     = 'Configuration [{}/{}]\tpopulation size: {}\titerations: {}\tevaluation budget: {}\tselector: {}\trecombinator: {}'.format(idx, total_configurations, pop_size, iter_count, eval_count, selector, recombinator)
            problem_str    = 'Problem [{}/{}]\t{}\tRows: {}\tTarget: {}\tRepetitions: {}'.format(i+1, data_count, problem_name, training_rows, target, reps)
            logger.info(fg.MAGENTA + config_str)
            logger.info(fg.MAGENTA + problem_str)

            problem_result = {}

            for j in reps_range:
                output = subprocess.check_output([bin_path, 
                    "--dataset", os.path.join(base_path, problem_csv), 
                    "--target", target, 
                    "--train", '{}:{}'.format(training_start, training_end), 
                    "--test", '{}:{}'.format(test_start, test_end),
                    "--iterations", str(iter_count), 
                    "--evaluations", str(eval_count), 
                    "--population-size", str(pop_size),
                    "--generations", str(gen_count),
                    "--selector", str(selector),
                    "--recombinator", str(recombinator),
                    "--enable-symbols", "exp,log,sin,cos"]);

                lines = list(filter(lambda x: x, output.split(b'\n')))
                result = '\t'.join([v.decode('ascii') for v in lines[-1].split(b'\t') ])
                logger.info('[{:#2d}/{}]\t{}\t{}'.format(j+1, reps, problem_name, result))

                meta = [ problem_name, pop_size, iter_count, eval_count, selector, recombinator ]
                problem_result[j] = meta  + [ np.nan if v == 'nan' else float(v) for v in lines[-1].split(b'\t') ]

                for i,line in enumerate(lines):
                    vals = [ np.nan if v == 'nan' else float(v) for v in line.split(b'\t') ]

            logger.info(fg.GREEN + config_str)
            logger.info(fg.GREEN + problem_str)

            df                         = pd.DataFrame.from_dict(problem_result, orient='index', columns=header)
            median                     = df.median(axis=0)
            q1, q3                     = df['R2 (train)'].quantile([0.25, 0.75])
            median['R2 (train) IQR']   = q3 - q1
            q1, q3                     = df['R2 (test)'].quantile([0.25, 0.75])
            median['R2 (test) IQR']    = q3 - q1
            q1, q3                     = df['NMSE (train)'].quantile([0.25, 0.75])
            median['NMSE (train) IQR'] = q3 - q1
            q1, q3                     = df['NMSE (test)'].quantile([0.25, 0.75])
            median['NMSE (test) IQR']  = q3 - q1

            for l in df.describe().to_string().split('\n'):
                logger.info(fg.CYAN + l)

            df.to_excel(os.path.join(results_path, '{}_{}_{}_{}_{}_{}_{}.xlsx'.format(prefix, problem_name, pop_size, iter_count, eval_count, selector, recombinator)))
            problem_results.append(df)
                        
df_all = pd.concat(problem_results, axis=0)
group = df_all.groupby(['Problem', 'Pop size', 'Iter count', 'Eval count', 'Selector', 'Recombinator'])
median_all = group.median(numeric_only=False)
q25 = group.quantile(0.25)
q75 = group.quantile(0.75)
median_all['R2 (train) IQR']   = q75['R2 (train)'] - q25['R2 (train)']
median_all['R2 (test) IQR']    = q75['R2 (test)'] - q25['R2 (test)']
median_all['NMSE (train) IQR'] = q75['NMSE (train)'] - q25['NMSE (train)']
median_all['NMSE (test) IQR']  = q75['NMSE (test)'] - q25['NMSE (test)']
for l in median_all.to_string().split('\n'):
    logger.info(fg.YELLOW + l)
df_all.to_excel(os.path.join(results_path, '{}.xlsx'.format(prefix)))
median_all.to_excel(os.path.join(results_path, '{}_median.xlsx'.format(prefix)))

