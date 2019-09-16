#!/bin/python

import os
import subprocess
import sys
import pandas as pd
import numpy as np
import itertools
import coloredlogs
import logging
import argparse
from colorama import init
from colorama import Fore as fg
from colorama import Back as bg
from colorama import Style as st
import json

init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument('--bin', help='Path to algorithm executable', type=str)
parser.add_argument('--data', help='Data path (can be either a directory or a .csv file', type=str)
parser.add_argument('--reps', help='The number of repetitions for each configuration', type=int)

args = parser.parse_args()

bin_path=args.bin
data_path=args.data
base_path = os.path.dirname(data_path)

reps = args.reps

population_size = [ 1000 ]
iteration_count = [ 0 ]
evaluation_budget = [ 500000 ]

meta_header = ['Problem', 
        'Pop size',
        'Iter count',
        'Eval count',
        'Run index']

output_header = ['Elapsed',
        'Generation', 
        'R2 (train)',
        'R2 (test)',
        'NMSE (train)',
        'NMSE (test)',
        'Average quality',
        'Average length',
        'Fitness evaluations',
        'Local evaluations',
        'Total evaluations']

header = meta_header + output_header

parameter_space = list(itertools.product(population_size, iteration_count, evaluation_budget))
total_configurations = len(parameter_space)
all_files = list(os.listdir(data_path)) if os.path.isdir(data_path) else [ os.path.basename(data_path) ]
data_files = [ f for f in all_files if f.endswith('.json') ]
data_count = len(data_files)
reps_range = list(range(reps))

coloredlogs.install(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("operon-gp")

idx = 0

total_idx = reps * len(parameter_space) * data_count
global_result = {}
for pop_size, iter_count, eval_count in parameter_space:
    idx = idx+1
    
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

            problem_name   = os.path.splitext(f)[0]
            config_str     = 'Configuration [{}/{}]\tpopulation size: {}\titerations: {}\tevaluation budget: {}'.format(idx, total_configurations, pop_size, iter_count, eval_count)
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
                    "--enable-symbols", "exp,log,sin,cos"]);

                n = len(global_result) 

                lines = list(filter(lambda x: x, output.split(b'\n')))
                result = '\t'.join([v.decode('ascii') for v in lines[-1].split(b'\t') ])
                logger.info('[{:#2d}/{}]\t{}\t{}'.format(j+1, reps, problem_name, result))

                meta = [ problem_name, pop_size, iter_count, eval_count, j+1 ]
                problem_result[j] = meta  + [ np.nan if v == 'nan' else float(v) for v in lines[-1].split(b'\t') ]

                for i,line in enumerate(lines):
                    vals = [ np.nan if v == 'nan' else float(v) for v in line.split(b'\t') ]
                    global_result[i + n] = meta + vals

            logger.info(fg.GREEN + config_str)
            logger.info(fg.GREEN + problem_str)

            df2 = pd.DataFrame.from_dict(problem_result, orient='index', columns=header) 
            for l in str(df2.median(axis=0)).split('\n'):
                logger.info(fg.CYAN + l)

            df2.to_excel('GP_{}_{}_{}_{}.xlsx'.format(problem_name, pop_size, iter_count, eval_count))
                        
df = pd.DataFrame.from_dict(global_result, orient='index', columns=header)
for l in str(df.groupby(['Problem', 'Pop size', 'Iter count', 'Eval count']).median(numeric_only=False)).split('\n'):
    logger.info(fg.YELLOW + l)
df.to_excel('GP.xlsx')

