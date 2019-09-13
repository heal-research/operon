#!/bin/python

import os
import subprocess
import sys
import pandas as pd
import numpy as np
import itertools
import coloredlogs
import logging

bin_path='../bin'
data_path='../data'

gp_program = 'operon-gp'
osgp_program = 'operon-osgp'

reps = int(sys.argv[1])

population_size = [ 5000 ]
iteration_count = [ 0 ]
evaluation_budget = [ 10000000 ]

meta_header = ['Problem', 
        'Pop size',
        'Iter count',
        'Eval count',
        'Run index']

output_header = ['Elapsed', 
        'Generation', 
        'Avg length', 
        'Avg quality', 
        'Sel pressure', 
        'Fitness evaluations', 
        'Local evaluations',
        'Total evaluations',
        'R2 (train)',
        'R2 (test)']

header = meta_header + output_header

df = pd.DataFrame(columns=header)
parameter_space = list(itertools.product(population_size, iteration_count, evaluation_budget))
data_files = list(os.listdir(data_path))
data_count = len(data_files)

coloredlogs.install(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("operon-osgp")

for pop_size, iter_count, eval_count in parameter_space:
    logger.info('Population size: {}, iterations: {}, evaluation budged: {}'.format(pop_size, iter_count, eval_count))
    
    for i,f in enumerate(data_files[:1]):
        with open(os.path.join(data_path, f), 'r') as h:
            lines = h.readlines()
            count = len(lines) - 1
            train = int(count * 2 / 3)
            target = lines[0].split(',')[-1].strip('\n')
            problem_name = os.path.splitext(f)[0]
            logger.info('Problem [{}/{}]\t{}\tRows: {}\tTarget: {}'.format(i+1, data_count, problem_name, train, target))

            for j in range(reps):

                output = subprocess.check_output([os.path.join(bin_path, osgp_program), 
                    "--dataset", os.path.join(data_path, f), 
                    "--target", target, 
                    "--train", '0:{}'.format(train), 
                    "--selection-pressure", str(100), 
                    "--iterations", str(iter_count), 
                    "--evaluations", str(eval_count), 
                    "--population-size", str(pop_size)]);

                n = df.shape[0]

                lines = list(filter(lambda x: x, output.split(b'\n')))
                result = '\t'.join([v.decode('ascii') for v in lines[-1].split(b'\t') ])
                logger.info('[{:#2d}/{}]\t{}\t{}'.format(j+1, reps, problem_name, result))

                for i,line in enumerate(lines):
                    meta = [ problem_name, pop_size, iter_count, eval_count, j+1 ]
                    vals = [ np.nan if v == 'nan' else float(v) for v in line.split(b'\t') ]
                    df.loc[i + n] = meta + vals
                        
df.to_excel('OSGP.xlsx')

