#!/bin/python

import os
import subprocess
import sys
import pandas as pd
import numpy as np
import itertools

bin_path='./bin'
data_path='./data'

gp_program = 'operon-gp'
osgp_program = 'operon-osgp'

reps = int(sys.argv[1])

runs = range(0, reps)
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
parameter_space = list(itertools.product(population_size, iteration_count, evaluation_budget, runs))

for f in os.listdir(data_path):
    with open(os.path.join(data_path, f), 'r') as h:
        lines = h.readlines()
        count = len(lines) - 1
        train = int(count * 2 / 3)
        target = lines[0].split(',')[-1].strip('\n')
        problem_name = os.path.splitext(f)[0]

        for pop_size, iter_count, eval_count, run in parameter_space:
            print(problem_name, pop_size, iter_count, eval_count, run)
            output = subprocess.check_output([os.path.join(bin_path, osgp_program), 
                "--dataset", os.path.join(data_path, f), 
                "--target", target, 
                "--train", '0:{}'.format(train), 
                "--selection-pressure", str(100), 
                "--iterations", str(iter_count), 
                "--evaluations", str(eval_count), 
                "--population-size", str(pop_size)]);

            n = df.shape[0]

            for i,line in enumerate(output.split(b'\n')):
                if not line:
                    continue
                meta = [ problem_name, pop_size, iter_count, eval_count, run + 1 ]
                vals = [ np.nan if v == 'nan' else float(v) for v in line.split(b'\t') ]
                df.loc[i + n] = meta + vals
                    
df.to_excel('OSGP.xlsx')

