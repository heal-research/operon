import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='whitegrid', font_scale=1.5)
from scipy import stats
import sys

# operon python bindings
import _operon as Operon

ds                     = Operon.Dataset('../data/Poly-10.csv', has_header=True)
training_range         = Operon.Range(0, ds.Rows // 2)
test_range             = Operon.Range(training_range.End, ds.Rows)

target                 = ds.GetVariable('Y')
inputs                 = Operon.VariableCollection(v for v in ds.Variables if v.Name != target.Name)

y_train                = ds.Values[0:training_range.End, target.Index]

grammar                = Operon.PrimitiveSet()
grammar.SetConfig(Operon.PrimitiveSet.Arithmetic | Operon.NodeType.Exp | Operon.NodeType.Log | Operon.NodeType.Sin | Operon.NodeType.Cos)

population_size        = 1000
min_length, max_length = 1, 50
min_depth, max_depth   = 1, 10
p_crossover            = 1.0
p_mutation             = 0.25
p_internal             = 0.9 # for crossover
max_generations        = 1000

# define gp operators
rng             = Operon.RomuTrio(random.randint(1, 100000))
# tree creator
initial_lengths = np.random.randint(min_length, max_length+1, population_size)
btc             = Operon.BalancedTreeCreator(grammar, inputs, bias=0.0)

# crossover
crossover      = Operon.SubtreeCrossover(p_internal, max_depth, max_length)

# mutation
mut_onepoint   = Operon.OnePointMutation()
mut_changeVar  = Operon.ChangeVariableMutation(inputs)
mut_changeFunc = Operon.ChangeFunctionMutation(grammar)
mut_replace    = Operon.ReplaceSubtreeMutation(btc, max_depth, max_length)
mutation       = [ mut_onepoint, mut_changeFunc, mut_replace, mut_changeVar ]

# selection operator (tournament selection
def select(pop, fit, group_size=3):
    i = int(population_size * random.random()) # this is ok since we don't deal with huge numbers
    s = 1
    while s < group_size:
        s += 1
        j = int(population_size * random.random()) 
        if fit[i] < fit[j]:
            i = j
        
    return pop[i]

# offspring generator (applies selection, crossover and mutation)
# to create one offspring individual
def generate(pop, fit):
    do_crossover = random.uniform(0, 1) < p_crossover
    do_mutation = random.uniform(0, 1) < p_mutation

    p1 = select(pop, fit)

    if do_crossover:
        p2 = select(pop, fit)
        child = crossover(rng, p1, p2)
    
    if do_mutation:
        op = random.choice(mutation)
        p1 = child if do_crossover else p1
        child = op(rng, child)
        
    if do_crossover or do_mutation:
        return child
    else:
        return p1

# run gp
if __name__ == '__main__':
    t0 = time.time()
    pop = [btc(rng, l, 0, 0) for l in initial_lengths]
    fit = Operon.CalculateFitness(pop, ds, training_range, target.Name, metric='rsquared')
    best = np.argmax(fit)
    r2_test = Operon.CalculateFitness(pop[best], ds, test_range, target.Name, metric='rsquared')

    max_ticks = 50
    interval = 1 if max_generations < max_ticks else int(np.round(max_generations / max_ticks, 0))

    for gen in range(max_generations+1):
        pop = [pop[best] if i == best else generate(pop, fit) for i in range(population_size)]
        fit = Operon.CalculateFitness(pop, ds, training_range, target.Name, metric='rsquared')
        new_best = np.argmax(fit)
        if new_best != best:
            best = new_best
            r2_test = Operon.CalculateFitness(pop[best], ds, test_range, target.Name, metric='rsquared')
        
        sys.stdout.write('\r')
        cursor = int(np.round(gen / max_generations * max_ticks))
        for i in range(cursor):
            sys.stdout.write('\u2588')
        sys.stdout.write(' ' * (max_ticks-cursor))
        sys.stdout.write(f' {100 * gen/max_generations:.1f}%, generation {gen}/{max_generations}, train quality: {fit[best]:.6f}, test quality: {r2_test:.6f}, elapsed: {time.time()-t0:.2f}s')
        sys.stdout.flush()
            
    t1 = time.time()
    best_model, best_quality = pop[best], fit[best]
    print(f'\nModel length: {best_model.Length}, model depth: {best_model.Depth}\n{Operon.InfixFormatter.Format(best_model, ds, 2)}')

