import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import random, time, sys, os, json

import _operon as Operon

ds             = Operon.Dataset('../data/Poly-10.csv', has_header=True)
training_range = Operon.Range(0, ds.Rows // 2)
test_range     = Operon.Range(ds.Rows // 2, ds.Rows)
target         = ds.GetVariable('Y')
inputs         = Operon.VariableCollection(v for v in ds.Variables if v.Name != target.Name)

y_train        = ds.Values[training_range.Start:training_range.End, target.Index]

rng            = Operon.RomuTrio(random.randint(1, 1000000))

problem        = Operon.Problem(ds, inputs, target.Name, training_range, test_range)
config         = Operon.GeneticAlgorithmConfig(generations=1000, max_evaluations=1000000, local_iterations=0, population_size=1000, pool_size=1000, p_crossover=1.0, p_mutation=0.25, seed=1)

selector       = Operon.TournamentSelector(objective_index=0)
selector.TournamentSize = 5

pset           = Operon.PrimitiveSet()
pset.SetConfig(Operon.PrimitiveSet.Arithmetic | Operon.NodeType.Exp | Operon.NodeType.Log | Operon.NodeType.Sin | Operon.NodeType.Cos)

minL, maxL     = 1, 50
maxD           = 10
btc            = Operon.BalancedTreeCreator(pset, inputs, bias=0.0)
initializer    = Operon.UniformInitializer(btc, maxD, maxL)
mut_onepoint   = Operon.OnePointMutation()
mut_changeVar  = Operon.ChangeVariableMutation(inputs)
mut_changeFunc = Operon.ChangeFunctionMutation(pset)
mut_replace    = Operon.ReplaceSubtreeMutation(btc, maxD, maxL)
mutation       = Operon.MultiMutation()
mutation.Add(mut_onepoint, 1)
mutation.Add(mut_changeVar, 1)
mutation.Add(mut_changeFunc, 1)
mutation.Add(mut_replace, 1)
crossover      = Operon.SubtreeCrossover(0.9, maxD, maxL)

evaluator      = Operon.RSquaredEvaluator(problem)
evaluator.Budget = 1000 * 1000
evaluator.LocalOptimizationIterations = 0

generator      = Operon.BasicOffspringGenerator(evaluator, crossover, mutation, selector, selector)
reinserter     = Operon.ReplaceWorstReinserter(0)
gp             = Operon.GeneticProgrammingAlgorithm(problem, config, initializer, generator, reinserter)

gen = 0
max_ticks = 50
interval = 1 if config.Generations < max_ticks else int(np.round(config.Generations / max_ticks, 0))
comp = Operon.SingleObjectiveComparison(0)
t0 = time.time()
def report():
    global gen
    best = gp.BestModel(comp)
    bestfit = best.GetFitness(0)
    sys.stdout.write('\r')
    cursor = int(np.round(gen / config.Generations * max_ticks))
    for i in range(cursor):
        sys.stdout.write('\u2588')
    sys.stdout.write(' ' * (max_ticks-cursor))
    sys.stdout.write(f'{100 * gen/config.Generations:.1f}%, generation {gen}/{config.Generations}, train quality: {1-bestfit:.6f}, elapsed: {time.time()-t0:.2f}s')
    sys.stdout.flush()
    gen += 1

gp.Run(rng, report, threads=24)
