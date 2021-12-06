# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

import sys
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import random
import operon._operon as op

class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """ Builds a symbolic regression model.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------

    Examples
    --------
    >>> from operon import SymbolicRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = SymbolicRegressor()
    >>> estimator.fit(X, y)
    """
    def __init__(self,
        allowed_symbols                = 'add,sub,mul,div,constant,variable',
        crossover_probability          = 1.0,
        crossover_internal_probability = 0.9,
        mutation                       = { 'onepoint' : 1.0, 'changevar' : 1.0, 'changefunc' : 1.0, 'insertsubtree' : 1.0, 'replacesubtree' : 1.0, 'removesubtree' : 1.0 },
        mutation_probability           = 0.25,
        offspring_generator            = 'basic',
        reinserter                     = 'replace-worst',
        objectives                     = ['r2'],
        max_length                     = 50,
        max_depth                      = 10,
        initialization_method          = 'btc',
        female_selector                = 'tournament',
        male_selector                  = 'tournament',
        population_size                = 1000,
        pool_size                      = None,
        generations                    = 1000,
        max_evaluations                = int(1000 * 1000),
        local_iterations               = 0,
        max_selection_pressure         = 100,
        comparison_factor              = 0,
        brood_size                     = 10,
        tournament_size                = 5,
        btc_bias                       = 0.0,
        n_threads                      = 1,
        time_limit                     = None,
        random_state                   = None
        ):

        # validate parameters
        self.allowed_symbols           = 'add,sub,mul,div,constant,variable' if allowed_symbols is None else allowed_symbols
        self.crossover_probability     = 1.0 if crossover_probability is None else crossover_probability
        self.crossover_internal_probability = 0.9 if crossover_internal_probability is None else crossover_internal_probability
        self.mutation                  = { 'onepoint' : 1.0, 'changevar' : 1.0, 'changefunc' : 1.0, 'insertsubtree' : 1.0, 'replacesubtree' : 1.0, 'removesubtree' : 1.0 } if mutation is None else mutation
        self.mutation_probability      = 0.25 if mutation_probability is None else mutation_probability
        self.offspring_generator       = 'basic' if offspring_generator is None else offspring_generator
        self.reinserter                = 'replace-worst' if reinserter is None else reinserter
        self.objectives                = ['r2'] if objectives is None else objectives
        self.max_length                = 50 if max_length is None else int(max_length)
        self.max_depth                 = 10 if max_depth is None else int(max_depth)
        self.initialization_method     = 'btc' if initialization_method is None else initialization_method
        self.female_selector           = 'tournament' if female_selector is None else female_selector
        self.male_selector             = 'tournament' if male_selector is None else male_selector
        self.population_size           = 1000 if population_size is None else int(population_size)
        self.pool_size                 = population_size if pool_size is None else int(pool_size)
        self.generations               = 1000 if generations is None else int(generations)
        self.max_evaluations           = 1000000 if max_evaluations is None else int(max_evaluations)
        self.local_iterations          = 0 if local_iterations is None else int(local_iterations)
        self.max_selection_pressure    = 100 if max_selection_pressure is None else int(max_selection_pressure)
        self.comparison_factor         = 0 if comparison_factor is None else comparison_factor
        self.brood_size                = 10 if brood_size is None else int(brood_size)
        self.tournament_size           = 5 if tournament_size is None else tournament_size # todo: set for both parent selectors
        self.btc_bias                  = 0.0 if btc_bias is None else btc_bias
        self.n_threads                 = 1 if n_threads is None else int(n_threads)
        self.time_limit                = sys.maxsize if time_limit is None else int(time_limit)
        self.random_state              = random_state
        self._model_vars               = {}
        self._interpreter              = op.Interpreter()


    def __init_primitive_config(self, allowed_symbols):
        symbols = allowed_symbols.split(',')

        known_symbols = {
            'add'    : op.NodeType.Add,
            'mul'    : op.NodeType.Mul,
            'sub'    : op.NodeType.Sub,
            'div'    : op.NodeType.Div,
            'aq'     : op.NodeType.Aq,
            'fmin'   : op.NodeType.Fmin,
            'fmax'   : op.NodeType.Fmax,
            'pow'    : op.NodeType.Pow,
            'abs'    : op.NodeType.Abs,
            'acos'   : op.NodeType.Acos,
            'asin'   : op.NodeType.Asin,
            'atan'   : op.NodeType.Atan,
            'cbrt'   : op.NodeType.Cbrt,
            'ceil'   : op.NodeType.Ceil,
            'cos'    : op.NodeType.Cos,
            'cosh'   : op.NodeType.Cosh,
            'erf'    : op.NodeType.Erf,
            'erfc'   : op.NodeType.Erfc,
            'exp'    : op.NodeType.Exp,
            'floor'  : op.NodeType.Floor,
            'log'    : op.NodeType.Log,
            'log1p'  : op.NodeType.Log1p,
            'sin'    : op.NodeType.Sin,
            'sinh'   : op.NodeType.Sinh,
            'sqrt'   : op.NodeType.Sqrt,
            'square' : op.NodeType.Square,
            'tan'    : op.NodeType.Tan,
            'tanh'   : op.NodeType.Tanh,
            'constant' : op.NodeType.Constant,
            'variable' : op.NodeType.Variable,
        }

        config = op.NodeType(0)
        for s in symbols:
            if s in known_symbols:
                config |= known_symbols[s]
            else:
                raise ValueError('Unknown symbol type {}'.format(s))

        return config


    def __init_creator(self, initialization_method, pset, inputs):
        if initialization_method == 'btc':
            return op.BalancedTreeCreator(pset, inputs, self.btc_bias)

        elif initialization_method == 'ptc2':
            return op.ProbabilisticTreeCreator(pset, inputs)

        elif initialization_method == 'koza':
            return op.GrowTreeCreator(pset, inputs)

        raise ValueError('Unknown initialization method {}'.format(initialization_method))


    def __init_selector(self, selection_method, comp):
        if selection_method == 'tournament':
            selector = op.TournamentSelector(comp)
            selector.TournamentSize = self.tournament_size
            return selector

        elif selection_method == 'proportional':
            selector = op.ProportionalSelector(comp)
            return selector

        elif selection_method == 'random':
            selector = op.RandomSelector()
            return selector

        raise ValueError('Unknown selection method {}'.format(selection_method))


    def __init_evaluator(self, objective, problem, interpreter):
        if objective == 'r2':
            return op.RSquaredEvaluator(problem, interpreter)

        elif objective == 'nmse':
            return op.NormalizedMeanSquaredErrorEvaluator(problem, interpreter)

        elif objective == 'rmse':
            return op.RootMeanSquaredErrorEvaluator(problem, interpreter)

        elif objective == 'mse':
            return op.MeanSquaredErrorEvaluator(problem, interpreter)

        elif objective == 'mae':
            return op.MeanAbsoluteErrorEvaluator(problem, interpreter)

        elif objective == 'length':
            return op.LengthEvaluator(problem)

        elif objective == 'shape':
            return op.ShapeEvaluator(problem)

        raise ValueError('Unknown objective {}'.format(objectives))


    def __init_generator(self, generator_name, evaluator, crossover, mutator, female_selector, male_selector):
        if male_selector is None:
            male_selector = female_selector

        if generator_name == 'basic':
            return op.BasicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)

        elif generator_name == 'os':
            generator = op.OffspringSelectionGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.MaxSelectionPressure = self.max_selection_pressure
            generator.ComparisonFactor = self.comparison_factor
            return generator

        elif generator_name == 'brood':
            generator = op.BroodOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        elif generator_name == 'poly':
            generator = op.PolygenicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        raise ValueError('Unknown generator method {}'.format(generator_name))


    def __init_reinserter(self, reinserter_name, comp):
        if reinserter_name == 'replace-worst':
            return op.ReplaceWorstReinserter(comp)

        elif reinserter_name == 'keep-best':
            return op.KeepBestReinserter(comp)

        raise ValueError('Unknown reinsertion method {}'.format(reinserter_name))


    def __init_mutation(self, mutation_name, inputs, pset, creator):
        if mutation_name == 'onepoint':
            return op.OnePointMutation()

        elif mutation_name == 'changevar':
            return op.ChangeVariableMutation(inputs)

        elif mutation_name == 'changefunc':
            return op.ChangeFunctionMutation(pset)

        elif mutation_name == 'replacesubtree':
            return op.ReplaceSubtreeMutation(creator, self.max_depth, self.max_length)

        elif mutation_name == 'insertsubtree':
            return op.InsertSubtreeMutation(creator, self.max_depth, self.max_length, pset)

        elif mutation_name == 'removesubtree':
            return op.RemoveSubtreeMutation(pset)

        raise ValueError('Unknown mutation method {}'.format(mutation_name))


    def get_model_string(self, precision):
        if len(self._model_vars) == 0:
            print('warning: model contains no variables', file=sys.stderr)
        return op.InfixFormatter.Format(self._model, self._model_vars, precision)


    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y                  = check_X_y(X, y, accept_sparse=False)
        D                     = np.column_stack((X, y))

        ds                    = op.Dataset(D)
        target                = max(ds.Variables, key=lambda x: x.Index) # last column is the target

        inputs                = op.VariableCollection(v for v in ds.Variables if v.Index != target.Index)
        training_range        = op.Range(0, ds.Rows)
        test_range            = op.Range(ds.Rows-1, ds.Rows) # hackish, because it can't be empty
        problem               = op.Problem(ds, inputs, target.Name, training_range, test_range)

        pset                  = op.PrimitiveSet()
        pcfg                  = self.__init_primitive_config(self.allowed_symbols)
        pset.SetConfig(pcfg)

        creator               = self.__init_creator(self.initialization_method, pset, inputs)

        single_objective      = True if len(self.objectives) == 1 else False

        evaluators = [] # placeholder for the evaluator(s)

        for obj in self.objectives:
            eval_         = self.__init_evaluator(obj, problem, self._interpreter)
            eval_.Budget  = self.max_evaluations
            eval_.LocalOptimizationIterations = self.local_iterations
            evaluators.append(eval_)

        if single_objective:
            evaluator = evaluators[0]
        else:
            evaluator = op.MultiEvaluator(problem)
            for eval_ in evaluators:
                evaluator.Add(eval_)
            evaluator.LocalOptimizationIterations = self.local_iterations
            evaluator.Budget = self.max_evaluations

        comparison            = op.SingleObjectiveComparison(0) if single_objective else op.CrowdedComparison()

        female_selector       = self.__init_selector(self.female_selector, comparison)
        male_selector         = self.__init_selector(self.male_selector, comparison)
        reinserter            = self.__init_reinserter(self.reinserter, comparison)
        cx                    = op.SubtreeCrossover(self.crossover_internal_probability, self.max_depth, self.max_length)

        mut                   = op.MultiMutation()
        mut_list = [] # this list is needed as a placeholder to keep alive the mutation operators objects (since the multi-mutation only stores references)
        for k in self.mutation:
            v = self.mutation[k]
            m = self.__init_mutation(k, inputs, pset, creator)
            mut.Add(m, v)
            mut_list.append(m)

        generator             = self.__init_generator(self.offspring_generator, evaluator, cx, mut, female_selector, male_selector)

        min_arity, max_arity  = pset.FunctionArityLimits()
        initializer           = op.UniformInitializer(creator, min_arity+1, self.max_length)
        initializer.MinDepth  = 1
        initializer.MaxDepth  = 1000

        if self.random_state is None:
            self.random_state = random.getrandbits(64)

        config                = op.GeneticAlgorithmConfig(
                                    generations      = self.generations,
                                    max_evaluations  = self.max_evaluations,
                                    local_iterations = self.local_iterations,
                                    population_size  = self.population_size,
                                    pool_size        = self.pool_size,
                                    p_crossover      = self.crossover_probability,
                                    p_mutation       = self.mutation_probability,
                                    seed             = self.random_state,
                                    time_limit       = self.time_limit
                                    )

        sorter                = None if single_objective else op.RankSorter()
        gp                    = op.GeneticProgrammingAlgorithm(problem, config, initializer, generator, reinserter) if single_objective else op.NSGA2Algorithm(problem, config, initializer, generator, reinserter, sorter) 
        rng                   = op.RomuTrio(np.uint64(config.Seed))

        gp.Run(rng, None, self.n_threads)
        comp                  = op.SingleObjectiveComparison(0)
        best                  = gp.BestModel(comp)
        nodes                 = best.Genotype.Nodes
        n_vars                = len([ node for node in nodes if node.IsVariable ])

        # add four nodes at the top of the tree for linear scaling
        y_pred                = op.Evaluate(self._interpreter, best.Genotype, ds, training_range)
        scale, offset         = op.FitLeastSquares(y_pred, y)
        nodes.extend([ op.Node.Constant(scale), op.Node.Mul(), op.Node.Constant(offset), op.Node.Add() ])

        self._model           = op.Tree(nodes).UpdateNodes()

        # update model vars dictionary
        self._model_vars = { node.HashValue : ds.GetVariable(node.HashValue).Name for node in nodes if node.IsVariable }

        self._stats = {
            'model_length':        self._model.Length - 4, # do not count scaling nodes?
            'model_complexity':    self._model.Length - 4 + 2 * n_vars,
            'generations':         gp.Generation,
            'fitness_evaluations': evaluator.FitnessEvaluations,
            'local_evaluations':   evaluator.LocalEvaluations,
            'random_state':        self.random_state
        }

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        ds = op.Dataset(X)
        rg = op.Range(0, ds.Rows)
        return op.Evaluate(self._interpreter, self._model, ds, rg)

