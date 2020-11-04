import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin 
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import random
import _operon


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
        error_metric                   = 'r2',
        max_length                     = 50,
        max_depth                      = 10,
        initialization_method          = 'btc',
        female_selector                = 'tournament',
        male_selector                  = 'tournament',
        population_size                = 1000,
        pool_size                      = 1000,
        generations                    = 1000,
        max_evaluations                = int(1000 * 1000),
        local_iterations               = 0,
        max_selection_pressure         = 100,
        comparison_factor              = 0,
        brood_size                     = 10,
        tournament_size                = 5,
        btc_bias                       = 0.0,
        n_threads                      = 1,
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
        self.error_metric              = 'r2' if error_metric is None else error_metric
        self.max_length                = 50 if max_length is None else int(max_length)
        self.max_depth                 = 10 if max_depth is None else int(max_depth)
        self.initialization_method     = 'btc' if initialization_method is None else initialization_method
        self.female_selector           = 'tournament' if female_selector is None else female_selector
        self.male_selector             = 'tournament' if male_selector is None else male_selector
        self.population_size           = 1000 if population_size is None else int(population_size)
        self.pool_size                 = 1000 if pool_size is None else int(pool_size)
        self.generations               = 1000 if generations is None else int(generations)
        self.max_evaluations           = 1000000 if max_evaluations is None else int(max_evaluations)
        self.local_iterations          = 0 if local_iterations is None else int(local_iterations)
        self.max_selection_pressure    = 100 if max_selection_pressure is None else int(max_selection_pressure)
        self.comparison_factor         = 0 if comparison_factor is None else comparison_factor
        self.brood_size                = 10 if brood_size is None else int(brood_size)
        self.tournament_size           = 5 if tournament_size is None else tournament_size # todo: set for both parent selectors
        self.btc_bias                  = 0.0 if btc_bias is None else btc_bias
        self.n_threads                 = 1 if n_threads is None else int(n_threads)
        self.random_state              = random.getrandbits(64) if random_state is None else random_state


    def __init_primitive_config(self, allowed_symbols):
        symbols = allowed_symbols.split(',')

        config = _operon.NodeType(0)
        for s in symbols:
            if s == 'add':
                config |= _operon.NodeType.Add
            elif s == 'sub':
                config |= _operon.NodeType.Sub
            elif s == 'mul':
                config |= _operon.NodeType.Mul
            elif s == 'div':
                config |= _operon.NodeType.Div
            elif s == 'exp':
                config |= _operon.NodeType.Exp
            elif s == 'log':
                config |= _operon.NodeType.Log
            elif s == 'sin':
                config |= _operon.NodeType.Sin
            elif s == 'cos':
                config |= _operon.NodeType.Cos
            elif s == 'tan':
                config |= _operon.NodeType.Tan
            elif s == 'sqrt':
                config |= _operon.NodeType.Sqrt
            elif s == 'cbrt':
                config |= _operon.NodeType.Cbrt
            elif s == 'square':
                config |= _operon.NodeType.Square
            elif s == 'constant':
                config |= _operon.NodeType.Constant
            elif s == 'variable':
                config |= _operon.NodeType.Variable
            else:
                raise ValueError('Unknown symbol type {}'.format(s))

        return config


    def __init_creator(self, initialization_method, pset, inputs):
        if initialization_method == 'btc':
            return _operon.BalancedTreeCreator(pset, inputs, self.btc_bias)

        elif initialization_method == 'ptc2':
            return _operon.ProbabilisticTreeCreator(pset, inputs)

        elif initialization_method == 'koza':
            return _operon.GrowTreeCreator(pset, inputs)

        raise ValueError('Unknown initialization method {}'.format(initialization_method))


    def __init_selector(self, selection_method, obj_index=0):
        if selection_method == 'tournament':
            selector = _operon.TournamentSelector(obj_index)
            selector.TournamentSize = self.tournament_size
            return selector

        elif selection_method == 'proportional':
            selector = _operon.ProportionalSelector(obj_index)
            return selector

        elif selection_method == 'random':
            selector = _operon.RandomSelector()
            return selector

        raise ValueError('Unknown selection method {}'.format(selection_method))


    def __init_evaluator(self, error_metric, problem):
        if error_metric == 'r2':
            return _operon.RSquaredEvaluator(problem)

        elif error_metric == 'nmse':
            return _operon.NormalizedMeanSquaredErrorEvaluator(problem)

        raise ValueError('Unknown error metric {}'.format(error_metric))


    def __init_generator(self, generator_name, evaluator, crossover, mutator, female_selector, male_selector):
        if male_selector is None:
            male_selector = female_selector

        if generator_name == 'basic':
            return _operon.BasicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)

        elif generator_name == 'os':
            generator = _operon.OffspringSelectionGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.MaxSelectionPressure = self.max_selection_pressure
            generator.ComparisonFactor = self.comparison_factor
            return generator

        elif generator_name == 'brood':
            generator = _operon.BroodOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        elif generator_name == 'poly':
            generator = _operon.PolygenicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        raise ValueError('Unknown generator method {}'.format(generator_name))


    def __init_reinserter(self, reinserter_name, obj_index=0):
        if reinserter_name == 'replace-worst':
            return _operon.ReplaceWorstReinserter(obj_index)

        elif reinserter_name == 'keep-best':
            return _operon.KeepBestReinserter(obj_index)

        raise ValueError('Unknown reinsertion method {}'.format(reinserter_name))


    def __init_mutation(self, mutation_name, inputs, pset, creator):
        if mutation_name == 'onepoint':
            return _operon.OnePointMutation()

        elif mutation_name == 'changevar':
            return _operon.ChangeVariableMutation(inputs)

        elif mutation_name == 'changefunc':
            return _operon.ChangeFunctionMutation(pset)

        elif mutation_name == 'replacesubtree':
            return _operon.ReplaceSubtreeMutation(creator, self.max_depth, self.max_length)

        elif mutation_name == 'insertsubtree':
            return _operon.InsertSubtreeMutation(creator, self.max_depth, self.max_length)

        elif mutation_name == 'removesubtree':
            return _operon.RemoveSubtreeMutation(pset)

        raise ValueError('Unknown mutation method {}'.format(mutation_name))


    def fit(self, X, y, show_model=False):
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
        
        ds                    = _operon.Dataset(D)
        target                = max(ds.Variables, key=lambda x: x.Index) # last column is the target
        inputs                = _operon.VariableCollection(v for v in ds.Variables if v.Index != target.Index)
        training_range        = _operon.Range(0, ds.Rows)
        test_range            = _operon.Range(ds.Rows-1, ds.Rows) # hackish, because it can't be empty
        problem               = _operon.Problem(ds, inputs, target.Name, training_range, test_range)

        pset                  = _operon.PrimitiveSet()
        pcfg                  = self.__init_primitive_config(self.allowed_symbols)
        pset.SetConfig(pcfg)

        creator               = self.__init_creator(self.initialization_method, pset, inputs)

        evaluator             = self.__init_evaluator(self.error_metric, problem)
        evaluator.Budget      = self.max_evaluations;
        evaluator.LocalOptimizationIterations = self.local_iterations

        female_selector       = self.__init_selector(self.female_selector, 0)
        male_selector         = self.__init_selector(self.male_selector, 0)
        reinserter            = self.__init_reinserter(self.reinserter, 0)
        cx                    = _operon.SubtreeCrossover(self.crossover_internal_probability, self.max_depth, self.max_length)

        mut                   = _operon.MultiMutation()
        mut_list = [] # this list is needed as a placeholder to keep alive the mutation operators objects (since the multi-mutation only stores references)
        for k in self.mutation:
            v = self.mutation[k]
            m = self.__init_mutation(k, inputs, pset, creator)
            mut.Add(m, v)
            mut_list.append(m)

        generator             = self.__init_generator(self.offspring_generator, evaluator, cx, mut, female_selector, male_selector)

        min_arity, max_arity  = pset.FunctionArityLimits()
        initializer           = _operon.UniformInitializer(creator, min_arity+1, self.max_length)

        config                = _operon.GeneticAlgorithmConfig(
                                    generations      = self.generations,
                                    max_evaluations  = self.max_evaluations,
                                    local_iterations = self.local_iterations,
                                    population_size  = self.population_size,
                                    pool_size        = self.pool_size,
                                    p_crossover      = self.crossover_probability,
                                    p_mutation       = self.mutation_probability,
                                    seed             = self.random_state
                                    )

        gp                    = _operon.GeneticProgrammingAlgorithm(problem, config, initializer, generator, reinserter)

        rng                   = _operon.RomuTrio(np.uint64(config.Seed))

        gp.Run(rng, None, self.n_threads)
        comp                  = _operon.SingleObjectiveComparison(0)
        best                  = gp.BestModel(comp)

        y_pred                = _operon.Evaluate(best.Genotype, ds, training_range)
        a, b                  = _operon.FitLeastSquares(y_pred, y)

        # add four nodes at the top of the tree for linear scaling
        nodes                 = best.Genotype.Nodes
        nodes.extend([ _operon.Node.Constant(b), _operon.Node.Mul(), _operon.Node.Constant(a), _operon.Node.Add() ])

        self._model           = _operon.Tree(nodes).UpdateNodes()

        if show_model:
            print(_operon.InfixFormatter.Format(self._model, ds, 12))
            print('internal model r2: ', 1 - best[0])

        self._stats = {
            'model_length':        self._model.Length - 4, # do not count scaling nodes?
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
        n_samples, n_features = X.shape
        ds = _operon.Dataset(X)
        rg = _operon.Range(0, ds.Rows)
        return _operon.Evaluate(self._model, ds, rg)

