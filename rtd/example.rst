=======
Example
=======

.. role:: bash(code)
   :language: cpp 

This example shows how to do symbolic regression using GP. The impatient can go directly to the `full code example <https://github.com/foolnotion/operon/blob/master/examples/gp.cpp>`_ on github. We assume *some* familiarity with GP concepts and terminology.

We solve a synthetic benchmark problem, namely the :math:`\textit{Poly-10}`:

.. math::
    
    F(\mathbf{x}) = x_1 x_2 + x_3 x_4 + x_5 x_6 + x_1 x_7 x_9 + x_3 x_6 x_{10}

This problem consists of 500 datapoints which we'll split equally between our *training* and *test* data. 

First, let's load the data from csv into a `Dataset` and set the data partitions:

.. code-block:: cpp

    Dataset ds("../data/Poly-10.csv", /* csv has header */ true);
    Range trainingRange { 0, ds.Rows() / 2 };
    Range testRange     { ds.Rows() / 2, ds.Rows() };

By convention, the program will use all the dataset columns (except for the target column) as features. The user is responsible for preprocessing the data prior to the modeling step.

Next, we define the optimization target and create a ``Problem``:

.. code-block:: cpp

    const std::string target = "Y";
    Problem problem(ds, ds.Variables(), target, trainingRange, testRange);
    problem.GetGrammar().SetConfig(Grammar::Arithmetic);

In the snippet above, :code:`problem.GetGrammar().SetConfig(Grammar::Arithmetic)` configures the problem to use an arithmetic grammar, consisting of the symbols :math:`+,-,\times,\div`. The ``Grammar`` class keeps track of the allowed symbols and their initial frequencies (taken into account when the population is initialized).  

In what follows, we define the *genetic operators*. Although the code is a bit verbose, it's purpose should be clear. 

Crossover and mutation 
    The so-called recombination operators generate offspring individuals by combining genes from the parents (crossover) and adding random perturbations (mutation). 

    - The *crossover* operator takes the tree *depth* and *length* limits and generates offspring that do not exceed them. It can be further parameterized by an *internal node bias* parameter which controls the probability of selecting an internal node (eg., a function node) as a cut point.

    - The *mutation* operator can apply different kind of perturbations to a tree individual: *point mutation* changes a leaf node's coefficient (eg., a variable node's *weight* or a constant node's *value*), *change variable* mutation changes a variable with another one from the dataset (eg., :math:`x_1 \to x_2`) and the *change function* mutation does the same thing to function symbols.

.. code:: cpp

    // set up crossover and mutation
    double internalNodeBias = 0.9;
    size_t maxTreeDepth  = 10;
    size_t maxTreeLength = 50;
    SubtreeCrossover crossover { internalNodeBias, maxTreeDepth, maxTreeLength };
    MultiMutation mutation;
    OnePointMutation onePoint;
    ChangeVariableMutation changeVar { problem.InputVariables() };
    ChangeFunctionMutation changeFunc { problem.GetGrammar() };
    mutation.Add(onePoint, 1.0);
    mutation.Add(changeVar, 1.0);
    mutation.Add(changeFunc, 1.0);


Selector
    The selection operator samples the distribution of fitness values in the population and picks parent individuals for taking part in recombination. *Operon* supports specifying different selection methods for the two parents (typically called *male* and *female* or *root* and *non-root* parents).
    We tell the selector how to compare individuals by providing a lambda function to its constructor:

.. code-block:: cpp

    // our lambda function simply compares the fitness of the individuals
    auto comp = [](Individual const& lhs, Individual const& rhs) { 
        return lhs[0] < rhs[0]; 
    };
    // set up the selector
    TournamentSelector selector(comp);
    selector.TournamentSize(5); 

Evaluator
    This operator is responsible for calculating fitness and is alloted a fixed evaluation budget at the beginning of the run. The *evaluator* is also capable of performing nonlinear least-squares fitting of model parameters if the *local optimization iterations* parameter is set to a value greater than zero. 

.. code-block:: cpp

    // set up the evaluator 
    RSquaredEvaluator evaluator(problem);
    evaluator.LocalOptimizationIterations(config.Iterations);
    evaluator.Budget(config.Evaluations);

Reinserter
    The reinsertion operator merges the pool of *recombinants* (new offspring) back into the population. This can be a simple replacement or a more sophisticated strategy (eg., keep the best individuals among the parents and offspring). Like the selector, the reinserter requires a lambda to specify how it should compare individuals. 

.. code-block:: cpp

    ReplaceWorstReinserter<> reinserter(comp);


Offspring generator 
    Implements a strategy for producing new offspring. This can be plain recombination (eg., crossover + mutation) or more elaborate logic like acceptance criteria for offspring or brood selection. In general, this operation may *fail* (returning a *maybe* type) and should be handled by the algorithm designer.

.. code-block:: cpp

    // the generator makes use of the other operators to generate offspring and assign fitness
    // the selector is passed twice, once for the male parent, once for the female parent.
    BasicOffspringGenerator generator(evaluator, crossover, mutation, selector, selector);

Tree creator
    The tree creator initializes random trees of any target length. The length is sampled from a uniform distribution :math:`U[1, \textit{maxTreeLength}]`. Maximum depth is fixed by the :math:`\textit{maxTreeDepth}` parameter. 

.. code-block:: cpp

    // set up the solution creator 
    std::uniform_int_distribution<size_t> treeSizeDistribution(1, maxTreeLength);
    BalancedTreeCreator creator { treeSizeDistribution, maxTreeDepth, maxTreeLength };

Finally, we can configure the genetic algorithm and run it. A callback function can be provided to the algorithm in order to report progress at the end of each generation.

.. code:: cpp

    GeneticAlgorithmConfig config;
    config.Generations          = 100;
    config.PopulationSize       = 1000;
    config.PoolSize             = 1000;
    config.Evaluations          = 1000000;
    config.Iterations           = 0;
    config.CrossoverProbability = 1.0;
    config.MutationProbability  = 0.25;
    config.Seed                 = 42;

    // set up a genetic programming algorithm
    GeneticProgrammingAlgorithm gp(problem, config, creator, generator, reinserter); 

    int generation = 0;
    auto report = [&] { fmt::print("{}\n", ++generation); };
    Random random(config.Seed);
    gp.Run(random, report);
