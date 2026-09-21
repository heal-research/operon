# Search and optimization

![Population search loop](../assets/diagrams/search-optimization.svg){ .diagram }

## Search loop

Population algorithms prepare selectors and evaluators, generate offspring through crossover and mutation, optionally fit coefficients, score candidates, then reinsert them into the next parent population. NSGA-II retains a multi-objective front; GP uses its configured comparison and reinsertion strategy.

Coefficient fitting is local search on a tree's numeric coefficients. It improves a candidate's score without changing the surrounding population contract.
