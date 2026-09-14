---
layout: default
---

# Operations Research

## MILP

* mixed-integer linear programming (MILP)

### CBC

* CBC (COIN-OR Branch and Cut): an open-source mixed-integer programming (MIP) solver.
* The key idea is that LPs are relatively easy to solve, while integer constraints make the problem much harder. MILP = LP solver + branch-and-bound + cutting planes + heuristics
  * LP relaxation: At each node, temporarily ignore integer constraints and solve the resulting LP to understand the node's best possible objective.
  * Bounding: Use the optimal LP-relaxation objective as the node's bound — upper bound for maximization, lower bound for minimization.
  * Pruning: If the node is infeasible (cannot be solved without violating any constraints), or the node's bound cannot beat the incumbent, discard the node and its entire subtree.
  * Cutting planes: Add valid inequalities to tighten the LP relaxation without removing any integer-feasible solutions, then re-solve the LP and obtain a tighter bound.
  * Heuristics: Try to construct an integer-feasible solution (incumbent) from the current information (e.g. Feasibility Pump).
  * Branching: If the LP solution is fractional and the node cannot be pruned, choose a fractional integer variable and split the problem into child nodes, each inheriting the parent's constraints.
  * Search / management: CBC repeatedly decides whether to cut, run a heuristic, branch, explore another node, or prune, until it proves that no unexplored node can beat the incumbent (best remaining node Upper Bound≤incumbent).
  * E.g.
                  x = 3.7
                /       \
             x ≤ 3      x ≥ 4
              /           \
          y = 5.2        y = 4.6
           / \             / \
        y≤5 y≥6         y≤4 y≥5
          /                |
    x = 2.4 y = 5        ...
      /      \
  x ≤ 2     x ≥ 3
    |         |
  ...     x = 3 y = 5 a valid integar solution/or not.

### PuLP

* PuLP: a Python library for formulating and solving optimization problems, especially linear programming (LP) and mixed-integer linear programming (MILP) problems. Think of it as a modeling interface: you describe your optimization problem in Python, and PuLP sends it to a solver such as CBC to actually find the solution.

```python
import pulp

# Create the problem
problem = pulp.LpProblem("Example", pulp.LpMaximize)

# Create variables
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

# Objective function
problem += 3*x + 2*y

# Constraints
problem += x + y <= 10
problem += 2*x + y <= 15

# Solve
problem.solve()

print(x.value())
print(y.value())
```

## LP

* linear programming (LP)

### Simplex

* The simplex algorithm is a classic algorithm for solving linear programming (LP) problems. 
* A fundamental property of linear programming is: If an optimal solution exists, at least one optimal solution occurs at a corner (vertex) of the feasible region. Simplex = walk from one feasible vertex to another until you reach the optimum.