# POP + Load Balancing

## Dependencies
Requires Java 11 and an installation of CPLEX  v12.10.0

## Running Experiments

To compile, run:

    mvn package

To test without POP:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -benchmark base

To test with POP:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -numSplits SPLITS -benchmark split

To test with the heuristic:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -benchmark heuristic

To run experiment shown in Figure 13:

    ./figure13.sh


## Python implementation details
Converted code to use cvxpy interface. Ensure that `cvxpy > 1.6.0` so we can use `HIGHS` solver. Run `pip install highspy` to get the HIGHS solver. Check if `HIGHS` is listed under the installed solvers by running `python -c "import cvxpy as cp; print(cp.installed_solvers())"`.
`IMPORTANT`: Make the following change in `reductions/solvers/qp_solvers/highs_qpif.py` to ensure best warm-start behavior for the `base-lp-relaxed` implementation.
```python
# Add this line after Line 199
# solver.setSolution(old_result["solution"]) --> this must already exist
                solver.setBasis(old_result["basis"])
```
