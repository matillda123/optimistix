# Running benchmarks 

This folder contains a larger collection of tough problems. These are  useful to test performance of new solvers and new features, and to catch any regressions early.  

Benchmarks can be run with 

```
pytest benchmarks/ --cutest
```

and will be skipped otherwise. (`pytest --cutest` would also work, but run the whole test suite as well.) 
With the option 

```
pytest benchmarks/ --cutest --benchmark-save=<file_path>
```

benchmark results can be saved in a .json file. Additional custom metrics (e.g. the number of iterations or the quality of the solution) can be configured in the benchmark test functions. 

This will just run benchmarks on our own solvers - you can use saved results to compare performance to previous versions or different commits. 
Saved results include the commit, branch, version, and an exact timestamp by default.

## Provenance of benchmark problems

We're benchmarking on CUTEst through [sif2jax](https://github.com/johannahaffner/sif2jax), a port of CUTEst problems from the original Fortran instructions to JAX.