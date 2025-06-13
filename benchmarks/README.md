# Running benchmarks 

We're using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) to assess performance. Benchmarks can be run with 

```
pytest benchmarks/ --cutest
```

and will be skipped otherwise. (`pytest --cutest` would also work, but run the whole test suite as well.) 
With the option 

```
pytest benchmarks/ --cutest --benchmark-save=<file_path>
```

benchmark results can be saved in a .json file. Additional custom metrics (e.g. the number of iterations or the quality of the solution) can be configured in the benchmark test functions. 

This will just run benchmarks on our own solvers - you can use saved results to compare performance to previous versions or different commits. Comparing against the last saved run is enabled with `pytest --benchmark-compare`, but specific iDs of previous runs may also be specified. (Consult the [documentation](https://pytest-benchmark.readthedocs.io/en/latest/) for more options.)

**If you want to dive a little deeper**:
Saved results include the commit, branch, version, and an exact timestamp by default.
Note that benchmarks are run with `throw=False` enabled, since otherwise no result is written in the json file, but we do want to know if we failed to solve a problem.
This means that benchmark results need to be filtered for successful solves during analysis, which is done in TODO.

## Provenance of benchmark problems

We're benchmarking on CUTEst through [sif2jax](https://github.com/johannahaffner/sif2jax), a port of CUTEst problems from the original Fortran instructions to JAX.