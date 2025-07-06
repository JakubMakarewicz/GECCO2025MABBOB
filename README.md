# GECCO2025MABBOB
This is a submission for the 2d and 5d categories for https://iohprofiler.github.io/competitions/mabbob25

The proposed method is a JADE, CMA-ES hybrid

The cma-es code is part of the library: 

@misc{nomura2024cmaessimplepractical,
      title={cmaes : A Simple yet Practical Python Library for CMA-ES}, 
      author={Masahiro Nomura and Masashi Shibata},
      year={2024},
      eprint={2402.01373},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2402.01373}, 
}


The method has a memory of evaluations, so every run needs to be on a new object

```python
for dim in [2,5]: 
    for idx in tqdm(range(1000)):
        for seed in range(10):
            f_new = ioh.problem.ManyAffine(xopt = np.array(opt_locs.iloc[idx])[:dim], 
                                        weights = np.array(weights.iloc[idx]),
                                        instances = np.array(iids.iloc[idx], dtype=int), 
                                        n_variables = dim)
            f_new.set_instance(idx)
            optimizer = JADE()
            optimizer(f_new, seed)
```
