from math import log10
from typing import List, Tuple
import numpy as np
import ioh
from collections import deque
from scipy.spatial.distance import pdist, squareform

import math
from typing import Any
from typing import cast
from typing import Optional
import warnings
warnings.filterwarnings("error")

class JADEConfig():
    def __init__(
        self,
        population_size: int,
        p: float,
        c: float,
        initial_mu_f: float,
        initial_mu_cr: float,
        archive_size: float,
        clone_distance_threshold: float,
        ls_no_imp_count: int,
        ls_imp_thresh: float,
        ls_best_n_specimens: int,
        ls_max_evals: int,
        cmaes_sigma: float,
        cmaes_step: float,
    ):
        self.population_size = population_size
        self.p = p
        self.c = c
        self.initial_mu_f = initial_mu_f
        self.initial_mu_cr = initial_mu_cr
        self.archive_size = archive_size
        self.clone_distance_threshold = clone_distance_threshold
        self.ls_no_imp_count = ls_no_imp_count
        self.ls_imp_thresh = ls_imp_thresh
        self.ls_best_n_specimens = ls_best_n_specimens
        self.ls_max_evals = ls_max_evals
        self.cmaes_sigma = cmaes_sigma
        self.cmaes_step = cmaes_step

class JADE:
    def __init__(self, budget_factor: int = 2000):
        self.budget_factor: int = budget_factor
        self.__total_evals: int = 0
        self.__eval_archive: dict = {}
        self.counter = 0

    def __initialize_population(self, problem: ioh.problem.RealSingleObjective, rng: np.random.RandomState) -> np.ndarray:
        dim = problem.bounds.lb.size
        
        population = rng.uniform(problem.bounds.lb, problem.bounds.ub, size=(self.config.population_size, dim))
        return population

    def __generate_mutation_vectors_jade(
        self,
        specimens: np.ndarray,
        fitness_scores: np.ndarray,
        rng: np.random.RandomState,
        archive: list,
        mu_F: float,
        p: float
    ) -> Tuple[np.ndarray, List[float]]:

        n, d = specimens.shape
        mutations = np.empty_like(specimens)
        F_list = []
        pbest_indices = fitness_scores[:max(2, int(p * n))]

        all_archive = np.array(archive) if archive else np.empty((0, d))
        combined = np.vstack((specimens, all_archive)) if all_archive.size else specimens

        for i in range(n):
            Fi = -1
            while Fi <= 0:
                Fi = rng.standard_cauchy() * 0.1 + mu_F
            Fi = min(Fi, 1)

            F_list.append(Fi)

            pbest = specimens[rng.choice(pbest_indices)]
            r1_idx = rng.randint(n - 1) 
            if r1_idx >= i:
                r1_idx += 1
            r1 = specimens[r1_idx]

            diff_r1 = np.any(combined != r1, axis=1)
            diff_xi = np.any(combined != specimens[i], axis=1)
            mask = np.logical_and(diff_r1, diff_xi)
            r2_pool = combined[mask]
            r2 = r2_pool[rng.randint(len(r2_pool))] if len(r2_pool) > 0 else r1

            mutation = specimens[i] + Fi * (pbest - specimens[i]) + Fi * (r1 - r2)
            mutation = np.clip(mutation, self.__problem.bounds.lb, self.__problem.bounds.ub)

            mutations[i] = mutation

        return mutations, F_list

    def __generate_trial_vectors(self, specimens: np.ndarray, mutations: np.ndarray, rng: np.random.RandomState, mu_CR: float) -> Tuple[np.ndarray, List[float]]:
        n, d = specimens.shape
        CR_list = []
        trials = np.empty_like(specimens)
        for i in range(n):
            CRi = np.clip(rng.normal(mu_CR, 0.1), 0, 1)
            CR_list.append(CRi)
            cross_points = rng.rand(d) < CR_list[i]
            if not np.any(cross_points):
                cross_points[rng.randint(0, d)] = True
            trials[i] = np.where(cross_points, mutations[i], specimens[i])
        return trials, CR_list
    
    def __cmaes(self, x, rng:np.random.RandomState, seed, step, sigma):
        step = step
        inner_lists = [[xi - step, xi + step] for xi in x]
        bounds = np.array(inner_lists, dtype=float)

        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

        mean = lower_bounds + (rng.rand(len(upper_bounds)) * (upper_bounds - lower_bounds))
        
        sigma = step * sigma / 5
        optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=seed)

        best = (x,self.__eval(x))

        optimizer.tell([(x, best[1])]*optimizer.population_size)
        evals = 0
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = self.__eval(x)
                evals += 1
                if value < best[1]:
                    best = (x,value)
                if self.__total_evals >= self._budget:
                    break 
                solutions.append((x, value))
            
            if self.__total_evals >= self._budget:
                break 
            optimizer.tell(solutions)

            if optimizer.should_stop():
                break
        return best

    def __eval(self, specimen: np.ndarray):
        spec_hash = tuple(specimen)
        if spec_hash in self.__eval_archive:
            return self.__eval_archive[spec_hash]
        else:
            if self.__total_evals >= self._budget:
                return 10000
            fitness = self.__problem(specimen)
            self.__eval_archive[spec_hash] = fitness
            self.__total_evals += 1
            return fitness 

    def __call__(self, problem: ioh.problem.RealSingleObjective, seed: int) -> None:
        dim = problem.bounds.lb.size
        self.config= JADEConfig(5*dim, 0.1, 0.1, 0.5, 0.8, 0.01*7*dim, #general
                        0.000000001, # clone
                        50, 0.1, 1, 60, 1.367, 0.03665241,# ls
        )
        rng = np.random.RandomState(seed)
        self._budget = self.budget_factor * problem.meta_data.n_variables
        specimens = self.__initialize_population(problem, rng)
        best = None
        self.__problem:ioh.problem.RealSingleObjective = problem
        self.__total_evals=0

        mu_F = self.config.initial_mu_f
        mu_CR = self.config.initial_mu_cr
        archive = []

        fitness_scores = np.apply_along_axis(self.__eval, 1, specimens)


        gen = 0

        last_n_best = deque(maxlen=self.config.ls_no_imp_count)
        while self.__total_evals < self._budget:
            if gen > 5000:
                break

            S_F = []
            S_CR = []
            sorted_fitness_indices = np.argsort(fitness_scores)
            last_n_best.append(fitness_scores[sorted_fitness_indices[0]])

            current_best = sorted_fitness_indices[0]
            if not best or best[1] > fitness_scores[current_best]:
                best = tuple([specimens[current_best], fitness_scores[current_best]])

            base = log10(last_n_best[0])
            improvements = [abs(base - log10(val)) for val in list(last_n_best)[1:]]

            if gen > self.config.ls_no_imp_count and all(imp < self.config.ls_imp_thresh for imp in improvements):
                explored = 0
                while explored < self.config.ls_best_n_specimens:
                    score = self.__cmaes(specimens[sorted_fitness_indices[explored]], rng, seed, self.config.cmaes_step, self.config.cmaes_sigma)
                    if best[1] > score[1]:
                        best = tuple([specimens[sorted_fitness_indices[explored]], score[1]])
                    explored+=1


            mutations, F_list = self.__generate_mutation_vectors_jade(specimens, sorted_fitness_indices, rng, archive, mu_F, self.config.p)
            trials, CR_list = self.__generate_trial_vectors(specimens,mutations, rng, mu_CR)

            fitness_scores_trials = np.apply_along_axis(self.__eval, 1, trials)
            if self.__total_evals >= self._budget:
                break

            for i in range(len(specimens)):
                if fitness_scores_trials[i] < fitness_scores[i]:
                    if len(archive) >= self.config.archive_size:
                        archive.pop(rng.randint(len(archive)))
                    archive.append(specimens[i].copy())
                    specimens[i] = trials[i]
                    fitness_scores[i] = fitness_scores_trials[i]
                    S_CR.append(CR_list[i])
                    S_F.append(F_list[i])

            if S_CR:
                mu_CR = (1 - self.config.c) * mu_CR + self.config.c * np.mean(S_CR)
            if S_F:
                meanL = np.sum(np.square(S_F)) / np.sum(S_F)
                mu_F = (1 - self.config.c) * mu_F + self.config.c * meanL
                

            distances = squareform(pdist(specimens, metric='euclidean'))
            np.fill_diagonal(distances, np.inf)
            is_clone = np.any(distances < self.config.clone_distance_threshold, axis=1)

            if np.any(is_clone):
                num_clones = np.sum(is_clone)
                new_randoms = rng.uniform(problem.bounds.lb, problem.bounds.ub, size=(num_clones, specimens.shape[1]))
                specimens[is_clone] = new_randoms

                fitness_scores[is_clone] = np.apply_along_axis(self.__eval, 1, specimens[is_clone])

                if self.__total_evals >= self._budget:
                    break
            gen += 1
        
        return best
    
### vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
### CMA-ES library:
# @misc{nomura2024cmaessimplepractical,
#       title={cmaes : A Simple yet Practical Python Library for CMA-ES}, 
#       author={Masahiro Nomura and Masashi Shibata},
#       year={2024},
#       eprint={2402.01373},
#       archivePrefix={arXiv},
#       primaryClass={cs.NE},
#       url={https://arxiv.org/abs/2402.01373}, 
# }


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMA:
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import CMA

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).

        lr_adapt:
            Flag for learning rate adaptation (optional; default=False)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        lr_adapt: bool = False,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim**2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean.copy()

        if cov is None:
            self._C = np.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # for learning rate adaptation
        self._lr_adapt = lr_adapt
        self._alpha = 1.4
        self._beta_mean = 0.1
        self._beta_Sigma = 0.03
        self._gamma = 0.1
        self._Emean = np.zeros([self._n_dim, 1])
        self._ESigma = np.zeros([self._n_dim * self._n_dim, 1])
        self._Vmean = 0.0
        self._VSigma = 0.0
        self._eta_mean = 1.0
        self._eta_Sigma = 1.0

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def __getstate__(self) -> dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            if name == "_C":
                sym1d = _compress_symmetric(self._C)
                attrs["_c_1d"] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    @property
    def mean(self) -> np.ndarray:
        """Mean Vector"""
        return self._mean

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        # keep old values for learning rate adaptation
        if self._lr_adapt:
            old_mean = np.copy(self._mean)
            old_sigma = self._sigma
            old_Sigma = self._sigma**2 * self._C
            old_invsqrtC = B @ np.diag(1 / D) @ B.T
        else:
            old_mean, old_sigma, old_Sigma, old_invsqrtC = None, None, None, None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = cast(
            np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T)
        )  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        try:
            self._sigma *= np.exp(
                (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
            )
        except RuntimeWarning:
            self._sigma = _SIGMA_MAX
        except Exception:
            self._sigma = _SIGMA_MAX

        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )

        # Learning rate adaptation: https://arxiv.org/abs/2304.03473
        if self._lr_adapt:
            assert isinstance(old_mean, np.ndarray)
            assert isinstance(old_sigma, (int, float))
            assert isinstance(old_Sigma, np.ndarray)
            assert isinstance(old_invsqrtC, np.ndarray)
            self._lr_adaptation(old_mean, old_sigma, old_Sigma, old_invsqrtC)

    def _lr_adaptation(
        self,
        old_mean: np.ndarray,
        old_sigma: float,
        old_Sigma: np.ndarray,
        old_invsqrtC: np.ndarray,
    ) -> None:
        # calculate one-step difference of the parameters
        Deltamean = (self._mean - old_mean).reshape([self._n_dim, 1])
        Sigma = (self._sigma**2) * self._C
        # note that we use here matrix representation instead of vec one
        DeltaSigma = Sigma - old_Sigma

        # local coordinate
        old_inv_sqrtSigma = old_invsqrtC / old_sigma
        locDeltamean = old_inv_sqrtSigma.dot(Deltamean)
        locDeltaSigma = (
            old_inv_sqrtSigma.dot(DeltaSigma.dot(old_inv_sqrtSigma))
        ).reshape(self.dim * self.dim, 1) / np.sqrt(2)

        # moving average E and V
        self._Emean = (
            1 - self._beta_mean
        ) * self._Emean + self._beta_mean * locDeltamean
        self._ESigma = (
            1 - self._beta_Sigma
        ) * self._ESigma + self._beta_Sigma * locDeltaSigma
        self._Vmean = (1 - self._beta_mean) * self._Vmean + self._beta_mean * (
            float(np.linalg.norm(locDeltamean)) ** 2
        )
        self._VSigma = (1 - self._beta_Sigma) * self._VSigma + self._beta_Sigma * (
            float(np.linalg.norm(locDeltaSigma)) ** 2
        )

        # estimate SNR
        sqnormEmean = np.linalg.norm(self._Emean) ** 2
        hatSNRmean = (
            sqnormEmean - (self._beta_mean / (2 - self._beta_mean)) * self._Vmean
        ) / (self._Vmean - sqnormEmean)
        sqnormESigma = np.linalg.norm(self._ESigma) ** 2
        hatSNRSigma = (
            sqnormESigma - (self._beta_Sigma / (2 - self._beta_Sigma)) * self._VSigma
        ) / (self._VSigma - sqnormESigma)

        # update learning rate
        before_eta_mean = self._eta_mean
        relativeSNRmean = np.clip(
            (hatSNRmean / self._alpha / self._eta_mean) - 1, -1, 1
        )
        self._eta_mean = self._eta_mean * np.exp(
            min(self._gamma * self._eta_mean, self._beta_mean) * relativeSNRmean
        )
        relativeSNRSigma = np.clip(
            (hatSNRSigma / self._alpha / self._eta_Sigma) - 1, -1, 1
        )
        self._eta_Sigma = self._eta_Sigma * np.exp(
            min(self._gamma * self._eta_Sigma, self._beta_Sigma) * relativeSNRSigma
        )
        # cap
        self._eta_mean = min(self._eta_mean, 1.0)
        self._eta_Sigma = min(self._eta_Sigma, 1.0)

        # update parameters
        self._mean = old_mean + self._eta_mean * Deltamean.reshape(self._n_dim)
        Sigma = old_Sigma + self._eta_Sigma * DeltaSigma

        # decompose Sigma to sigma and C
        eigs, _ = np.linalg.eigh(Sigma)
        logeigsum = sum([np.log(e) for e in eigs])
        self._sigma = np.exp(logeigsum / 2.0 / self._n_dim)
        self._sigma = min(self._sigma, _SIGMA_MAX)
        self._C = Sigma / (self._sigma**2)

        # step-size correction
        self._sigma *= before_eta_mean / self._eta_mean

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        if np.any(D == 0):
            return True
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False


def _is_valid_bounds(bounds: Optional[np.ndarray], mean: np.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        return False
    if not np.all(bounds[:, 0] <= mean):
        return False
    if not np.all(mean <= bounds[:, 1]):
        return False
    return True


def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = np.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start : start + n - i] = sym2d[i][i:]  # noqa: E203
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out
