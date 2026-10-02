# Team Brain.exe: Reactor Yield Prediction

A physics-informed machine learning model that predicts the overall yield of product B in a continuous-flow, non-isothermal reactor.

## Problem

The reactor runs a series-parallel reaction network:

- **A → B** (rate constant `k1`) is the desired reaction.
- **B → C** (rate constant `k2`) is the side reaction that destroys the product.

Yield is not monotonic in residence time or temperature. Too little time under-converts A, and too much over-converts B into C. The task is to predict `overall_yield` (% of B at the reactor exit) from five operating variables, using only 150 labelled observations.

## Inputs and target

| Column | Role |
|---|---|
| `flow_rate_L_min` | feature |
| `concentration_mol_L` | feature |
| `inlet_temperature_K` | feature |
| `length_m` | feature |
| `jacket_temperature_K` | feature |
| `overall_yield` | target (%) |

## Approach

The notebook [Team_Brain_exe.ipynb](Team_Brain_exe.ipynb) walks through these steps:

1. **EDA.** The target is strongly bimodal. Residence time (`L / Q`) has an inverted-U relationship with yield, which is the signature of an A → B → C series reaction.
2. **Baseline.** A tuned `ExtraTreesRegressor` with engineered features (residence time, 1/T, temperature gap) reaches an RMSE in the high teens.
3. **Physics-informed model.** The reactor is simulated as a sequence of small residence-time steps. Each step includes:
   - relaxation of the fluid temperature toward the jacket temperature,
   - Arrhenius-style `k1` and `k2` with reference temperature `T_ref = T_cross`, the point where `k1 = k2`,
   - non-first-order reaction orders,
   - reaction heat feedback.
4. **Fitting.** The two rate constants are learned with `scipy.optimize.least_squares`. The seven structural parameters are chosen by grid search (972 combinations) scored by 5-fold CV RMSE.
5. **Robustness check.** A wide-range search with all nine parameters free converges to roughly the same neighbourhood.
6. **Evaluation.** The physics-informed model beats the ExtraTrees baseline by about 6× in RMSE, with a CV RMSE of about 3.1. The notebook also includes residual and sensitivity plots.

## Usage

Requirements: Python 3 with `numpy`, `pandas`, `matplotlib`, `scipy` and `scikit-learn`.

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

Place `train_dataset.csv` (5 features and the target) and `test_dataset.csv` (5 features) in the same folder as the notebook, then run it:

```bash
jupyter notebook Team_Brain_exe.ipynb
```

Running all cells writes the predictions to `Brain.exe.csv`.

## Files

- `Team_Brain_exe.ipynb`: the full pipeline, from data loading through to final predictions and evaluation.
- `train_dataset.csv`, `test_dataset.csv`: input data, not included in this repo.
- `Brain.exe.csv`: generated predictions.
