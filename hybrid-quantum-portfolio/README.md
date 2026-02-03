# Hybrid AI & Quantum-Inspired Portfolio Optimization

An end-to-end framework for portfolio construction combining
classical risk modeling, machine learning, and quantum-inspired
optimization techniques under realistic trading constraints.

## Project Overview

This project implements a full-stack portfolio optimization pipeline that
demonstrates how **hybrid AI and quantum-inspired optimization** can outperform
classical heuristics under realistic financial constraints.

The framework combines:

- Classical financial risk modeling
- Scenario-aware covariance estimation
- Unsupervised machine learning for market regime detection
- Constraint-aware optimization (cardinality, turnover, transaction costs)
- Quantum-inspired solvers formulated via QUBO

All methods are evaluated on the **same data, asset universe, and constraints**
to ensure a fair and rigorous comparison.

## Key Features & Contributions

- **Robust Data Pipeline**  
  Automated data ingestion, cleaning, and feature engineering to ensure
  reproducible and auditable financial experiments.

- **Scenario-Aware Risk Modeling**  
  Rolling covariance estimation with base, stress, and calm risk scenarios
  to capture changing market conditions.

- **AI-Based Market Regime Detection**  
  Unsupervised learning on covariance structure to classify market regimes
  into low-, mid-, and high-volatility states.

- **Classical Constrained Portfolio Construction**  
  Baseline greedy optimization incorporating explicit cardinality,
  turnover, and transaction cost constraints.

- **QUBO Formulation for Portfolio Selection**  
  Binary asset selection formulated as a Quadratic Unconstrained Binary
  Optimization (QUBO) problem.

- **Quantum-Inspired Optimization Algorithms**  
  Implementation and comparison of Simulated Annealing and
  Tensor-Network–based Ising solvers.

- **Rigorous Evaluation Framework**  
  Portfolio risk evaluated directly from solver-generated binary
  selections to ensure fair and methodologically sound comparisons.


## Problem Definition and Constraints

The objective of this project is to construct a **monthly rebalanced,
long-only equity portfolio** that minimizes portfolio risk under realistic
trading and operational constraints.

### Portfolio Characteristics

- Asset universe size: 25 equities
- Portfolio type: Long-only
- Rebalancing frequency: Monthly
- Lookback window: Rolling 60 trading days
- Evaluation metric: Annualized portfolio volatility

### Constraints

The optimization problem incorporates the following constraints:

- **Cardinality constraint**  
  The number of selected assets is limited to a maximum of 10.

- **Turnover constraint**  
  Portfolio turnover between consecutive rebalancing periods is constrained
  to reduce excessive trading.

- **Transaction costs**  
  A proportional transaction cost model of 10 basis points is applied
  to all portfolio changes.

- **No short selling**  
  All asset weights are non-negative.

These constraints significantly increase the combinatorial complexity
of the optimization problem, motivating the use of quantum-inspired
optimization techniques.

## Project Pipeline

The project is organized as a sequential pipeline, where each stage builds
systematically on the outputs of the previous stage.

### Stage 1 — Data Ingestion and Baseline Construction
- Download historical equity price data
- Clean and align time series
- Compute daily returns and rolling volatility
- Construct an equal-weight baseline portfolio
- Persist raw and processed datasets for reproducibility

### Stage 2 — Scenario-Aware Risk Modeling
- Estimate rolling covariance matrices using a 60-day window
- Generate stress, base, and calm risk scenarios
- Define monthly rebalance dates
- Provide a unified risk access API

### Stage 3 — Classical Portfolio Optimization
- Implement a greedy low-volatility asset selection heuristic
- Enforce cardinality, turnover, and transaction cost constraints
- Perform time-consistent monthly rebalancing
- Store classical optimization results

### Stage 4 — Market Regime Detection (Machine Learning)
- Extract volatility and correlation-based features from covariance matrices
- Apply unsupervised clustering (KMeans)
- Classify market regimes into low-, mid-, and high-volatility states
- Persist regime labels for downstream optimization

### Stage 5 — QUBO Formulation and Quantum-Inspired Optimization
- Define binary decision variables for asset selection
- Construct a regime-aware QUBO objective function
- Encode risk, turnover, transaction cost, and cardinality penalties
- Solve the QUBO using Simulated Annealing
- Solve the QUBO using a Tensor-Network–based Ising solver
- Maintain state-dependent portfolio evolution across rebalancing dates

### Stage 6 — Evaluation and Analysis
- Compute portfolio risk from actual solver-generated selections
- Compare classical and quantum-inspired methods under identical conditions
- Analyze regime-conditioned performance
- Generate tables, figures, and executive summaries


## Results Summary and Key Findings

The performance of classical and quantum-inspired optimization techniques
is evaluated under identical market data, constraints, and rebalancing
rules to ensure methodological rigor.

### Portfolio Risk Comparison

Across the full evaluation period:

- The classical constrained baseline exhibits the highest average
  annualized portfolio volatility.
- Simulated Annealing consistently produces lower-risk portfolios by
  solving the full combinatorial selection problem.
- The Tensor-Network–based solver achieves the lowest average volatility
  and the most stable risk profile over time.

### Average Annualized Portfolio Volatility

| Optimization Method | Average Volatility | Relative Risk Reduction |
|--------------------|-------------------|-------------------------|
| Classical Baseline | ~0.16             | —                       |
| Simulated Annealing | ~0.14            | ~12–15%                 |
| Tensor Network     | ~0.12             | ~18–22%                 |

### Key Observations

- Risk reductions arise purely from improved optimization, as all inputs
  and constraints are held constant.
- Quantum-inspired solvers handle cardinality and turnover constraints
  more effectively than classical heuristics.
- Tensor-Network optimization converges to lower-energy and more stable
  solutions than Simulated Annealing.
- Machine learning–based market regime classification enhances solver
  performance by adapting risk sensitivity across market conditions.

These results demonstrate that quantum-inspired optimization delivers
practical, measurable benefits for constrained portfolio construction.


## How to Run the Project

The project is organized as a sequential pipeline. Each stage depends on
artifacts generated by the previous stage.

---

### 1. Environment Setup

Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

```

Install required dependencies:
```
pip install numpy pandas scikit-learn matplotlib seaborn yfinance tqdm
```


## 2. Stage 1 — Data Ingestion and Baseline

Run the data preparation script to download prices and compute baseline statistics:

```
python stage_1_data_pipeline.py
```

**Outputs:**

- `data/raw/prices.csv`
- `data/processed/returns.csv`
- `data/processed/volatility.csv`
- `data/processed/correlations.pkl`


## 3. Stage 2 — Scenario-Aware Risk Modeling

Generate rolling covariance matrices and risk scenarios:

```
python stage_2_risk_modeling.py
```

**Outputs:**

- `data/processed/covariance_matrices.pkl`

## 4. Stage 3 — Classical Portfolio Optimization

Run the classical constrained optimization baseline:

```
python stage_3_classical_optimization.py
```

**Outputs:**

- `experiments/classical/results.csv`

## 5. Stage 4 — Market Regime Detection (Machine Learning)

Detect market regimes using unsupervised learning:

```
python stage_4_market_regimes.py
```

**Outputs:**

- `data/processed/market_regimes.csv`

## 6. Stage 5 — Quantum-Inspired Optimization

Solve the QUBO using quantum-inspired solvers:

```
python stage_5_quantum_sa.py
python stage_5_quantum_tn.py
```

**Outputs:**

- `experiments/quantum_sa/results.csv`
- `experiments/quantum_tn/results.csv`
- Pickled solver outputs for downstream analysis

## 7. Stage 6 — Evaluation and Visualization

Generate tables, figures, and summary statistics:

```
python stage_6_evaluation.py
```

**Outputs:**

- Tables and figures saved in the `results/` directory
- Portfolio volatility comparisons
- Regime-conditioned performance analysis

## Dependencies and Requirements

This project was developed and tested using the following environment:

- Python 3.9 or later
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- yfinance
- tqdm

It is recommended to use a virtual environment to ensure reproducibility.

---

## Future Work

This framework is designed to be extensible. Potential future directions
include:

- Integration with quantum hardware backends (e.g., QAOA or quantum annealers)
- Adaptive hyperparameter tuning for QUBO penalty weights
- Multi-objective optimization incorporating expected returns and risk
- Expansion to larger asset universes and multi-asset portfolios
- ESG-aware and factor-based constraint extensions
- Real-time regime detection and online portfolio rebalancing

---

## License and Citation

This project is released for academic and research purposes.

If you use or build upon this work, please cite it as:

```text
Hybrid AI and Quantum-Inspired Portfolio Optimization,
Author(s): J K Pawan Kumar,
Year: 2026
