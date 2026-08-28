# Fraud Detection in Transaction Graphs

A reproducible benchmark for identifying fraudulent accounts from a payment network, using three different views of the same data:

- an **autoencoder** that learns normal account behavior without fraud labels;
- an **MLP** that classifies each account from aggregated activity features; and
- **GraphSAGE**, which also learns from the account's neighbors in the transaction graph.

The project is built with PyTorch and PyTorch Geometric. It includes synthetic data generation, leakage-aware preprocessing, training, validation-based threshold selection, cost-sensitive evaluation, automated tests, and multi-seed experiment reporting.

## Results

The main benchmark uses three random seeds on one graph of 10,000 synthetic accounts. Its primary operating point assumes investigators can review only the top-scored 2% of test accounts—30 alerts per run. Values are mean ± standard deviation.

| Model | PR-AUC | Precision @ 2% budget | Recall @ 2% budget | False alerts / 1,000 |
|---|---:|---:|---:|---:|
| Autoencoder | 0.472 ± 0.039 | 0.467 ± 0.000 | 0.350 ± 0.000 | 10.66 ± 0.00 |
| MLP | 0.674 ± 0.038 | **0.633 ± 0.033** | **0.475 ± 0.025** | **7.33 ± 0.67** |
| GraphSAGE | **0.708 ± 0.060** | 0.611 ± 0.051 | 0.458 ± 0.038 | 7.77 ± 1.02 |

![Three-model benchmark comparison](artifacts/credible/summary.png)

GraphSAGE produced the strongest overall ranking quality, but the MLP was slightly better within the strict top-2% review queue. GraphSAGE caught 45.8% of fraud accounts while reviewing 2% of accounts; the MLP caught 47.5%. The difference is smaller than the across-seed variation, so the defensible conclusion is that GraphSAGE is competitive rather than decisively superior.

The original cost-selected operating point is still reported for comparison and reaches 1.000 supervised recall because it assigns a cost of 25 to a missed fraud and 1 to a false alert. It is no longer the headline result. See [RESULTS.md](RESULTS.md) for the complete scorecard and per-seed results.

## What the pipeline does

1. **Generates a payment network** containing synthetic accounts, normal transactions, and fraud rings.
2. **Builds graph and tabular inputs** from the same records. Nodes represent accounts, directed edges represent payments, and node features summarize balances plus incoming and outgoing activity.
3. **Prevents preprocessing leakage** by fitting scalers on the training split only.
4. **Trains three models** against the same data split and evaluation contract.
5. **Selects decision thresholds on validation data**, never on the test set.
6. **Evaluates discrimination and operations** with PR-AUC, cost-selected metrics, and precision/recall under a fixed 2% investigation budget.
7. **Aggregates repeated runs** into CSV, JSON, and PNG artifacts with mean and standard deviation.

## Model comparison

### Autoencoder

The autoencoder is trained only on legitimate accounts. At inference time, unusually high reconstruction error is treated as evidence of fraud. It provides a useful unsupervised baseline, but produces substantially more false alerts in this benchmark.

### MLP

The MLP is the controlled tabular baseline. It sees the same account-level features as GraphSAGE but has no access to graph connectivity. This makes the comparison useful: any GraphSAGE improvement must come from relational context rather than a richer input table.

### GraphSAGE

The GNN classifies accounts after learning representations that aggregate information from their transaction neighbors. The repository also contains optional deeper, wider, residual, directed, and amount-aware variants. Those variants were evaluated and rejected as defaults because they did not improve validation PR-AUC consistently; the simpler two-layer GraphSAGE remained the strongest reliable configuration.

## Quick start

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/yorpl0/fraud-gnn-benchmark.git
cd fraud-gnn-benchmark
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and PyTorch Geometric dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[test]"
pip install git+https://github.com/SantanderAI/gen-fraud-graph.git
```

For the exact environment used to produce the checked-in results:

```bash
pip install -r requirements-lock.txt
```

Run a fast end-to-end smoke test:

```bash
python scripts/run_pipeline.py --config configs/smoke.yaml
```

The pipeline generates data, prepares features, trains all three models, evaluates them, and writes outputs under `artifacts/`.

## Reproduce the main benchmark

Run the three-seed experiment used in the results table:

```bash
python scripts/run_multiseed.py \
  --config configs/default.yaml \
  --seeds 42 43 44 \
  --output-root artifacts/credible
```

To run the stages individually:

```bash
python scripts/generate_data.py --config configs/default.yaml
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train_autoencoder.py --config configs/default.yaml
python scripts/train_mlp.py --config configs/default.yaml
python scripts/train_gnn.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

Run the automated checks with:

```bash
pytest -q
```

## Repository structure

```text
fraud-gnn-benchmark/
├── configs/                 # Default and smoke-test experiment settings
├── scripts/                 # Data, training, evaluation, and orchestration CLIs
├── src/fraud_gnn/
│   ├── data.py              # Feature preparation and graph construction
│   ├── metrics.py           # Metrics, thresholds, and business cost
│   ├── experiments.py       # Repeated-run aggregation and reporting
│   └── models/              # Autoencoder, MLP, and GraphSAGE implementations
├── tests/                   # Data, model, and experiment tests
├── artifacts/credible/      # Published aggregate benchmark outputs
├── RESULTS.md               # Detailed experiment record
└── requirements-lock.txt    # Exact reproducibility environment
```

## Experimental design

- **Split:** stratified train, validation, and test partitions.
- **Scaling:** fit on training data only, then applied to validation and test data.
- **Class imbalance:** supervised models use a positive-class weight computed from the training split.
- **Thresholding:** the validation split selects the operating threshold; test data is used once for final reporting.
- **Repetition:** the credible benchmark reports three seeds rather than a favorable single run.
- **Primary ranking metrics:** PR-AUC and F1, which are more informative than accuracy for imbalanced fraud data.
- **Operational metric:** `25 × false negatives + 1 × false positives`, also normalized per 1,000 accounts.
- **Capacity metric:** rank test accounts by model score, alert only the top 2%, and report recall, precision, and false alerts without using test labels to select a threshold.

## Limitations

This is a research and portfolio project, not a production fraud system.

- The data is synthetic, so the absolute scores should not be generalized to a bank or payment network.
- The current benchmark is transductive: the full graph structure is available while labels are split for training, validation, and testing.
- Fraud behavior and business costs are simplified.
- Three seeds are enough to expose major instability, but not enough for strong statistical claims about the small MLP–GraphSAGE difference.

The most valuable next step is evaluation on a real temporal transaction graph with time-based splits, followed by inductive testing on previously unseen accounts and calibrated cost-sensitive thresholds.

## References

- Hamilton, Ying, and Leskovec, [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), 2017.
- PyTorch Geometric [documentation](https://pytorch-geometric.readthedocs.io/).

## License

Released under the [MIT License](LICENSE).
