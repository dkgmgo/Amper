# AmPer

Graph surrogates under a distance filtration.

Load an AS-level graph, fit a Gamma mixture on its edge distances, split the edges into layers, rebuild the graph layer by layer, compare persistent homology.

## Modules

| file | what it does |
|---|---|
| `datasets.py` | graphml loading, distance validation, graph discovery and identity |
| `mixture.py` | K-component Gamma mixture, EM with BIC model selection |
| `layers.py` | component ordering, per-edge layer assignment, layer diagnostics |
| `builder.py` | the surrogate generators (`er`, `er_strat`, `ws_layer`, `er_hier`, `ws_hier`) |
| `topology.py` | filtration, persistence diagrams, diagram and Betti-curve distances |
| `dashboard.py` | plotting |
| `main.py` | orchestration and CSV output |

`GENERATORS.md` describes each generator precisely. `CLAUDE.md` records what has been
measured, what has been refuted, and which numbers not to trust.

Flat layout; `conftest.py` makes the root importable from `tests/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run

```bash
python main.py
```

## Experiments

All are standalone and write under `data/out/`. They take optional graph keys;
with none, they run every graph in `data/in/`.

```bash
python experiments/gen_results.py [KEY ...]
python experiments/permutation_control.py [KEY ...]
```

Start to finish on one graph:

```bash
python experiments/gen_results.py CN
python experiments/permutation_control.py CN
```

Drop the `CN` to run the whole corpus.
