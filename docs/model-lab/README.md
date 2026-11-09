# AutoML and SHAP visual lab

Follow the four stages in order. No installation, API key or Python execution is needed to use the browser interface.

## 1. Select a pipeline

Predict whether a synthetic customer leaves. Maximise mean CV precision subject to mean CV recall of at least 0.70. Choose features, maximum tree depth and minimum leaf size. Evaluate both / 2 / 5, neither / 4 / 15 and recency / 4 / 15. Explain your selection using both scores.

The controls choose a pipeline configuration. Fitting learns splits and leaf predictions. All candidates use identical three-fold stratified CV partitions of 900 development cases. The fixed prediction threshold is 0.5. The browser reveals precomputed scores, rather than training on every click.

## 2. Automate the search

Use seed 7. Inspect the initial population and advance one generation. Identify the parents, the recombination point and any mutation. Watch the best eligible score and unique evaluations. Compare random search under the same proposal budget, then repeat with other seeds.

The teaching genetic algorithm has population size four and eight generations, with four children per generation. Two-way tournaments select parents. A one-point crossover combines three configuration genes. With probability 0.7, one gene mutates to a different legal value. Parents and children compete to retain the best four distinct candidates. Eligibility comes first, followed by precision; if both candidates are ineligible, prefer higher recall. Exact ties use candidate ID.

Both methods have 36 proposals including initialisation. Revisits use a cache. Proposal counts do not necessarily equal unique evaluations. This 48-candidate space can be enumerated exhaustively. The animation does not demonstrate a runtime advantage or superiority over random search. Search seeds change proposals only; the score table is fixed.

## 3. Explain with SHAP

Keep M19 fixed and choose customer 1. Start at the baseline and reveal each contribution. Check that baseline plus all contributions equals the predicted probability. Then change the customer while keeping the model fixed.

These are exact empirical interventional Shapley values, computed by enumerating all 32 subsets of five inputs. Each coalition keeps present features at the customer's values and fills absent features from 60 fixed development-background rows. Absent columns are drawn jointly from each row. The model stays fixed. Marginal contributions use standard Shapley subset weights. No tree-path contribution heuristic is used. Positive-class probability is the explained output, so contributions are in probability points. For other explainers the output scale can be log-odds; always check it.

The global chart is mean absolute SHAP over 100 audit cases. This summary discards sign. It is different from permutation importance. With correlated features the hybrid inputs can be unrealistic, and changing the background or feature-dependence assumption changes the explanation question. Interventional SHAP does not by itself establish causal effects.

## 4. Research connection

Compare M19 and M31 with the same customer, background and explanation method. Recency and inactivity are deliberately exact duplicate measurements. These models can give the same prediction while crediting different feature names. At tolerance 0.02, nine of the 48 candidates meet recall and lie within two precision points of the best eligible CV precision.

This is a finite empirical Rashomon set. It is not a confidence interval, the set of all good models, or a guarantee of equivalent future performance. Stability must specify what changes and what stays fixed. Stable explanations can still be wrong about the real world.

## Reproducibility

`build_data.py` recreates the synthetic dataset, scores and explanations. Run `python3 build_data.py` in this folder with numpy, pandas, matplotlib and scikit-learn. `data.js` stores package versions. There are 1500 synthetic independent cases: 900 development, 300 audit and 300 reserved test. The reserved test is unused. The script checks local additivity for 4800 explanations and zero attribution for unused inputs.

The optional original notebook is not required for this revised lab. The lecture PDF is in `lectures/week-07/AutoML_SHAP_and_Explanation_Stability.pdf`.

## Sources

- Olson and Moore (2016), TPOT: https://proceedings.mlr.press/v64/olson_tpot_2016.html
- Lundberg and Lee (2017), SHAP: https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Fisher, Rudin and Dominici (2019), model class reliance: https://jmlr.org/papers/v20/18-760.html
- SHAP documentation: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

## GitHub Pages

Serve the `docs` directory from the `main` branch. The lab is at `model-lab/` under the repository's Pages site. The repository file viewer alone does not execute the lab. Alternatively download this folder and open `index.html`; data are loaded as a local script, with no fetch request or server requirement.
