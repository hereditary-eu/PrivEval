# PrivEval: a tool for interactive evaluation of privacy metrics in synthetic data generation

PrivEval, a tool for assisting users in evaluating the privacy properties of a synthetic dataset.
Here, the user can explore how privacy is estimated through privacy metrics as well as their applicability for specific scenarios and the implications thereof.
This means that PrivEval is a first step to bridge the gap between privacy experts and the general public for making privacy estimation more transparent.

## How to run the demo on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run demo.py
   ```

## Tables of available privacy metrics for each attack

### Reconstruction risk
| **Metric**               | **Description**                                                                                                                                                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Attribute Inference Risk | AIR measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the a weighted F1-score.                              |
| GeneralizedCAP           | GCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm. |
| ZeroCAP                  | ZCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm. |

### Re-identification risk
### Membership inference / Tracing risk


## Other information
```
A python notebook is available to generate the real and synthetic dataset using the file gen_dataset.ipynb.
```
