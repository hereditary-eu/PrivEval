# PrivEval: a tool for interactive evaluation of privacy metrics in synthetic data generation

<p align="center">
<img center=True width="904" height="246" alt="PrivEval_model" src="https://github.com/user-attachments/assets/dabde83a-634a-4e70-8d16-fede51e268c6" />
</p>

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

The definitions of each privacy metric can be found in the [Technical Report](https://doi.org/10.48550/arXiv.2507.11324). For less elaborate definitions of the privacy metrics, we refer to this table:

### Reconstruction risk
| **Metric**               | **Description**                                                                                                                                                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Attribute Inference Risk | AIR measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the a weighted F1-score.                              |
| GeneralizedCAP           | GCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm. |
| ZeroCAP                  | ZCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm. |

### Re-identification risk
| **Metric**                             | **Description**                                                                                                                                                                                                                                                                                                          |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hidden Rate                            | Hidden Rate estimates the risk of identifying whether an individual contributed their data to the real dataset while only having access to the synthetic data.                                                                                                                                                           |
| Hitting Rate                           | Hitting Rate measures the risk of identifying whether an individual contributed their data to the real dataset while having access to the synthetic data. To identify whether or not you contributed your data, the metrics tries to find individuals with matching categorical and similar continuous attribute values. |
| Membership Inference Risk              | MIR estimates the risk of identifying whether an individual contributed their data to the real dataset while having access to the synthetic data and a subset of the real data. Here, a classifier is trained to determine whether individuals are synthetic or real and estimates privacy through this.                 |
| Nearest Neighbour Adversarial Accuracy | NNAA estimates the risk of identifying whether an individual contributed their data to the real dataset while only having access to the synthetic data. This is done by mapping the datasets to 2D and estimated using distances to nearest neoghbors.                                                                   |

### Membership inference / Tracing risk
| **Metric**                          | **Description**                                                                                                                                                                                                                                                                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Authenticity                        | Auth measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. The Auth risk is measured as the probability that a synthetic nearest neighbour is closer than a real nearest neighbour over the real dataset.                  |
| Close Value Probability             | CVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. This is measured as a probabilistic likelihood of synthetic individuals being 'too close'.                                                                       |
| Common Rows Proportion              | CRP measures the risk of re-identification as a probability of a real individual's row being a row in the synthetic data.                                                                                                                                                                                                        |
| Distance to Closest Record          | DCR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. This is done by measuring the distance to the nearest neighbour in data transformed to 2 dimensions.                                                             |
| DetectionMLP                        | D-MLP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated, while having access to a subset of the real data. Here an MLP classifier is trained to real individuals and tested on synthetic individuals.                   |
| Distant Value Probability           | DVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. This is measured as a reverse probabilistic likelihood of synthetic individuals being 'too far away'.                                                            |
| Identifiability Score               | IdScore estimates the risk of re-identifying any real individual while only having access to the synthetic data. It estimates this as the probability that the distance to the closest synthetic individual is closer than the distance from the closest real individual in weighted versions of the real and synthetic dataset. |
| Median Distance to Closest Record   | MDCR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. This is measured as the median distance between the real and synthetic data points.                                                                             |
| Nearest Synthetic Neighbor Distance | SND measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated through a distance measure. The score is calculated as the mean min-max reduced distance to the nearest synthetic neighbour.                                      |
| Nearest Neighbor Distance Ratio     | NNDR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated. This is measure as the distance ratio between real and synthetic data.                                                                                          |

## Requirements
The input datasets must be single table CSV files with no individuals repeating in the dataset for correct metric calculation.
The input datasets must not contain missing values

## Other information

A python notebook is available to generate the real and synthetic dataset using the file gen_dataset.ipynb.

For some metrics, we refer to their individual implementations in the Metrics folder to change e.g. thresholds and iterations.

If you use this code, please cite the associated paper:
```

```
