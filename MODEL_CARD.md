# PrivEval: a tool for interactive evaluation of privacy metrics in synthetic data generation

> \*\*Research use only. Not a medical device.\*\*
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.



## 1\. Overview

|Field|Value|
|-|-|
|Name|Frederik Marinus Trudslev|
|Version||
|Type|Demo|
|Owner partner and contact|Aalborg University (AAU), Frederik Marinus Trudslev, fmtr@cs.aau.dk|
|License|Apache-2.0|
|Repository and DOI|https://github.com/hereditary-eu/PrivEval|
|Release date|01 August 2025|
|Related task or deliverable||
|How to cite|Frederik Marinus Trudslev. (2025). PrivEval: A Tool for Interactive Evaluation of Privacy Metrics in Synthetic Data Generation. https://github.com/hereditary-eu/PrivEval|

## 2\. Intended use

**Primary intended use.** The user can investigate what the privacy risks for the attendee are, if they join the demo, and how confident a data curator should be from a privacy perspective, if the synthetic dummy data is released?

**Intended users.** Researchers, data scientists.

**Out-of-scope uses.** This release must not be used for

* clinical diagnosis, prognosis or treatment of individual patients;
* decisions about individuals, such as insurance, employment or access to care;
* re-identification of individuals or linkage with other data sources-



## 3\. Data provenance

This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.



## 4\. Method

* **Algorithm or architecture.** PrivEval is an interactive, Streamlit-based software tool for evaluating and interpreting privacy metrics for synthetic data generation. It provides a unified interface for exploring privacy metrics, their underlying assumptions, applicability, and implications for different evaluation scenarios. The tool covers reconstruction, re-identification, membership inference, and tracing risks. 
* **Training setting.** Not applicable to PrivEval itself.
* **Privacy techniques:** Evaluates privacy risks in synthetic data; does not itself guarantee differential privacy.



## 5\. Performance

* **Evaluation data.** User-provided real and synthetic tabular CSV datasets.
* **Metrics and results.** Computes multiple privacy-risk metrics. Results depend on the datasets, generation method, and metric configuration. No single standardized performance score applies.
* **Results by subgroup.** No standardized subgroup-performance results reported.



## 6\. Known limitations and risks

* **Generalizability.** Depends on dataset characteristics, synthesis methods, and metric assumptions. Primarily designed for tabular data.
* **Known biases and confounders.** Dataset size, class imbalance, preprocessing, and metric-specific assumptions may affect results.
* **Failure modes.** Unsupported inputs, including missing values or repeated individuals, may affect metric validity.
* **Privacy risks.** Metrics estimate privacy risks but do not guarantee anonymity or prevent data leakage. No comprehensive independent membership-inference or model-inversion assessment is established.



## 7\. Responsible use and reporting

* **Terms of use.** Publicly available under the Apache License 2.0
* **Contact for misuse or vulnerability reports.** Aalborg University (AAU), Frederik Marinus Trudslev, fmtr@cs.aau.dk



## 8\. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.

