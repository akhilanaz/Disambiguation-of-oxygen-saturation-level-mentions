**Oxygen Supplementation Classifier for COVID-19 Patient Discharge Summaries**

**Overview**
This repository contains the code and data for classifying COVID-19 patients based on whether they received oxygen supplementation. The classification is performed using various machine learning (ML) algorithms, from classical ML models to deep learning (DL) models. The decisions made by these classifiers are explained using Local Interpretable Model-agnostic Explanations (LIME).

**Background**
Oxygen saturation is a crucial indicator of COVID-19 severity. Identifying patients who received oxygen supplementation is essential for COVID-19 research, especially in cases of silent hypoxemia. However, this information is often buried within clinical narratives in electronic health records (EHRs). This project aims to automate the extraction of such information, reducing the manual review burden on physicians.

**Method**
Data: Anonymized COVID-19 patient discharge summaries in German were used.
Task: Binary classification to identify patients who received oxygen supplementation.
Models: Various ML models, including both classical approaches and deep learning models, were trained and compared.
Explanation: LIME was used to explain model decisions and visualize the most relevant features at the token level.

**Results**
Both classical ML and DL models achieved comparable performance, with F-measures ranging from 0.942 to 0.955.
Classical ML models were faster in terms of computation.
Embedding visualizations revealed notable differences in encoding patterns between classical and DL models.
LIME provided qualitative explanations for the model performance by highlighting important features.


**Requirements**
Python 3.x
Required packages are listed in requirements.txt.

Journal: Abdulnazar, A., Kugic, A., Schulz, S. et al. O2 supplementation disambiguation in clinical narratives to support retrospective COVID-19 studies. BMC Med Inform Decis Mak 24, 29 (2024). https://doi.org/10.1186/s12911-024-02425-2
