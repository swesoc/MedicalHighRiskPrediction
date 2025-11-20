# MedicalHighRiskPrediction
The repo contains Project to detect a person is at high risk medically

## Problem Statement

Organizations in the healthcare and insurance domains need the ability to accurately identify individuals who are at high risk due to a combination of demographic, socio-economic, and medical factors. Early identification enables better risk mitigation, personalized interventions, and optimized resource allocation. However, determining risk manually is challenging because it depends on many interacting variables and non-linear relationships.

The objective of this project is to develop a machine-learning model that predicts whether a person is at high risk, based on a diverse set of input features. These features may include:

Demographic attributes: age, income level, employment status, socio-economic background

Health-related attributes: existing diseases, past medical history, chronic conditions

Insurance-related attributes: insurance policy type, coverage details, co-pay amounts, policy term duration, previous claim history

Other contextual factors that may influence risk level

The model will learn from historical labeled data—where the risk status of individuals is already known—and will identify patterns that correlate with high-risk outcomes. The primary task is framed as a binary classification problem (high risk vs. not high risk). Secondary objectives may include identifying the most influential features and evaluating model fairness across demographic groups.

## Solution Overview
A successful solution will provide:

Accurate risk predictions for new individuals

Explainable insights into key risk drivers

Scalable and ethical deployment for real-world decision-making

## Run locally 
```bash
git clone https://github.com/swesoc/MedicalHighRiskPrediction.git
----------------
Windows machine
winget install --id=astral-sh.uv -e
Linux/macOS
# Using pipx
pipx install uv
# Or using curl for Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
------------

uv venv .venv





```
### Install Dependencies

```
uv pip install pandas numpy seaborn matplotlib.pyplot  statsmodels scikit-learn


```
### Activate virtual environment
```bash
For windows
.venv\Scripts\activate

 Linux/macOS
source .venv/bin/activate

```



### Deactivate the environment
```
deactivate
exit
```

for training the model on command line 

```

python train.py 

```
## Project File Structure

parent_directory/  
    data/  
    ── medical_insurance.csv  
    notebook.ipynb  
    train.py  

