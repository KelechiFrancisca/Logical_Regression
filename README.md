# Loan_Approval_Prediction

## Loan approval prediction using Logistic Regression, OneHotEncoding, and StandardScaler. Achieved 92% accuracy

## Case Study
> This project uses Logistic Regression to predict loan approvals based on applicant data.
> It demonstrates end‑to‑end machine learning workflow: preprocessing, model training, evaluation, and business insights.

---
## Datasets
> loan_id	 no_of_dependents	 education	 self_employed	 income_annum	 loan_amount	 loan_term	 cibil_score	 residential_assets_value	 commercial_assets_value	 luxury_assets_value	 bank_asset_value	 loan_status
1	2	1	 No	9600000	29900000	12	778	2400000	17600000	22700000	8000000	1
2	0	0	 Yes	4100000	12200000	8	417	2700000	2200000	8800000	3300000	0
3	3	1	 No	9100000	29700000	20	506	7100000	4500000	33300000	12800000	0
4	3	1	 No	8200000	30700000	8	467	18200000	3300000	23300000	7900000	0
5	5	0	 Yes	9800000	24200000	20	382	12400000	8200000	29400000	5000000	0
6	0	1	 Yes	4800000	13500000	10	319	6800000	8300000	13700000	5100000	0
7	5	1	 No	8700000	33000000	4	678	22500000	14800000	29200000	4300000	1
8	2	1	 Yes	5700000	15000000	20	382	13200000	5700000	11800000	6000000	0
9	0	1	 Yes	800000	2200000	20	782	1300000	800000	2800000	600000	1
10	5	0	 No	1100000	4300000	10	388	3200000	1400000	3300000	1600000	0
11	4	1	 Yes	2900000	11200000	2	547	8100000	4700000	9500000	3100000	1
12	2	0	 Yes	6700000	22700000	18	538	15300000	5800000	20400000	6400000	0
13	3	0	 Yes	5000000	11600000	16	311	6400000	9600000	14600000	4300000	0
14	2	1	 Yes	9100000	31500000	14	679	10800000	16600000	20900000	5000000	1
<img width="897" height="301" alt="image" src="https://github.com/user-attachments/assets/c456337d-0cb3-4156-88a1-190e3b87664b" />

---
## Steps
## Python codes for Polynomial Logical Regression
## Importing the libraries
``` Codes
```
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
```
---
## Importing the dataset
```
dataset = pd.read_csv('loan_approval_edited.csv')
#exclude the first column
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
