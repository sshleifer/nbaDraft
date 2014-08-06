##nbaDraft
Predicting NBA performance with college box-score stats, combine measurements and python.
###datasets
The data used to make predictions is in colPro.csv, comprised from draft_mix.csv, colData.csv, bestRapm.csv and measurements.
###scrapers
A handful of web requests that make the datasets you see in the datasets folder.
They take ~1hr to run, so I'll update them when they need to be updated.
###assemble_dataset.py
  Merges many of the datasets into colPro.csv, and then splits that into train.csv and test.csv.
###model.py
  My attempts at using NCAA boxscore stats and measurements to predict Regularized Adjusted Plus Minus for former US college players. I try different techniques for feature engineering, missing data, and ensemble modeling.
