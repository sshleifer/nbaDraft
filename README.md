##nbaDraft
Predicting NBA performance using college box-score stats and NBA combine measurements, using python.
###scrapeDX.py
  Creates colData.csv, proData.csv, and measurements.csv,
  by scraping draftexpress.com.
  It takes a while to to run, so I posted the csvs it produces.
###assemble_dataset.py
  Produces train.csv and validated_set.csv.
  Takes almost no time to run.
  assemble_dataset.dummy_out will handle NAs,
  when called from model.py
###model.py
  My attempts at using NCAA boxscore stats and measurements to predict Regularized Adjusted Plus Minus for former US college players. Some work better than others.
