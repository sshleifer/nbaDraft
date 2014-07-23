nbaDraft
========

Predicting NBA performance using college box-score stats and combine measurements.


scrapeDX.py: 
  Creates colData.csv, proData.csv, and measurements.csv,
  by scraping draftexpress.com.
  It takes a while to to run, so I'd recommend using
  the csvs I posted.

assemble_dataset.py:
  Produces train.csv and validated_set.csv.
  Takes almost no time to run.
  assemble_dataset.dummy_out will handle NAs,
  when called from model.py


model.py:
  My attempts at predicting Regularized Adjusted Plus Minus for former US college players,
  using various sklearn modules.
