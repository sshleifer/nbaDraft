##nbaDraft
Predicting NBA performance with college box-score stats and NBA combine measurements and python.
###scrapeDX.py
  Creates dx_mock_drafts.csv, colData.csv, proData.csv, and measurements.csv
  by scraping draftexpress.com.
  It takes a while to to run, so I posted the csvs it produces.
###assemble_dataset.py
  Produces train.csv and validated_set.csv.
  Takes almost no time to run.
  I use the dummying out strategy for missing data.
###model.py
  My attempts at using NCAA boxscore stats and measurements to predict Regularized Adjusted Plus Minus for former US college players. Some work better than others.
