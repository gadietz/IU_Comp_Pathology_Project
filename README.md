# IU_Comp_Pathology_Project

##Research/Data Engineering Judgement
###1. Data leakage
What kinds of data leakage would you worry about in a medical imaging dataset?
One form of data leakage is that there could be multiple chest X-rays from the same patient. This is especially an issue if they are in both the training and test/validation sets, as this could cause overfitting. Depending on how data was collected there could be issues with standardization as scans taken by different equipment or even at different hospitals could have differences that correlate within that group but don't broadly generalize. For example a set of data taken from one machine might be brighter or darker than the others or have some specific artifacting.
Does this dataset contain enough information for you to rule all of them out?
No, we don't have patient, hopsital, or machine data here so it's impossible to check for these types of leakage as we can't really "match patients" or after the fact guess on hospital or machine use groupings. We also know we can't rule this out because we found a ton of duplicates using hashing in our validation process which could present a huge issue.



##How to Run It
First install dependencies included in requirements.txt. Then, obtain the dataset by running src/download.py. To get the metadata tables, run build_dataset.py. To get the SQL analysis run analyze.py. To get the model and evaluation, run model.py. In order to run the entire pipeline you can run the following command: pip install -r requirements.txt && python src/download.py && python src/build_dataset.py && python src/analyze.py && python src/model.py

