# IU_Comp_Pathology_Project

## Research/Data Engineering Judgement
### 1. Data leakage
#### What kinds of data leakage would you worry about in a medical imaging dataset?
One form of data leakage is that there could be multiple chest X-rays from the same patient. This is especially an issue if they are in both the training and test/validation sets, as this could cause overfitting. Depending on how data was collected there could be issues with standardization as scans taken by different equipment or even at different hospitals could have differences that correlate within that group but don't broadly generalize. For example a set of data taken from one machine might be brighter or darker than the others or have some specific artifacting.
#### Does this dataset contain enough information for you to rule all of them out?
No, we don't have patient, hopsital, or machine data here so it's impossible to check for these types of leakage as we can't really "match patients" or after the fact guess on hospital or machine use groupings. We also know we can't rule this out because we found a ton of duplicates using hashing in our validation process which could present a huge issue.

### 2. Scaling
#### Imagine the same pipeline now handles: 1,000,000 medical images arriving continuously from several hospitals. What are the first three things you would change about your implementation? Do not design an entire cloud architecture. We want to understand your priorities.
First, I would change how I was storing data to be in a SQL database with cloud storage, as it wouldn't make sense to have a single local file. Second, I would make it so data could be uploaded asynchronously so that hospitals could submit scans whenever into the system and then be processed so that it could be updated continuously and/or in batches. Third, I would make the duplicate image detection be through a look-up index rather than getting the full dataset everytime. 

### 3. Incremental processing
#### Suppose 10,000 new images arrive tomorrow. How would you avoid recalculating metadata for the existing 1,000,000 images? How would you recognize: a new image, an exact duplicate, an image that had previously failed processing
I would treat the new images as an incremental delta job. Essentially I would check if source_key/GUID has been seen, if no, I check that there isn't an exact duplicate and then run validation and process accordingly. If it fails at the processing stage store the GUID with a "failed" flag. A new image would be identified when the source file is unique and doesn't have a duplicate. A duplicate is identified by the SHA-256 hash. A previously failed image would be identified by looking at the processing flag. 

### 4. Validation
#### Describe one scenario where your pipeline could execute successfully but still produce a scientifically misleading result. How would you detect or prevent it?
The SHA-256 hashing only detects exact duplicates so there could be duplicates that are just a pixel off that it doens't detect and then we would have duplicatese that could skew results. USing perceptual hashing as a secondaery check could hep mitigate this. 


## How to Run It
First install dependencies included in requirements.txt. Then, obtain the dataset by running src/download.py. To get the metadata tables, run build_dataset.py. To get the SQL analysis run analyze.py. To get the model and evaluation, run model.py. In order to run the entire pipeline you can run the following command: pip install -r requirements.txt && python src/download.py && python src/build_dataset.py && python src/analyze.py && python src/model.py

## Approach
This project establishes a complete end-to-end medical image classification pipeline for PneumoniaMNIST, combining local dataset ingestion with relational metadata indexing, exploratory SQL analytics, and a scikit-learn baseline model. The pipeline stores flattened 28x28 grayscale pixel arrays in a local .npz container while extracting sample-level metrics (mean, standard deviation, min/max intensity, and split designations) into SQLite (output/metadata.db) and Parquet (output/metadata.parquet), using SHA-256 hashes to enforce exact byte-for-byte deduplication without committing heavy binary assets to Version Control. Part 2 leverages SQL queries to evaluate a persistent ~74% class imbalance toward Pneumonia across all splits and identifies extreme dynamic-range outliers for visual inspection. Part 3 fits a Logistic Regression model (C=0.1, max_iter= 500, solver=lbfgs) strictly on the training set, uses a PredefinedSplit grid search on the validation set to tune hyperparameters, and achieves 0.8285 accuracy, 0.9846 sensitivity, and an AUC of 0.9185 on the test set. Key architectural limitations include the inability of cryptographic hashes to detect near-duplicate images (such as minor JPEG compression or 1-pixel shifts) and the absence of patient-level IDs in PneumoniaMNIST, which prevents ruling out cross-split patient overlap leakage.

## AI Assistance
I used Gemini for some basic code assistance. This included asking for syntax on how to do certain coding tasks and asking for pointers to documentation. I also used it to help reword my writing for the approach section. I don't have a specific example of a piece of code, but I used it more as a tool to check documentation and then implemented it into my code. When relevant, I also did some testing with some print statement to make sure I was appropriately understanding and using different modules. 