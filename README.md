# Zero-Shot Rationalization by Multi-Task Transfer Learning from Question Answering
**EMNLP 2020 Findings**
## To Reproduce
**We provide the reproduce code for proposed SBM model**
Before reproducing, make sure the below python packages installed
```
torch 
    Version: 1.5.0
json
tqdm
pickle
argparse
transformers
    Version: 2.5.1
```
## How to train
To trained our proposed model, follow the step below

1. Clone the repositorry with git lfs
    `git lfs clone`
2. cd in **Zero-Shot-Rationalization-By-QA/**
3. Download **train-v2.0.json and dev-v2.0.json** from SQuAD2.0 webpage and put them into **data/**
4. preprocessing all data, run
    `./make_data.sh`
5. To train the model, run
    `./_run_train.sh`
    You can chage the using GPU and other hyperparameters in this shell script
6. The best model(with lowest valid loss) will happen in the second or third epochs, check the output when training to select the best model
7. The model will be store in **Zero-Shot-Rationalization-By-QA/model_merge3/** and in that directory, model_0.pickle is the first epoch model, model_1.pickle is the second, and so on.

## How to extract rationales
After training is dene, the model will be saved in **Zero-Shot-Rationalization-By-QA/model_merge3/** , pick the best model from output valid loss and change the **--predict_model** in **_run_beer_reason.sh** and **_run_movie_reason.sh**

### Beer rationalizing
To extract the rationales using our model, follow the steps below
1. run appearance(0), aroma(1), palate(2)
    `./_run_beer_reason.sh 0 `
    The first input argument is the choosing aspects, 0 is appearance, 1 is aroma, 2 is palate. The output will be store in software/predict_beer_[0/1/2].json
2. The output from stdout will contain **AVG Len**, you can scale the **--want_len** in **./_run_beer_reason.sh** script to acquire rationales with different highlighted ratio. For example, since the avg context length is 127, if you want a 10% highlighted rationales, the output **AVG Len** should be 12. If 20% highlighted rationales is expected, the output **AVG Len** should be 25. 
### Movie rationalizing
To extract the rationales using our model, simply execute the command below:
```
./_run_movie_reason.sh
```
The output will be store in software/predict_movie.json

## Data description
Before running code, please download the data/ directory and move it under software/ directory, both are provided as a zip file when submit the paper.
And make sure the data list below was download and placed in the data/ directory
### SQuAD dataset
* train-v2.0.json *(Not provided)*
    * Download it from https://rajpurkar.github.io/SQuAD-explorer/
* dev-v2.0.json *(Not provided)*
    * Download it from https://rajpurkar.github.io/SQuAD-explorer/

### Beer Advocate dataset
We use the data provided by the paper ("Rationalizing Neural Predictions". Tao Lei, Regina Barzilay and Tommi Jaakkola. EMNLP 2016), the link of data can be find in http://people.csail.mit.edu/taolei/beer/
* annotations.json *(Provided)*
    * Directly download from the link above, is already in the data directory
* reviews.mixed3.txt *(Provided)*
    * randomly picked 20000 data from **reviews.aspect0.train.txt**, **reviews.aspect1.train.txt**, **reviews.aspect2.train.txt** and merge together
* reviews.mixed_heldout.txt *(Provided)*
    * randomly picked 20000 data from **reviews.aspect0.heldout.txt**, **reviews.aspect1.heldout.txt**, **reviews.aspect2.heldout.txt** and merge together

### Movie reviews dataset
We use the data provide from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews as training and valid data, and use the data provided in (ERASER benchmark)http://www.eraserbenchmark.com as test data

* IMDB_dataset.csv *(Provided)*
    * Downloaded directly from the above link, contains 50000 data for training and validation
* movie_sentence_layer_test.json *(Provided)*
    * We split each sentence in the test data provided by ERASER to form a dataset in SQuAD 2.0 format. This is used to feed into our model for rationalizing
* movie_rationale.pickle *(Provided)*
    * We extract the labeled rationales in test data and store it into a dictionary

## Code description
all codes needed for reproduce our experiment is provided in software/

* script/train.py
    * Include model structure, training process and predict process
* script/data_preproc.py
    * script forr data preprocessing
* script/evaluate_qa.py
    * script contains evaluation metrics
* script/process_beer.py
    * preprocessing for beer training and valid data
* script/process_movie.py
    * preprocessing for movie training and valid data
* script/ask_beer_reason.py
    * Creating SQuAD format data for beer rationalization
* make_data.sh
    * run to preprocces data for training and testing
* _run_train.sh
    * run to train our model
* _run_beer_reason.sh
    * shell script for beer rationale prediction, and evaluate the IOU F1 and Token F1 scores
* _run_movie_reason.sh
     * shell script for movie rationale prediction, and evaluate the IOU F1 and Token F1 scores

