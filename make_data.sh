#Beer train/valid
python3 script/process_beer.py --input data/reviews.mixed3.txt --output data/beer_data_3aspect_20000_albert.pickle --tokenizer albert
python3 script/process_beer.py --input data/reviews.mixed_heldout.txt --output data/beer_data_3aspect_1000_valid_albert.pickle --tokenizer albert
# Beer test
python3 script/ask_beer_reason.py data/beer_reason_0_Q.json 0 1000
python3 script/ask_beer_reason.py data/beer_reason_1_Q.json 1 1000
python3 script/ask_beer_reason.py data/beer_reason_2_Q.json 2 1000
#Movie train/valid
python3 script/process_movie.py \
    --input data/IMDB_dataset.csv \
    --output_t data/movie_train_albert.pickle \
    --output_v data/movie_valid_albert.pickle \
    --valid_num 5000 \
    --tokenizer albert
#SQuAD data train/valid
python3 script/data_preproc.py --data_file data/train-v2.0.json --output_dir data/ --data_tag train
python3 script/data_preproc.py --data_file data/dev-v2.0.json --output_dir data/ --data_tag valid
