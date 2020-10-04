python3 script/data_preproc.py --data_file data/movie_sentence_layer_test.json --output_dir data/ --data_tag test
CUDA_VISIBLE_DEVICES=0 python3 script/train.py  \
--predict \
--batch_size 256 \
--test_file  data/test_data.pickle \
--predict_model model_merge3/model_2.pickle \
--predict_output predict_reason.json \
--predict_type MOVIE \
--score \
--score_movie \
--raw_data_file data/movie_rationale.pickle 


