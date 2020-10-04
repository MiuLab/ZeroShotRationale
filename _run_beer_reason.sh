python3 script/data_preproc.py --data_file data/beer_reason_${1}_Q.json --output_dir data/ --data_tag test
CUDA_VISIBLE_DEVICES=0 python3 script/train.py  \
--predict \
--batch_size 256 \
--test_file  data/test_data.pickle \
--predict_model model_merge3/model_2.pickle \
--predict_output predict_beer_${1}.json \
--predict_type BEER \
--want_len 0.035 \
--score \
--raw_data_file data/beer_reason_${1}_Q.json


