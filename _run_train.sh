CUDA_VISIBLE_DEVICES=0 python3 script/train.py \
--train \
--batch_size 12 \
--batch_size_1 3 \
--epoch_num 5 \
--train_file data/train_data.pickle \
--valid_file data/valid_data.pickle \
--train_file_1 data/beer_data_3aspect_20000_albert.pickle \
--train_file_1_num 10000 \
--train_step_num 10 \
--valid_file_1 data/beer_data_3aspect_1000_valid_albert.pickle \
--train_file_2 data/movie_train_albert.pickle \
--valid_file_2 data/movie_valid_albert.pickle \
--beer_loss_mul 10 \
--movie_loss_mul 10 \
--output_model model_merge3


