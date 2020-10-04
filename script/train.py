import torch
import json
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
import os.path
import math
import json
import random
import sys
from os import path
from transformers import AlbertTokenizer, AlbertModel
from evaluate_qa import *

beer_loss_mul=1
movie_loss_mul=1
want_len = 0.035

#check cuda
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def load_data(file_name):
    print("Start loading file[%s]" % file_name)
    with open(file_name,"rb") as F:
        data=pickle.load(F)
    F.close()
    print("End loading")
    return(data)
def make_dataloader(data_file, shuffle=False, batch_size=8, train_step_num = 1, beer_data_num = -1):
    data_data = load_data(data_file)
    if beer_data_num !=-1:#beer
        #appearance aroma plate each 20000
        rand_start = [ random.randint(0, 20000-1 - beer_data_num) + 20000*i for i in range(3)]
        rand_end = [ i + beer_data_num for i in rand_start]
        data_data = data_data[rand_start[0]:rand_end[0]] + data_data[rand_start[1]:rand_end[1]] + data_data[rand_start[2]:rand_end[2]]

    #make dataset and dataloader
    batch_total_num = math.ceil(len(data_data)//batch_size)
    batch_num = batch_total_num//train_step_num
    if batch_num < 1:
        print("Warning, batch_num < 1, decrease train_step_num!")
        exit(0)
    data_loader_list = []
    for i in range(train_step_num):
        if i == train_step_num-1:#last
            data_dataset = QADataset(data = data_data[i*batch_num*batch_size:])
        else:
            data_dataset = QADataset(data = data_data[i*batch_num*batch_size:(i+1)*batch_num*batch_size])
        data_loader_list.append(DataLoader(dataset=data_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn = collate_fn))
    return data_loader_list

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return torch.LongTensor(self.data[index]['text']), self.data[index]['q_id'], self.data[index]['answers'], self.data[index]['attn_mask'], self.data[index]['token_type_ids']
def collate_fn(batch):
    text = torch.stack([b[0] for b in batch])
    ids = [b[1] for b in batch]
    answers = [b[2] for b in batch]
    attn_mask = torch.stack([b[3] for b in batch])
    token_type_ids = torch.stack([b[4] for b in batch])
    return text, ids, answers, attn_mask, token_type_ids

class BertQA(nn.Module):
    def __init__(self, drop_prob=0):
        super(BertQA, self).__init__()
        self.bert = AlbertModel.from_pretrained("albert-large-v2")
        #for QA
        self.ans_se = nn.Linear(1024, 2)
        #for Beer, ten level sentiment
        self.sentiment = nn.Linear(1024, 1)
        self.sentiment_movie = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, attn_mask, token_type_ids, data_flag):
        #data_flag might be QA, BEER, MOVIE
        out = self.bert(x,attention_mask = attn_mask, token_type_ids = token_type_ids)
        out = out[0] # become [batch, seq_len, 768]
        if data_flag=="SQUAD":
            ans_se = self.ans_se(out)
            return  ans_se[:,:,0], ans_se[:,:,1]
        if data_flag=="BEER":
            out = out[:,0,:].squeeze()
            beer_sent = self.sigmoid(self.sentiment(out))
            return beer_sent.squeeze()
        if data_flag=="MOVIE":
            out = out[:,0,:].squeeze()
            movie_sent = self.sigmoid(self.sentiment_movie(out))
            return movie_sent.squeeze()


def run_epoch(model, data_loader, loss_func, optimizer, is_train, epoch_i, epoch_num, batch_size, output_model , data_flag, step_num, train_step_num=1):
    if is_train:
        model.train()
        model_type="Train"
    else:
        model.eval()
        model_type="Valid"
    avg_loss = 0.
    iter_bar=tqdm(data_loader, leave = False)
    cnt = 0
    for x, ids, answers, attn_mask, token_type_ids in iter_bar:
        loss=0
        batch_size = x.size(0)
        optimizer.zero_grad()
        iter_bar.set_description("Run iter [%s]" % data_flag)
        if is_train:
            if data_flag=="SQUAD":
                out_ans_start, out_ans_end= model(x.to(device),attn_mask.to(device), token_type_ids.to(device), data_flag)
                ans_start_list = (torch.LongTensor([i[0][0] for i in answers])).to(device)
                ans_end_list = (torch.LongTensor([i[0][1] for i in answers])).to(device)
                loss_s = loss_func(out_ans_start, ans_start_list)
                loss_e = loss_func(out_ans_end, ans_end_list)
                loss = (loss_s + loss_e)/2
            if data_flag=="BEER":
                answers=torch.FloatTensor(answers).to(device)
                cnt_aspect=0
                for aspect in BEER_IDX:
                    for i in range(batch_size):
                        x[i][int(torch.sum(token_type_ids[i]))+4] = aspect #replace question
                    beer_sent = model(x.to(device),attn_mask.to(device), token_type_ids.to(device),data_flag)
                    #beer sent --> [batch_size, 1], score = [batch_size*5]
                    loss += loss_func(beer_sent, answers[:,cnt_aspect])
                    cnt_aspect+=1
                loss = loss * beer_loss_mul
            if data_flag=="MOVIE":
                answers=torch.FloatTensor(answers).to(device)
                movie_sent = model(x.to(device),attn_mask.to(device), token_type_ids.to(device),data_flag)
                loss = loss_func(movie_sent, answers)
                loss = loss
            loss.backward()
            optimizer.step()
        if is_train == False:
            with torch.no_grad():
                if data_flag=="SQUAD":
                    out_ans_start, out_ans_end = model(x.to(device),attn_mask.to(device), token_type_ids.to(device), data_flag)
                    ans_start_list = (torch.LongTensor([i[0][0] for i in answers])).squeeze().to(device)
                    ans_end_list = (torch.LongTensor([i[0][1] for i in answers])).squeeze().to(device)
                    loss_s = loss_func(out_ans_start, ans_start_list)
                    loss_e = loss_func(out_ans_end, ans_end_list)
                    loss = (loss_s + loss_e)/2
                if data_flag=="BEER":
                    answers=torch.FloatTensor(answers).to(device)
                    cnt_aspect=0
                    for aspect in BEER_IDX:
                        for i in range(batch_size):
                            x[i][int(torch.sum(token_type_ids[i]))+4] = aspect #replace question
                        beer_sent = model(x.to(device),attn_mask.to(device), token_type_ids.to(device),data_flag)
                        #beer sent --> [batch_size, 1], score = [batch_size*5]
                        loss += loss_func(beer_sent, answers[:,cnt_aspect])
                        cnt_aspect+=1
                if data_flag=="MOVIE":
                    answers=torch.FloatTensor(answers).to(device)
                    movie_sent = model(x.to(device),attn_mask.to(device), token_type_ids.to(device),data_flag)
                    loss = loss_func(movie_sent, answers)

        avg_loss += loss.item()
        cnt+=1
    if (step_num % (math.ceil(train_step_num/5)) == 0):
        if is_train:
            torch.save(model, '{}/check_point_{}_{}.pickle'.format(output_model, epoch_i, step_num))  # save the whole net
        print("[{}][{}] Epoch {}/{} Step {}/{} Done, Total Loss: {}".format(model_type, data_flag ,epoch_i, epoch_num, step_num, train_step_num, avg_loss/len(data_loader)))

        

def train(model, train_loader, valid_loader, train_loader_1, valid_loader_1, train_loader_2, valid_loader_2, epoch_num, start_epoch, loss_func, loss_func_1,optimizer, batch_size, output_model, train_step_num):
    epoch_bar = tqdm(range(start_epoch, epoch_num))
    print("----------------------------------------------------")
    for i in epoch_bar:
        #run train
        epoch_bar.set_description("Processing epoch[%d]"%i)
        squad_step_num=0
        beer_step_num=0
        iter_step = tqdm(range(train_step_num), leave=False) 
        for j in iter_step:
            iter_step.set_description("Running Epoch[{}] steps...".format(i+1))
            run_epoch(model, train_loader_1[j], loss_func_1, optimizer, True , i+1, epoch_num, batch_size, output_model,"BEER",j+1, train_step_num)
            run_epoch(model, train_loader_2[j], loss_func_1, optimizer, True , i+1, epoch_num, batch_size, output_model, "MOVIE",j+1, train_step_num)
            run_epoch(model, train_loader[j], loss_func, optimizer, True , i+1, epoch_num, batch_size, output_model, "SQUAD",j+1, train_step_num)
        torch.save(model, '{}/model_{}.pickle'.format(output_model, i))  # save the whole net
        torch.save(optimizer.state_dict(),  '{}/optim_{}.pickle'.format(output_model, i))
        run_epoch(model, valid_loader_1[0], loss_func_1, optimizer, False , i+1, epoch_num, batch_size, output_model, "BEER", 1)
        run_epoch(model, valid_loader_2[0], loss_func_1, optimizer, False , i+1, epoch_num, batch_size, output_model, "MOVIE", 1)
        run_epoch(model, valid_loader[0], loss_func, optimizer, False , i+1, epoch_num, batch_size, output_model, "SQUAD", 1)
        print("----------------------------------------------------")
        # run valid
    print("Done training")
    return model

def predict(model, test_loader, predict_name, tokenizer, predict_type):
    model.eval()
    #run predict 
    iter_bar=tqdm(test_loader)
    all_predict = {}
    softmax = nn.Softmax(dim = 0)
    no_ans = 0
    total_ans_len = 0
    prev_last = -1
    for x, ids, answers, attn_mask, token_type_ids in iter_bar:
        batch_size = x.size(0)
        model.zero_grad()
        with torch.no_grad():
            out_ans_start, out_ans_end = model(x.to(device),attn_mask.to(device), token_type_ids.to(device), "SQUAD")
            for i in range(batch_size):
                c_len = int(torch.sum(token_type_ids[i]).item())
                si = softmax(out_ans_start[i][:c_len])
                ei = softmax(out_ans_end[i][:c_len])
                if predict_type == "BEER":
                    ans_start = torch.argmax(si)
                    ans_end = torch.argmax(ei)
                    if ans_start==0 or ans_end ==0:
                        new_ans = ""
                    else:
                        si[0]=0
                        ei[0]=0
                        if ans_end < ans_start:
                            if si[ans_start]>ei[ans_end]:
                                ei[ans_end]=0
                                ans_end = ans_start+1+torch.argmax(ei[ans_start+1:])
                            else:
                                si[ans_start]=0
                                ans_start = torch.argmax(si[:ans_end])
                        prev_start = ans_start
                        prev_end = ans_end
                        while(ans_end >= ans_start and (ans_end.item()- ans_start.item()+1)/ torch.sum(token_type_ids[i]).item() < want_len):
                            s_big = si[ans_start]
                            e_big = ei[ans_end]
                            si[ans_start] = 0
                            ei[ans_end] = 0
                            ans_start_2 = torch.argmax(si)
                            ans_end_2 = torch.argmax(ei)
                            s_sec = si[ans_start_2]
                            e_sec = ei[ans_end_2]
                            if s_big - s_sec > e_big - e_sec:
                                if ans_end_2 > ans_end and (ans_end.item()- ans_start.item()+1)/ torch.sum(token_type_ids[i]).item() < want_len:
                                    ans_end = ans_end_2
                                if  ans_start > ans_start_2 and (ans_end.item()- ans_start.item()+1)/ torch.sum(token_type_ids[i]).item() < want_len:
                                    ans_start = ans_start_2
                            else:
                                if ans_start > ans_start_2 and (ans_end.item()- ans_start.item()+1)/ torch.sum(token_type_ids[i]).item() < want_len:
                                    ans_start = ans_start_2
                                if  ans_end_2 > ans_end and (ans_end.item()- ans_start.item()+1)/ torch.sum(token_type_ids[i]).item()<want_len:
                                    ans_end = ans_end_2
                            if ans_end == prev_end and ans_start == prev_start:
                                si[ans_start_2] =0
                                ei[ans_end_2] = 0
                            prev_start = ans_start
                            prev_end = ans_end
                    new_ans = tokenizer.decode((x[i][ans_start:ans_end+1])[:40])

                elif predict_type == "SQUAD" or predict_type=="MOVIE":
                    if predict_type == "MOVIE":
                        si[0]=0
                        ei[0]=0
                    ans_start = torch.argmax(si)
                    ans_end = torch.argmax(ei)
                    if ans_start==0 or ans_end ==0:
                        new_ans = ""
                    else:
                        new_ans = tokenizer.decode((x[i][ans_start:ans_end+1])[:40])


                total_ans_len += min(ans_end - ans_start + 1, 40)
                    #print("------------- ans_start: %d ---------------" % ans_start)
                    #print("------------- ans_end: %d ---------------" % ans_end)
                    #print(new_ans)
                if new_ans=="":
                    no_ans+=1
                new_ans = new_ans.replace("[UNK]","").replace("[CLS]","").replace("[SEP]","")
                if predict_type == "MOVIE":
                    if not (ids[i] in all_predict):
                        all_predict[ids[i]]=[""]
                    if new_ans != "":
                        if prev_last == ids[i]:
                            all_predict[ids[i]][-1] += (" "+new_ans)
                        else:
                            all_predict[ids[i]].append(new_ans)
                        if new_ans !="" and ans_end+1 >= len(ei)-1:#get last in sentence
                            prev_last = ids[i]
                        else:
                            prev_last=-1
                else:
                    all_predict[ids[i]] = new_ans
        
        iter_bar.set_description("Run iter")

    with open(predict_name,"w") as F:
        json.dump(all_predict, F)
    F.close()
    print("NO_ans: ",no_ans)
    print("AVG_len: ", total_ans_len/len(all_predict))


def predict_perf(model, test_loader, predict_type, ans_dict, beer_index):
    model.eval()
    #run predict 
    iter_bar=tqdm(test_loader)
    all_predict = {}
    softmax = nn.Softmax(dim = 0)
    cnt_true=0
    cnt=0
    out_ans = {}
    for x, ids, answers, attn_mask, token_type_ids in iter_bar:
        batch_size = x.size(0)
        model.zero_grad()
        with torch.no_grad():
            sent = model(x.to(device),attn_mask.to(device), token_type_ids.to(device), predict_type)
            for i in range(x.size(0)):
                if predict_type=="MOVIE":#binary
                    if sent[i]>0.5:
                        ans = 1
                    else:
                        ans = 0
                    if int(ids[i])<=1000:
                    #if answers[i] > 0:
                        ans_ = 1
                    else:
                        ans_ = 0
                else:
                    ans = round(sent[i].item(),1)
                    ans_ = ans_dict[ids[i]][beer_index]
                if ans//0.2 == ans_//0.2:
                    cnt_true+=1
                out_ans[ids[i]] = ans
                cnt+=1
    with open('perf_predict.pickle',"wb") as F:
        pickle.dump(out_ans, F)
    F.close()
    print("Performence:", cnt_true/cnt)


if __name__ =="__main__":
    #parse args 
    parser = argparse.ArgumentParser()
    # For both train and prredict
    parser.add_argument("--batch_size", help="the batch size when iterating", type=int,dest="batch_size")
    parser.add_argument("--batch_size_1", help="the batch size when iterating", type=int,dest="batch_size_1")
    parser.add_argument("--train_step_num", help="total training step", default= 10,type=int,dest="train_step_num")
    # For training
    parser.add_argument("--train", action='store_true', help="if spcified, set to training mode", dest="train")
    parser.add_argument("--epoch_num", help="the epoch_num when training, default = 1", type=int, default = 1,dest="epoch_num")
    parser.add_argument("--lr", help="the learning rate when training, default = 2e-3", type=float, default = 2e-3,dest="lr")
    parser.add_argument("--train_file", help="path of training file", dest="train_file")
    parser.add_argument("--valid_file", help="path of validation file", dest="valid_file")
    parser.add_argument("--train_file_1", help="path of training file", dest="train_file_1")
    parser.add_argument("--train_file_1_num", help="beer data used training num, default = 4000", type=int, default = 4000, dest="train_file_1_num")
    parser.add_argument("--beer_loss_mul", help="multiple for beer loss", type=float, default = 1.0, dest="beer_loss_mul")
    parser.add_argument("--movie_loss_mul", help="multiple for movie loss", type=float, default = 1.0, dest="movie_loss_mul")
    parser.add_argument("--valid_file_1", help="path of validation file", dest="valid_file_1")
    parser.add_argument("--train_file_2", help="path of training file", dest="train_file_2")
    parser.add_argument("--valid_file_2", help="path of validation file", dest="valid_file_2")
    parser.add_argument("--train_file_3", help="path of training file", dest="train_file_3")
    parser.add_argument("--valid_file_3", help="path of validation file", dest="valid_file_3")
    parser.add_argument("--output_model", help="path to output model and optimizer", dest="output_model")
    parser.add_argument("--random_seed", help="random seed for training, default = 0", type=int, default = 0, dest="random_seed")
    # for continue training
    parser.add_argument("--load_model", help="path of existing model", dest="load_model")
    parser.add_argument("--load_optim", help="path of existing optim", dest="load_optim")
    parser.add_argument("--continue", action='store_true', help="if spcified, continue to train a model", dest="cont")
    parser.add_argument("--start_epoch", help="the start epoch index when continue training, default = 0", type=int, default = 0 ,dest="start_epoch")

    # for predict
    parser.add_argument("--predict", action='store_true', help="predict_mode", dest="predict")
    parser.add_argument("--predict_perf", action='store_true', help="predict_performence", dest="predict_perf")
    parser.add_argument("--perf_file", help="path of performence answer file", dest="perf_file")
    parser.add_argument("--test_file", help="path of testing file", dest="test_file")
    parser.add_argument("--predict_model", help="path of predict model", dest="predict_model")
    parser.add_argument("--predict_output", help="path to output predict file", dest="predict_output")
    parser.add_argument("--predict_type", help=" predict type, SQUAD, BEER, MOVIE", dest="predict_type")
    parser.add_argument("--beer_index", help=" predict score, beer aspect",type=int, default=-1, dest="beer_index")

    # for cal predict score
    parser.add_argument("--score", action='store_true', help="score_mode", dest="score")
    parser.add_argument("--raw_data_file", help="path of raw data file for cal score", dest="raw_data_file")
    parser.add_argument("--predict_file", help="path to get predict file, when only cal score but don't predict", dest="predict_file")
    parser.add_argument("--score_movie", action='store_true', help="if score movie", dest="score_movie")
    parser.add_argument("--want_len", help=" When predicting beer rationale, set the threshold to control extract length",type=float, default=0.035, dest="want_len")
    args = parser.parse_args()
    #set random_seed
    torch.manual_seed(args.random_seed)
    if not(args.train or args.predict or args.score or args.predict_perf):
        print("ERROR: one of --train or --predict or --score should be specified...")
        exit(0)
    if args.train:
        #Check train args exist
        if not (args.batch_size is not None and args.train_file is not None and args.valid_file is not None and args.output_model is not None):
            print("Error: missing training arguments...")
            exit(0)
        #Check if load models
        if args.load_model is not None:
            if args.load_model is None:
                print("Error: missing load model training arguments...")
                exit(0)
            if args.cont: #continue training
                if path.exists(args.output_model) == False:
                    print("Error: when using continue training, the output_model dir should exist...")
                    exit(0)
            print("Try to load model [%s]", args.load_model)
            model = torch.load(args.load_model)
            print("Successfully load model")
        if args.load_optim is not None:
            print("Try to load optim [%s]", args.load_optim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            optimizer.load_state_dict(torch.load(args.load_optim))
            print("Successfully load optim")
            optimizer.lr = args.lr 
        #Load squad
        train_loader = make_dataloader(args.train_file, False, args.batch_size, args.train_step_num)
        valid_loader = make_dataloader(args.valid_file, False, args.batch_size )
        train_loader_1 = make_dataloader(args.train_file_1, True, args.batch_size_1, args.train_step_num , args.train_file_1_num )
        valid_loader_1 = make_dataloader(args.valid_file_1, False, args.batch_size_1)
        train_loader_2 = make_dataloader(args.train_file_2, True, args.batch_size, args.train_step_num)
        valid_loader_2 = make_dataloader(args.valid_file_2, False, args.batch_size )
        #set model args
        if args.load_model is None:
            os.mkdir(args.output_model, 0o755)
            print("create model dir {}".format(args.output_model))
            print("Set model args...")
            model = BertQA()
            model.to(device)


        loss_func = nn.CrossEntropyLoss().to(device)
        loss_func_1 = nn.MSELoss().to(device)
        
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
        BEER_STR=["appearance","aroma","palate","taste"] #don't add total here yet
        global BEER_IDX
        BEER_IDX = []
        for aspect in BEER_STR:
            BEER_IDX.append(tokenizer.encode(aspect, add_special_tokens=False)[0])
        print("BEER_IDX: ",BEER_IDX)
        beer_loss_mul = args.beer_loss_mul
        movie_loss_mul = args.movie_loss_mul
        print("Starting Training of {} model".format("ALbertMerge"))
        if args.load_optim is None:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
        train(model, train_loader, valid_loader,train_loader_1, valid_loader_1,train_loader_2, valid_loader_2,epoch_num=args.epoch_num, start_epoch=args.start_epoch, loss_func=loss_func, loss_func_1=loss_func_1, optimizer=optimizer, batch_size = args.batch_size, output_model = args.output_model, train_step_num = args.train_step_num)
    if args.predict:
        test_file=args.test_file
        test_data = load_data(test_file)
        test_dataset = QADataset(data = test_data)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn = collate_fn)
        want_len = args.want_len
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
        model = torch.load(args.predict_model)
        predict(model, test_loader, args.predict_output, tokenizer, args.predict_type)
    if args.predict_perf:
        if args.predict_type=="BEER":
            with open(args.perf_file,"rb") as F:
                ans_dict=pickle.load(F)
        else:
            ans_dict=None
        test_file=args.test_file
        test_data = load_data(test_file)
        test_dataset = QADataset(data = test_data)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn = collate_fn)
        model = torch.load(args.predict_model)
        predict_perf(model, test_loader, args.predict_type, ans_dict, args.beer_index)
    if args.score:
        raw_data_file=args.raw_data_file
        if args.score_movie:
            with open(raw_data_file,"rb") as F:
                raw_data=pickle.load(F)
        else:
            with open(raw_data_file,"r") as F:
                for i in F:
                    raw_data=json.loads(i)
        if args.predict:
            predict_file = args.predict_output
        else:
            predict_file = args.predict_file
        with open(predict_file,"r") as F:
            for i in F:
                predict_data=json.loads(i)
        if args.score_movie:
            evaluate_movie(predict_data, raw_data)
        else:
            evaluate(predict_data,raw_data)








