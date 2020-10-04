import torch
import pickle
import json
import sys
from transformers import BertTokenizer
from transformers import AlbertTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import csv
import random


MAX_LENGTH = 384
tokenizer = None

cnt_over = 0

#Question = tokenizer.tokenize(Question)

Question='How is this movie rated, positive or negtive?'

def process_input(context, question=Question):
    global cnt_over
    if type(context) == str:
        context = tokenizer.tokenize(context)
    question = tokenizer.tokenize(question)
    if len(context)+len(question)+3 > 384:
        cnt_over+=1

    text = ['[CLS]'] + context
    preserve_len = len(question) + 2
    text = text[:MAX_LENGTH - preserve_len]
    
    text += ['[SEP]']
    segment = [1]*len(text)

    text += question
    text += ['[SEP]']

    segment += [0]*(len(text) - len(segment))
    mask = [1] * len(segment)

    pad_len = max(0, MAX_LENGTH - len(text))
    pad_str = tokenizer.decode([0])
    text += [pad_str] * pad_len
    segment += [0] * pad_len
    mask += [0] * pad_len

    text = tokenizer.convert_tokens_to_ids(text)
    assert len(text) == len(segment)
    assert len(mask) == len(segment)
    assert len(text) == len(mask)

    return torch.tensor(text, dtype=torch.long), torch.tensor(segment, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)



def find_ans(tokens, start_tag, end_tag):
    idx = 0
    reline = []
    answers = []
    #print('tokens',tokens, start_tag)
    
    tmp_start = -1
    tmp_end = -1

    for tok in tokens:
        if tok==start_tag:
            tmp_start = idx
            #print("!!")
        elif tok==end_tag:
            tmp_end = idx
            answers.append([tmp_start, tmp_end])
        else:
            reline.append(tok)
            idx += 1
    #print(answers)
    #input()
    #print(reline[answers[0][0]:answers[0][1]])
    return reline, answers







global id_cnt
id_cnt = 0
pos_or_neg = ''

def load_file(file_name):
    with open(file_name, newline="") as csvfile:
        rows = csv.reader(csvfile)
        qas = []
        first_line=True
        for row in tqdm(rows):
            if first_line:
                #skip first line, which is the column name of csv file
                first_line=False
                continue
            line = row[0].replace("<br />","") # there's some '<br />' in context, delete it
            sent = row[1]
            if sent=="positive":
                pos_or_neg = 1
            elif sent=="negative":
                pos_or_neg = 0
            else:
                print("ERROR")
                exit(0)


            global id_cnt
            if args.task != 'bert':
                text, token_type_ids, attn_mask = process_input(line)
                now_id = str(id_cnt)
                id_cnt += 1
                #answers = []
                dat = {
                    'q_id': now_id,
                    'text': text,
                    'attn_mask': attn_mask,
                    'is_impossible': False,
                    'token_type_ids': token_type_ids,
                    'answers': pos_or_neg,
                    #'answers': answers
                }
                qas.append(dat)
                id_cnt += 1

            else:
                #print('BERT')
                now_id = str(id_cnt)
                bert_question = 'Why is this movie ' + 'positive' if pos_or_neg == 1 else 'negtive' + '?'
                
                start_tag = '<POS>' if pos_or_neg == 1 else '<NEG>'
                end_tag = '</POS>' if pos_or_neg == 1 else '</NEG>'

                token_line = tokenizer.tokenize(line)
                line, answers = find_ans(token_line, start_tag, end_tag)
                for anss in answers:
                    text, token_type_ids, attn_mask = process_input(line, bert_question)
                    now_id = str(id_cnt)
                    dat = {
                            'q_id': now_id,
                            'text': text,
                            'attn_mask': attn_mask,
                            'is_impossible': False,
                            'token_type_ids': token_type_ids,
                            'answers': answers
                    }
                    id_cnt += 1
                    qas.append(dat)
                    '''
                    if tok == ('<POS>' if pos_or_neg=='positive' else '<NEG>'):
                        del token_line[i]
                        ans_start = i

                    if tok == ('</POS>' if pos_or_neg=='positive' else '</NEG>'):
                        ans_end = i
                        text, token_type_ids, attn_mask = process_input(line, bert_question)
                        answers.append([[ans_start, ans_end]])
                        dat = {
                            'q_id': now_id,
                            'text': text,
                            'attn_mask': attn_mask,
                            'is_impossible': False,
                            'token_type_ids': token_type_ids,
                            'answers': answers
                        }
                        tmp_input = []
                        qas.append(dat)
                        id_cnt += 1
                    '''

        print('qas_len:', len(qas))
        return qas

def main(args):
    global tokenizer
    if args.tokenizer == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer =="albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
    elif args.tokenizer =="albertx":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')

    data = load_file(args.input)
    random.shuffle(data)
    print("Total_num: ",len(data))
    print("Over: ",cnt_over)
    with open(args.output_t, 'wb') as f:
        pickle.dump(data[:-args.valid_num], f)
    with open(args.output_v, 'wb') as f:
        pickle.dump(data[-args.valid_num:], f)



def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input a movie data", type=str)
    parser.add_argument("--output_t", type=str, help="output train file name")
    parser.add_argument("--output_v", type=str, help="output valid file name")
    parser.add_argument("--valid_num", type=int, help="output valid data_num")
    parser.add_argument("--task", help="bert or other", type=str, default='other')
    parser.add_argument("--tokenizer", help="bert or albert", type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_argument()
    main(args)










