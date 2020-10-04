import torch
import pickle
import json
import sys
from transformers import BertTokenizer, AlbertTokenizer
from argparse import ArgumentParser
from tqdm import tqdm

tokenizer = None
MAX_LENGTH = 384



#Question = tokenizer.tokenize(Question)

Question='What is this beer appearance score?'


def process_input(context, question=Question):
    context = tokenizer.tokenize(context)
    question = tokenizer.tokenize(question)

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



all_features = ['appearance', 'aroma', 'palate', 'taste', 'total']
def get_type_id(feature):
    for idx, f in enumerate(all_features):
        if feature==f:
            return idx
    return -1


global id_cnt
id_cnt = 0

def load_file(file_name):
    with open(file_name, 'r') as f:
        qas = []
        for line in tqdm(f.readlines()):
            if line=='\n':
                continue
            line = line.split('\t')
            scores = line[0]
            scores = scores.split(' ')
            scores = [ float(i) for i in scores]
            rates = list(map(lambda x: float(x)*10, scores))

            line = line[1]
            
            #line = tokenizer.tokenize(line)
            #print(scores)
            #print(line)

            type_id = get_type_id(args.feature)
            if type_id == -1:
                print('feature error')
                exit(1)

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
                    'answer_able': 1,
                    'token_type_ids': token_type_ids,
                    'answers': scores
                    #'answers': answers
                }
                qas.append(dat)
                id_cnt += 1
            '''
            else:
                now_id = str(id_cnt)
                bert_question = 'Why is this beer rate ' + pos_or_neg + '?'

                tmp_input = []
                ans_start = 0
                ans_end = 0
                token_line = tokenizer.tokenize(line)
                for i, tok in enumerate(token_line):
                    tmp_input.append(tok)
                    answers = []
                    #print('tok', tok)
                    #if tok == ('<POS>' if pos_or_neg=='positive' else '<NEG>'):
                    #    ans_start = i
                    #if tok == ('</POS>' if pos_or_neg=='positive' else '</NEG>'):
                    ans_end = i
                    text, token_type_ids, attn_mask = process_input(line, bert_question)
                    answers.append([[ans_start, ans_end]])
                    dat = {
                           'q_id': now_id,
                            'text': text,
                            'attn_mask': attn_mask,
                            'answerable': 1,
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
    if args.tokenizer=="bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer=="albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
    elif args.tokenizer=="albertx":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    tokenizer.add_tokens(["<POS>"])
    tokenizer.add_tokens(["</POS>"])
    tokenizer.add_tokens(["<NEG>"])
    tokenizer.add_tokens(["</NEG>"])
    data = load_file(args.input)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input a beer data", type=str)
    parser.add_argument("--output", type=str, help="output file name")
    parser.add_argument("--tokenizer", type=str, help="using tokenizer, bert or albert")
    parser.add_argument("--task", help="bert or other", type=str, default='other')
    parser.add_argument("--feature", help="list of appearance/aroma/palate/taste/total, seperate by space", type=str, default='appearance')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_argument()
    main(args)









