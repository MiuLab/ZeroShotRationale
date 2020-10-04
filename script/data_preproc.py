import torch
from transformers import AlbertTokenizer
import json
import argparse
import pickle
import random
import itertools


def find_indexes(a_list, value):
    return [ i for i,x in enumerate(a_list) if x == value]
def find_answer_indexes_old(context_list, answer_list):
    start = -1
    end = -1
    for i in range(len(context_list)-len(answer_list)+1):
        if answer_list == context_list[i:i+len(answer_list)]:
            #print(answer_list,context_list[i:i+len(answer_list)])
            start = i
            end = i + len(answer_list)
    return start, end

def find_answer_indexes_new(context_list, answer_list, ans_start, context, tokenizer):
    start = len(tokenizer.encode(context[:ans_start],add_special_tokens = True))-1
    end = start + len(answer_list) - 1
    #check
    if answer_list != context_list[start:end+1]:
        start = -1
        end = -1
    return start, end

def make_data(tokenizer,data_list, is_predict):
    total_drop = 0
    bad_data_drop = 0
    drop_answers = 0
    repeat_answers = 0
    #make_data is a list of {q_id:"str", text:"str(context(truncated)+question), answer_able: true or false,answers:[{answer_start:int, answer_len}]"}
    make_data=[]
    #each para_list have some paragraph
    for para_list in data_list:
        #each para has one context and multiple question
        for para in para_list:
            para_qas=[]
            context_token = tokenizer.encode(para['context'], add_special_tokens = True)
            c_len = len(context_token)
            for qa in para['qas']:
                q_id = qa['id']
                question_token = tokenizer.encode(qa['question'], add_special_tokens = False)
                q_len = len(question_token)
                #calc max context len
                max_context_len = 382 - q_len
                answers=[]
                not_use = False
                good_data = True
                if c_len > max_context_len:
                    new_text = context_token[:max_context_len] + [102] + question_token + [102] 
                    if is_predict == False and "answers" in qa.keys() and qa['is_impossible'] == False:
                        answer_able = True
                        good_data = False
                        for ans in qa['answers']:
                            ans_token = tokenizer.encode(ans['text'], add_special_tokens = False)
                            #ans_start, ans_end = find_answer_indexes(context_token, ans_token)
                            ans_start, ans_end = find_answer_indexes_new(context_token, ans_token,ans['answer_start'], para['context'], tokenizer)
                            if ans_start != -1 and ans_end !=-1:
                                if [ans_start, ans_end] not in answers:
                                    answers.append([ans_start, ans_end])
                                    good_data = True
                                else:
                                    repeat_answers += 1
                            else:
                                drop_answers +=1
                                continue #can't find answer
                            if ans_end >= max_context_len:#been cut!
                                not_use = True
                                break
                    else: #unanswerable or testing
                        answer_able =False
                        answers = [[0,0]]
                else:#c_len is okay 
                    if is_predict == False and "answers" in qa.keys() and qa['is_impossible'] == False:
                        answer_able = True
                        good_data = False
                        for ans in qa['answers']:
                            ans_token = tokenizer.encode(ans['text'], add_special_tokens = False)
                            ans_start, ans_end = find_answer_indexes_new(context_token, ans_token,ans['answer_start'], para['context'], tokenizer)
                            if ans_start != -1 and ans_end !=-1:
                                if [ans_start, ans_end] not in answers:
                                    answers.append([ans_start, ans_end])
                                    good_data = True
                                else:
                                    repeat_answers += 1
                            else:
                                drop_answers +=1
                    else:
                        answer_able = False
                        answers = [[0,0]]
                    new_text = context_token + question_token + [102]
                #padding new text
                real_context_len = min(c_len, max_context_len+1)
                token_type_ids = torch.LongTensor([1]*real_context_len + [0]*(384 - real_context_len))
                attn_mask = torch.FloatTensor([1]*len(new_text) + [0] * (384-len(new_text)))
                new_text  = new_text + [0] * ( 384-len(new_text) )
                if len(new_text) != 384:
                    print("Error: context should be padding to length 384")
                if  not_use==False and good_data == True:
                    if len(answers) == 0:
                        print("Error: No answer...")
                        exit(0)
                    dat = {
                            "q_id" : q_id,
                            "text" : new_text,
                            "answer_able" : 1 if answer_able else -1,
                            "answers" : answers,
                            "attn_mask" : attn_mask,
                            "token_type_ids" : token_type_ids
                    }
                    para_qas.append(dat)
                else:
                    if not_use == True:
                        total_drop+=1
                    if good_data == False:
                        bad_data_drop += 1

            make_data.append(para_qas)
    #shuffle by context
    random.shuffle(make_data)
    make_data = list(itertools.chain(*make_data))
    print(len(make_data), total_drop, bad_data_drop)
    print("Cut answer drop: %d, Bad data drop : %d, Drop_rate:%f" %(total_drop, bad_data_drop, (total_drop + bad_data_drop) / (len(make_data)+total_drop+bad_data_drop)))
    print("Drop answers: ", drop_answers)
    print("Repeat answers: ", repeat_answers)

    return make_data


def main():
    # -------------------  parse args  -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="data file path",dest="data_file")
    parser.add_argument("--output_dir", help="output_directory", dest="output_dir")
    parser.add_argument("--data_tag", help="data tags, train, dev, valid, test", dest="data_tag")
    args = parser.parse_args()
    #Check lost or wrong arguments
    if args.data_file is None or args.output_dir is None or args.data_tag is None:
        print("Lost arguments... need --data_file, --output_dir, --data_tag specified")
        exit(0)
    if args.data_tag == "train" or args.data_tag == "valid":
        is_predict = False
    elif args.data_tag == "dev" or args.data_tag == "test":
        is_predict = True
    else:
        print("Error! --data_tag should be set to train, valid, dev or test")
        exit(0)
    with open(args.data_file,"r") as F:
        for i in F:
            data = json.loads(i)['data']
        F.close()
    data_list = ([i['paragraphs'] for i in data])
    tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
    tokenizer.do_lower_case = True
    new_data = make_data(tokenizer, data_list, is_predict)
    output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir + "/"
    with open(output_dir + args.data_tag+"_data.pickle", "wb") as F:
        pickle.dump(new_data, F)
        F.close()
    print("Done pre-making data")

main()
