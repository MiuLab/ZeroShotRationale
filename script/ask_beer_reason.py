import csv
import json
import string
import random
import sys
import re


data_path = 'data/annotations.json'
target_path = sys.argv[1]
ask_aspect = int(sys.argv[2])
random_list = False


# aspect
# appearance, aroma, palate, taste, total


def get_question(score, aspect):
    #nowq = 'Why does this beer get '+str(score)+' points in the aspect of '+str(aspect)+'?'
    #nowq = 'Why is this comment related to '+str(aspect)+'?'
    nowq = 'What is this beer '+str(aspect)+' score?'
    return nowq

aspect_list = ['appearance', 'aroma', 'palate', 'taste', 'overall']
    
data = []


root = {}

qid = 0
qid_check=[]


cnt = 0
data_num = int(sys.argv[3])

with open(data_path, 'r') as jsonFile:
    for lines in jsonFile:
        if lines is None:
            break
        if cnt >= data_num:
            break
        jrow = json.loads(lines)
        
        raw = json.dumps(eval(jrow['raw']))
        raw = json.loads(raw)
        score = jrow['y']    
        # print(raw.keys())
        #/input()
        comments = {}
        ###### the title 
        article = {"title": 'Data2_Reason'}
        paragraphs = []
        paragraph_num = 1
               
        
        for i in range(paragraph_num):
            context = {"context": raw['review/text']}
            
            qas = []
            # each annotation -> one QA.
                        
            index = ask_aspect
            now_answers = []
            now_qa = {}
            now_answer = [-1,-1]
            for one_answer in jrow[str(index)] :
                if now_answer[0]==-1:
                    now_answer[0] = one_answer[0]
                now_answer[1] = one_answer[1]
            tmp_list = jrow['x'][now_answer[0]:now_answer[1]]

            tmp_str = (' ').join(str(x) for x in tmp_list)
            total_str = (' ').join(str(x) for x in jrow['x'])


            #print('tmp_str', tmp_str)
            
            now_ans = {}

            # ANS start
            now_ans['answer_start'] = total_str.find(tmp_str)
            now_ans['text'] = tmp_str

            now_answers.append(now_ans)
            # print('now_answers', now_answers)

            now_qa['answers'] = now_answers
            now_qa['id'] = str(qid)
            qid_check.append(qid)
            now_qa['is_impossible'] = False
            nowas = aspect_list[index]
            # print(nowas)
            now_qa['question'] = get_question(int(score[index]*10), nowas)
            # print(now_qa['question'])
            qas.append(now_qa)
            qid += 1
                
            context['qas'] = qas

            paragraphs.append(context)
    
        article['paragraphs'] = paragraphs
        
        # print(article)
        data.append(article)
        
        cnt += 1
            
if random_list == True:
    random.shuffle(data)


root = {'data': data, 'version': '2.2'}
with open(target_path, 'w') as output_file:
    json.dump(root, output_file)

all_right = True    
qid_list = [data[i]['paragraphs'][0]['qas'][0]['id'] for i in range(994)]
for i in range(994):
    if str(i) not in qid_list:
        print("ERROR:",i)
        all_right = False
if all_right == True:
    print("CLEAR!")


