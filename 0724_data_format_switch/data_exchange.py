import os
import json
import uuid
import random
import numpy as np

def generate_unique_id(): 
    unique_id = str(uuid.uuid4()) 
    return unique_id

def multi_format(prompt,response,label):
    content = {'prompt':prompt,'input':'','response':[response , label]}
    return content

def standard_format(id,data,domain):
    sample =  {'id':id,'data':data,'domain':[domain]}
    return sample



def multi_format_activate(id,multi_round_bucket,domain):
    round_scond = [multi_round_bucket[-2],multi_round_bucket[-1]]
    round_third =  [multi_round_bucket[-3],multi_round_bucket[-2],multi_round_bucket[-1]]
    round_fourth =  [multi_round_bucket[-4],multi_round_bucket[-3],multi_round_bucket[-2],multi_round_bucket[-1]]

    round_scond_sample = standard_format(id,round_scond,domain)
    round_third_sample = standard_format(id,round_third,domain)
    round_fourth_sample = standard_format(id,round_fourth,domain)



    return round_scond_sample, round_third_sample,round_fourth_sample


# 数据层
def data_bucket():
    jiaoyanyue = "/mnt/pfs/jinfeng_team/SFT/dengshuhao1/data/DataForSFT/Multiple_rounds_dialogue_data/internal/jiaoyanyun/head_5w/jiaoyanyun_all.jsonl"
    math401 = "/mnt/pfs/jinfeng_team/SFT/dengshuhao1/data/DataForSFT/Multiple_rounds_dialogue_data/testsets/math401_ceval_cot_datasets/math401/math401.json" 
    ceval = "/mnt/pfs/jinfeng_team/SFT/dengshuhao1/data/DataForSFT/Multiple_rounds_dialogue_data/testsets/math401_ceval_cot_datasets/C-Eval/C-Eval-no-shot_v2_middle_school_mathematics.json"
    fr_jiaoyanyue= open(jiaoyanyue).readlines()
    fr_math = open(math401).readlines()
    fr_ceval = open(ceval).readlines()
    return fr_ceval,fr_math,fr_jiaoyanyue





def math401_multi_round_prepare(fr_ceval, fr_math,fr_jiaoyanyue,output_file):
    
    fw_second = open(output_file.split('.')[0] + "_2.jsonl",'w')
    fw_third = open(output_file.split('.')[0] + "_3.jsonl",'w')
    fw_fourth = open(output_file.split('.')[0] + "_4.jsonl",'w')
  


    fw = open(output_file,'w')


    for l in  range(len(fr_math)):
       sample =  json.loads(fr_math[l])
     

       task_lst = ['math401','ceval']

       # 防止重采样
       jiaoyanyun_ = np.arange(0,len(fr_jiaoyanyue))
       ceval_ = np.arange(0,len(fr_ceval))
       ceval_ = ceval_[ceval_ != l]
       
       # 轮数拼接

       multi_round_bucket = []

       for i in range(7):
           task = random.choice(task_lst)
        #    print(task)
           if task == "math401":
                count = np.random.choice(jiaoyanyun_)
                jiaoyanyun_ = jiaoyanyun_[jiaoyanyun_ != count]
                pro = json.loads(fr_jiaoyanyue[count])['data'][0]["prompt"]
                res = json.loads(fr_jiaoyanyue[count])['data'][0]["response"][0][0]
                label = json.loads(fr_jiaoyanyue[count])['data'][0]["response"][0][1]
                multi_round_bucket.append(multi_format(pro,res,label))
               
           elif task == "ceval":
               ceval_count = np.random.choice(ceval_)
               ceval_ = ceval_[ceval_ != ceval_count]
               pro = json.loads(fr_ceval[ceval_count])['prompt']
               res = json.loads(fr_ceval[ceval_count])['response']
               label = json.loads(fr_ceval[ceval_count])['domain']
               multi_round_bucket.append(multi_format(pro,res,label))

        # 准备最后一轮样本
       id = generate_unique_id()
       prompt = sample['query']
       response = sample['response']
       domain = "math401"
       multi_round_bucket.append(multi_format(prompt,response,domain))
       sample_fromat = standard_format(id,multi_round_bucket,domain)

       second_ , third_, fourth_ = multi_format_activate(id,multi_round_bucket,domain)
   

       fw_second.write(json.dumps(second_,ensure_ascii=False)+"\n")
       fw_third.write(json.dumps(third_,ensure_ascii=False)+"\n")
       fw_fourth.write(json.dumps(fourth_,ensure_ascii=False)+"\n")
       fw.write(json.dumps(sample_fromat,ensure_ascii=False)+"\n")

    fw.close()
    fw_second.close()
    fw_third.close()
    fw_fourth.close()



def ceval_multi_round_prepare(fr_ceval, fr_jiaoyanyue,output_file):
    
    fw_second = open(output_file.split('.')[0] + "_2.jsonl",'w')
    fw_third = open(output_file.split('.')[0] + "_3.jsonl",'w')
    fw_fourth = open(output_file.split('.')[0] + "_4.jsonl",'w')
  


    fw = open(output_file,'w')

    for l in  range(len(fr_ceval)):
       sample =  json.loads(fr_ceval[l])
     

       task_lst = ['math401','ceval']

       # 防止重采样
       jiaoyanyun_ = np.arange(0,len(fr_jiaoyanyue))
       ceval_ = np.arange(0,len(fr_ceval))
       ceval_ = ceval_[ceval_ != l]
       
       # 轮数拼接

       multi_round_bucket = []

       for i in range(7):
           task = random.choice(task_lst)
        #    print(task)
           if task == "math401":
                count = np.random.choice(jiaoyanyun_)
                jiaoyanyun_ = jiaoyanyun_[jiaoyanyun_ != count]
                pro = json.loads(fr_jiaoyanyue[count])['data'][0]["prompt"]
                res = json.loads(fr_jiaoyanyue[count])['data'][0]["response"][0][0]
                label = json.loads(fr_jiaoyanyue[count])['data'][0]["response"][0][1]
                multi_round_bucket.append(multi_format(pro,res,label))
               
           elif task == "ceval":
               ceval_count = np.random.choice(ceval_)
               ceval_ = ceval_[ceval_ != ceval_count]
               pro = json.loads(fr_ceval[ceval_count])['prompt']
               res = json.loads(fr_ceval[ceval_count])['response']
               label = json.loads(fr_ceval[ceval_count])['domain']
               multi_round_bucket.append(multi_format(pro,res,label))

        # 准备最后一轮样本
       id = sample['id']
       prompt = sample['prompt']
       response = sample['response']
       domain = sample['domain']
       choice = sample['choice']
       multi_round_bucket.append(multi_format(prompt,response,domain))
       sample_fromat = standard_format(id,multi_round_bucket,domain)
       sample_fromat['choice']= choice
       second_ , third_, fourth_ = multi_format_activate(id,multi_round_bucket,domain)
       second_['choice']= choice
       third_['choice']= choice
       fourth_['choice']= choice

       fw_second.write(json.dumps(second_,ensure_ascii=False)+"\n")
       fw_third.write(json.dumps(third_,ensure_ascii=False)+"\n")
       fw_fourth.write(json.dumps(fourth_,ensure_ascii=False)+"\n")
       fw.write(json.dumps(sample_fromat,ensure_ascii=False)+"\n")

    fw.close()
    fw_second.close()
    fw_third.close()
    fw_fourth.close()





    



if __name__ == "__main__":

    

    # 数据准备
    fr_ceval,fr_math,fr_jiaoyanyue = data_bucket()

    print(len(fr_ceval),len(fr_math),len(fr_jiaoyanyue))

    flag = 'math401'
    if flag == 'ceval':
        output_file = "/mnt/pfs/jinfeng_team/SFT/dengshuhao1/wzd_folder/exp/0724_report_validation_v3/ceval/ceavl_val_v3.jsonl"
        ceval_multi_round_prepare( fr_ceval,fr_jiaoyanyue,output_file)
    elif flag == "math401":
        output_file = "/mnt/pfs/jinfeng_team/SFT/dengshuhao1/wzd_folder/exp/0724_report_validation_v3/tmp/math401_test.jsonl"
        math401_multi_round_prepare(fr_ceval, fr_math,fr_jiaoyanyue,output_file)




