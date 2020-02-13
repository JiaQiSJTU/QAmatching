import json
import operator
def load_data():
    result = []
    with open("./results/result.json",'r') as file: #pairwise预测结果路径
        data = json.load(file)
        # print(data[0])
        num=0
        tempT=[] #存放真实标签
        tempP=[] #存放预测的score
        for i in range(len(data)): #对于每一个NQ得到一个其对于之前每一个Q的真实标签序列和score序列
            if i==0:
                num+=1
                tempT.append(data[i]["gold_label"])
                tempP.append(data[i]["T_prob"])
                continue
            if(data[i]['sessionID']==data[i-1]['sessionID'] and data[i]['sentence2ID']==data[i-1]['sentence2ID']):
                if(data[i]['sentence1ID']>data[i-1]['sentence1ID']):
                    num+=1
                    tempT.append(data[i]['gold_label'])
                    tempP.append(data[i]['T_prob'])
                else:
                    print("error 乱序")
                    exit(0)
            else:
                result.append([tempT,tempP])
                tempT=[]
                tempP=[]
                num+=1
                tempT.append(data[i]["gold_label"])
                tempP.append(data[i]["T_prob"])
        result.append([tempT, tempP])
    print(num)
    return result
def processPredictions(threathold):
    data = load_data()
    result=[]
    checknum=0
    for item in data:

        length=len(item[1])
        checknum+=length
        maximum=max(item[1])
        maxid=item[1].index(max(item[1]))
        if(maximum>threathold):
            item[1]=[0]*length
            item[1][maxid]=1
        else:
            item[1] = [0 ]* length
        result.append([item[0],item[1]])
    print(checknum)
    return result #对于每一个NQ,返回两个list，第一个是预测的trulabel，第二个是predict的最多含一个1的label





if __name__ == '__main__':

    resulta=processPredictions(0.5)
    with open("./results/result.json",'r') as file: #pairwise预测结果路径
        data = json.load(file)
        # print(data[0])
        count=0
        for item in resulta:
            for i in range(len(item[0])):
                data[count]['predict_label']=item[1][i]
                count+=1
        print(len(data))
        print(count)
        json_str = json.dumps(data)
        with open("./results/result-1.json", 'w') as json_file: #pairwise0/1预测结果
            json_file.write(json_str)







