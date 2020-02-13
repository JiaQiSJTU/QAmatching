import json
import operator
def load_original(originalfile):
    with open(originalfile,'r') as fh:
        source=json.load(fh)
    return source

def load_predict():
    with open("./results/result-1.json",'r') as file: #输入pairwise0/1预测结果
        data = json.load(file)
    return data

if __name__ == '__main__':
    outputfile=open('./results/result.txt','a+',encoding='utf-8') #输出对话：预测、真实、角色、语句
    outputfile2=open('./results/result-QAlabel.json','a+',encoding='utf-8') #输出预测的QAlabel，每个对话一行
    originalPath = './datafile/test-200.json'
    source = load_original(originalPath)
    predict=load_predict()
    dialogList = source["data"]
    for dialogNum,dialog in enumerate(dialogList):
        QAlabel=dialog['label']
        roleLabel=dialog['role']
        sentences=dialog['sent']
        Prlabel=['O']*len(QAlabel)
        for i in range(len(QAlabel)):
            if 'Q' in QAlabel[i]:
                Prlabel[i]=QAlabel[i]
        for item in predict:
            if (item['sessionID']-1)==dialogNum:
                if item['predict_label']==1:
                    senAid=item['sentence2ID']
                    senQid=item['sentence1ID']
                    # print(QAlabel[senQid])
                    # print(Prlabel[senAid])
                    Prlabel[senAid]=QAlabel[senQid].replace('Q','A')
        print(dialogNum)
        outputfile.write('Dialog {}\n'.format(dialogNum))
        for i in range(len(QAlabel)):
            outputfile2.write(Prlabel[i]+' ')
            outputfile.write(Prlabel[i]+'\t'+QAlabel[i]+'\t'+roleLabel[i]+'\t'+sentences[i]+'\n')
        outputfile.write('\n')
        outputfile2.write('\n')













