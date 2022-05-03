from sklearn.ensemble import RandomForestClassifier
import numpy as np
from math import log

class Domain:
    def __init__(self, _name, _truename,_label, _length, _num, _entropy):
        self.name = _name
        self.truename=_truename
        self.label = _label
        self.length = _length
        self.num = _num
        self.entropy = _entropy
    
    def returnData(self):
        return [self.length, self.num, self.entropy]
 
    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def cal_num(str):
    num = 0
    for i in str:
        if i.isdigit():
            num += 1
    return num

def getentropy(anArray):

    entropy = 0.0                     

    feature_dict = {}                  
    for i in list(anArray):
        feature_dict.update({i:None,})
    
   
    for i in feature_dict:                              
        ct = list(anArray).count(i)                     
        pi = ct / len(anArray)                          
        if pi == 0:                                     
            entropy = entropy - 1
        else:
            entropy = entropy - pi * log(pi,2)          
    return entropy


def initData(filename,domainlist):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                 continue
            tokens = line.split(",")
            name = tokens[0]
            if len(tokens) > 1:
                label = tokens[1]
            else:
                label = "?"

            realname = name.split(".")
            truename=realname[0]
            length = len(truename)
            num = cal_num(truename)
            entropy = getentropy(truename)
            domainlist.append(Domain(name,truename, label, length, num, entropy))

def main():
    domainlist1 = []
    initData("train.txt",domainlist1)
    featureMatrix = []
    labelList = []
    print("Training...")
    for item in domainlist1:
         featureMatrix.append(item.returnData())
         labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state = 0)
    clf.fit(featureMatrix,labelList)

    domainlist2 = []
    initData("test.txt",domainlist2)
    print("Judging...")
    with open("result.txt","w") as f:
        for i in domainlist2:
            f.write(i.name)
            f.write(",")
            if clf.predict([i.returnData()])[0] == 0:
                f.write("notdga")
            else:
                f.write("dga")
            f.write("\n")
    print("Completed")

if __name__ == '__main__':
    main()
