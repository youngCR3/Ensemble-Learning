"""
Data: 2021/01/16
Author: Yang Zifeng
Descirption: 通过sklearn检验CART算法和GBDT算法
"""


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def load(path):
    for j in range(3):
        data = []
        label = []
        file = open(path[j])
        for line in file.readlines():
            line = line.strip()
            line = line.split(',')
            line = [float(i) for i in line]
            data.append(line[:-1])
            label.append(line[-1])
        N = int(len(data))
        data = data[:N]
        label = label[:N]
        # print(len(data))
        if j == 0:
            trainData = data
            trainLabel = label
        elif j == 1:
            validationData = data
            validationLabel = label
        else:
            testData = data
            testLabel = label
    return trainData, trainLabel, testData, testLabel

trainData, trainLabel, testData, testLabel = load([r'housing\\data.txt', r'housing\\validate.txt', r'housing\\test.txt'])
'''
CART算法
'''
# solver = DecisionTreeRegressor(criterion="mse", max_depth=10, min_samples_leaf=5, splitter="best")
'''
GBDT算法
'''
solver = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=4, learning_rate=0.2)
solver = solver.fit(trainData, trainLabel)
label = solver.predict(testData)
loss = rand = accurate = 0
ave = sum(testLabel) / len(testLabel)
for i in range(len(testData)):
    accurate += abs(testLabel[i] - label[i]) / testLabel[i]
    loss += (testLabel[i] - label[i]) ** 2
    rand += (testLabel[i] - ave) ** 2
print(1 - loss / rand)
print(accurate / len(testData))
