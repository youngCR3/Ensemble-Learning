"""
Data: 2021/01/13
Author: Yang Zifeng
Descirption: 实现决策树的ID3, C4.5生成算法和决策树的剪枝
"""

import pandas as pd
import numpy as np
from math import log2
import random
from enum import Enum

class Algorithm(Enum):
    """
    两种决策树构建算法, 分别为ID3和ID45
    """
    ID3 = 1
    ID45 = 2


class Selection(Enum):
    """
    两种特征选择方法, 选择最优特征或随机选择
    """
    BEST = 1
    RANDOM = 2


class TreeNode:
    def __init__(self):
        """
        TreeNode为决策树的节点, 保存该节点的最优特征序号A, 该节点的数据集data和label, category代表该节点所属分类, 
        feature表示该点在A特征上的取值, child为子节点
        """
        self.A = None
        self.data = None
        self.label = None
        self.category = None
        self.feature = None
        self.child = []


class DecisionTree:
    def __init__(self, algorithm=Algorithm.ID45, selection=Selection.BEST):
        self.selection = selection
        self.algorithm = algorithm
        self.trainData = None
        self.trainLabel = None
        self.testData = None
        self.testLabel = None
        self.features = None
        self.tree = None

    def loadDataSet(self, train: str, test: str) -> None:
        """
        从path读取数据集文件, 将其划分为特征和标签并保存
        """
        # 读取训练集
        data = pd.read_excel(train)
        data = np.array(data)
        label = data[:, -1].tolist()
        data = data[:, :-1].tolist()
        self.trainData = data
        self.trainLabel = label
        self.features = list(range(len(data[0])))
        # 读取测试集
        data = pd.read_excel(test)
        data = np.array(data)
        label = data[:, -1].tolist()
        data = data[:, :-1].tolist()
        self.testData = data
        self.testLabel = label
        # 数据预处理
        self.dataProcess()
        
    def dataProcess(self):
        """
        数据预处理, 取出数据集中含未知数据的样本
        """
        data = []
        label = []
        for i in range(len(self.trainData)):
            for j in self.trainData[i]:
                if j == ' ?':
                    break
            else:
                data.append(self.trainData[i])
                label.append(self.trainLabel[i])
        self.trainData = data.copy()
        self.trainLabel = label.copy()
        data = []
        label = []
        for i in range(len(self.testData)):
            for j in self.testData[i]:
                if j == ' ?':
                    break
            else:
                data.append(self.testData[i])
                label.append(self.testLabel[i])
        self.testData = data.copy()
        self.testLabel = label.copy()

    def getEntropy(self, label: []) -> float:
        """
        计算数据集的熵, 详见公式(5.7)
        Args:
            label([]): 数据集的标签
        """ 
        C = {}
        N = len(label)
        for i in label:
            if i not in C:
                C[i] = 1
            else:
                C[i] += 1
        entropy = 0
        for i in C:
            entropy -= C[i] / N * log2(C[i] / N)
        return entropy
    
    def getConditionalEntropy(self, data: [], label: [], k: int) -> float:
        """
        计算根据第k个特征划分数据集时求得的条件熵, 详见公式(5.8)
        """
        cnt = {}                # cnt[i][j]表示特征k取值为i且属于第j类的样本的个数
        total = {}              # total[i]表示特征k取值为i的样本数量
        for i in range(len(data)):
            if data[i][k] not in cnt:
                cnt[data[i][k]] = {}
                total[data[i][k]] = 0
            total[data[i][k]] += 1
            if label[i] not in cnt[data[i][k]]:
                cnt[data[i][k]][label[i]] = 1
            else:
                cnt[data[i][k]][label[i]] += 1
        conditionalEntropy = 0
        for i in cnt:
            tmp = 0
            for j in cnt[i]:
                tmp += (cnt[i][j] / total[i]) * log2(cnt[i][j] / total[i])
            tmp /= len(data)
            conditionalEntropy -= tmp
        return conditionalEntropy
    
    def getInformationGain(self, k: int, data: [], label: []) -> float:
        """
        计算训练集D针对第k个特征的信息增益, 详见公式(5.9)
        """
        H_D = self.getEntropy(label)
        H_D_A = self.getConditionalEntropy(data, label, k)
        return H_D - H_D_A
    
    def getInformationGainRatio(self, k: int, data: [], label: []) -> float:
        """
        计算训练集D针对第k个特征的信息增益比, 详见公式(5.10)
        """
        g = self.getInformationGain(k, data, label)
        H_A_D = self.getH_A_D(k, data)
        if H_A_D == 0:
            return float('inf')
        return g / H_A_D
    
    def getH_A_D(self, k: int, data: []) -> float:
        """
        对于数据集D, 给定特征A, 计算D关于A的值的熵H_A(D), 详见公式(5.10)
        """
        cnt = {}
        for i in data:
            cnt[i[k]] = cnt[i[k]] + 1 if i[k] in cnt else 1
        N = len(data)
        H_A_D = 0
        for i in cnt:
            H_A_D -= cnt[i] / N * log2(cnt[i] / N)
        return H_A_D

    def featureSelection(self, features: [], data: [], label: []) -> (int, float):
        """
        针对数据集D， 从features中选择信息增益最大的特征作为最优特征
        """
        bestFeature = []
        maxGain = 0
        for i in features:
            if self.algorithm == Algorithm.ID45:
                gain = self.getInformationGainRatio(i, data, label)
            elif self.algorithm == Algorithm.ID3:
                gain = self.getInformationGain(i, data, label)
            if gain > maxGain:
                maxGain = gain
                bestFeature = [i]
            elif gain == maxGain:
                bestFeature.append(i)
        bestFeature = random.choice(bestFeature)        # 若有多个特征其信息增益一致, 则随机选择一个
        return bestFeature, maxGain
    
    def featureSelectionRand(self, features: [], data: [], label: []) -> (int, float):
        """
        随机选取一个特征
        """
        k = random.choice(features)
        gain = self.getInformationGainRatio(k, data, label)
        return k, gain

    def buildTree(self, data: [], label: [], features: [], threshold: float) -> TreeNode:
        """
        递归地生成决策树
        Args:
            data([]): 数据集特征
            label([]): 数据集标签
            features([]): 可选择的特征
            threshold(float): 作为单结点数返回的阈值
        """
        # 数据集为空, 直接返回None
        if not data:
            return None
        node = TreeNode()
        node.data = data.copy()
        node.label = label.copy()
        # 停止条件1: 若所有实例都属于同一类, 则返回单节点树, 且该类即为节点的类标记
        if len(set(label)) == 1:
            node.category = label[0]
            return node
        # 统计实例的类标记数
        cnt = {}
        maxCount = 0
        maxLabel = []
        for i in label:
            cnt[i] = cnt[i] + 1 if i in cnt else 1
            if cnt[i] > maxCount:
                maxCount = cnt[i]
                maxLabel = [i]
            elif cnt[i] == maxCount:
                maxLabel.append(i)
        maxLabel = random.choice(maxLabel)  # 若有多个最多的类, 随机选择一个作为最多的类标记
        node.category = maxLabel
        # 停止条件2: 若可选特征为空, 将D中实例最多的类作为该节点的类标记, 返回单节点数
        if not features:
            return node
        # 计算各特征对D的信息增益, 选择信息增益最大的特征作为最优特征
        if self.selection == Selection.BEST:
            k, informationGain = self.featureSelection(features, data, label)
        else:
            k, informationGain = self.featureSelectionRand(features, data, label)
        node.A = k
        # 停止条件3: 若最优特征的信息增益小于阈值, 则返回单节点数
        if informationGain < threshold:
            return node
        # 否则, 对最优特征的每一可能值将数据集D分割为若干子集Di, 将Di中实例数最大的类作为结点标记, 递归地构建子节点
        newData = {}
        newLabel = {}
        for i in range(len(data)):
            if data[i][k] not in newData:
                newData[data[i][k]] = []
                newLabel[data[i][k]] = []
            newData[data[i][k]].append(data[i])
            newLabel[data[i][k]].append(label[i])
        features.remove(k)
        for i in newData:
            child = self.buildTree(newData[i], newLabel[i], features.copy(), threshold)
            if child:
                child.feature = i
                node.child.append(child)
        return node

    def cutBranch(self, node: TreeNode, alpha: int) -> (TreeNode, float, int):
        """
        对决策树进行剪枝, 具体地, 当一组叶节点回缩到父节点后的损失函数变小或不变, 则进行剪枝, 详见公式(5.15)
        基于DP思想, 在每个节点处比较在此处剪枝前后的loss, 若更小或不变则剪枝, 相当于只对决策树进行一遍DFS
        Args:
            node (TreeNode): 当前树的根节点
            alpha (float): 权重系数, =0相当于不考虑模型复杂度, 只考虑对训练集数据的拟合程度
        Returns: 
            node (TreeNode): 当前树经剪枝后的树根
            loss (float): 当前树经剪枝后的局部loss
            cut (int): 剪枝次数
        """
        
        if not node:
            return node, 0, 0
        loss = 0
        cnt = {}                # 计算当前结点作为叶节点时候的局部loss
        for i in node.label:
            cnt[i] = cnt[i] + 1 if i in cnt else 1
        for i in cnt:
            p = cnt[i] / len(node.label)
            loss -= p * log2(p)
        loss += alpha
        if not node.child:      # 该结点本身即为叶节点
            return node, loss, 0
        else:                   # 对比在该节点进行剪枝前后的loss
            childLoss = 0
            newChild = []       # 子节点经剪枝后得到newChild
            cut = 0
            for i in node.child:
                child, loss1, c = self.cutBranch(i, alpha)
                if child:
                    newChild.append(child)
                    childLoss += loss1
                cut += c
            if loss <= childLoss:   # 比较childLoss和当前结点称为叶节点后的loss, 若后者不大于childLoss则进行剪枝
                return node, loss, cut + 1
            else:
                node.child = newChild
                return node, childLoss, cut

    def predict(self, data: [], node: TreeNode):
        """
        利用训练好的决策树进行分类, 返回分类结果
        """
        if not node.child:
            return node.category
        feature = data[node.A]
        for i in node.child:
            if i.feature == feature:
                return self.predict(data, i)
        else:
            return node.category
    
    def printTree(self):             
        """
        按层序遍历打印决策树
        """
        queue = [(self.tree, 0)]
        head = 0
        while head < len(queue):
            node, depth = queue[head]
            if head > 0 and depth > queue[head - 1][1]:
                print()
                print('-----------------------------------------------------')
            for i in node.child:
                queue.append((i, depth + 1))
            head += 1
            print(node.A, node.feature, node.category, end=' ')
        print()

    def train(self, node: TreeNode) -> (int, int):
        """
        计算决策树训练准确率
        Returns:
            total (int): 总样本数
            correct (int): 分类正确样本数
        """
        if not node:
            return 0, 0
        if not node.child:
            total = len(node.label)
            correct = 0
            for i in node.label:
                if i == node.category:
                    correct += 1
            return total, correct
        else:
            total = correct = 0
            for i in node.child:
                t, c = self.train(i)
                total += t
                correct += c
            return t, c

    def test(self, testData: [], testLabel: []) -> None:
        """
        利用测试集检验模型准确率
        """
        total = len(testLabel)
        correct = 0
        for i in range(len(testData)):
            label = self.predict(testData[i], self.tree)
            if label == testLabel[i]:
                correct += 1
        rate = correct / total * 100
        print('Num of total cases is', total, ', num of correct cases is', correct, ', test accuracy is', rate, '%')


if __name__ == "__main__":
    solver = DecisionTree(selection=Selection.BEST)
    solver.loadDataSet(r"adult\\train.xlsx", r"adult\\test.xlsx")
    solver.tree = solver.buildTree(solver.trainData, solver.trainLabel, solver.features, 0.2)
    _, _, cut = solver.cutBranch(solver.tree, 3)
    # solver.printTree()
    print('Total cut is', cut)
    t, c = solver.train(solver.tree)
    print('Train accuracy is', c / t * 100, '%')
    solver.test(solver.testData, solver.testLabel)

