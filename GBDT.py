"""
Data: 2021/01/17
Author: Yang Zifeng
Descirption: 实现基于GBDT的提升树模型
"""

import pandas as pd
import math

class TreeNode:
    def __init__(self, j=-1, s=-1, c=0, data=[]):
        """
        Args:
            left(TreeNode): 左子树
            right(TreeNode): 左子树
            j(int): 划分的最优特征的索引
            s(float): 最优划分点
            c(float): 输出值
            data([]): 该区域的样本在数据集中的索引
        """
        self.left = None
        self.right = None
        self.j = j
        self.s = s
        self.c = c
        self.data = data
    
    def copy(self):
        node = TreeNode(self.j, self.s, self.c, self.data.copy())
        if self.left:
            node.left = self.left.copy()
        if self.right:
            node.right = self.right.copy()
        return node


class CART:
    def __init__(self, 
    trainData=None, trainLabel=None,
    testData=None, testLabel=None, 
    validationData=None, validationLabel=None, 
    maxDepth=10):
        self.tree = None
        self.testData = testData
        self.testLabel = testLabel
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.validationData = validationData
        self.validationLabel = validationLabel
        self.maxDepth = maxDepth

    def loadDataSet(self, path: str) -> None:
        """
        加载txt格式的数据集
        """   
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
                self.trainData = data
                self.trainLabel = label
            elif j == 1:
                self.validationData = data
                self.validationLabel = label
            else:
                self.testData = data
                self.testLabel = label
                self.testData.extend(self.validationData)
                self.testLabel.extend(self.validationLabel)
    
    def getCm(self, data: []) -> float:
        """
        计算区域m的输出值cm
        Args:
            data([]): 该区域的样本在数据集中的索引
        """
        if not data:
            return 0
        res = 0 
        for i in data:
            res += self.trainLabel[i]
        res /= len(data)
        return res
    
    def divide(self, data: [], j: int, s: float) -> ([], []):     
        """
        对于给定的数据集, 按第j个特征的划分点s划分
        Args:
            data([]): 该区域的样本在数据集中的索引
            j(int): 根据样本的第j个特征划分
            s(float): 划分点
        Returns:
            data1([]): 属于子区域1的样本在数据集中的索引
            data2([]): 属于子区域2的样本在数据集中的索引
        """
        if not data:
            return
        data1 = []
        data2 = []
        for i in data:
            if self.trainData[i][j] <= s:
                data1.append(i)
            else:
                data2.append(i)
        return data1, data2
    
    def getLoss(self, data: [], c: float) -> float:
        """
        计算损失函数
        Args:
            data([]): 该区域的样本在数据集中的索引
            c(float): 该区域的输出值
        Returns:
            loss(float): 损失函数值
        """
        if not data:
            return 0
        loss = 0
        for i in data:
            loss += (self.trainLabel[i] - c) ** 2
        return loss
    
    def getMinMax(self, data: [], j: int) -> (float, float):
        """
        对样本data的第j个特征值, 求其最大值和最小值
        Args:
            data ([]): 该区域的样本在数据集中的索引
            j(int): 特征值索引
        Returns:
            minVal(float): 特征值的最小值
            maxVal(float): 特征值的最大值
        """
        minVal = float('inf')
        maxVal = -float('inf')
        for i in data:
            minVal = min(minVal, self.trainData[i][j])
            maxVal = max(maxVal, self.trainData[i][j])
        return minVal, maxVal

    def featureSelection(self, features: [], data: []) -> (int, int, float):
        '''
        按照最小二乘原则进行特征和切分点选择
        Args:
            features ([]): 可选的特征索引
            data ([]): 该区域的样本在数据集中的索引
        Returns:
            bestFeature (int): 该区域划分的最优特征
            bestPoint (int): 该最优特征的最优划分点
            minLoss (float): 最小损失函数
        '''
        minLoss = float('inf')
        best_j = -1
        best_s = -1
        for j in features:
            minVal, maxVal = self.getMinMax(data, j)        # 找到样本第j个特征值中的最大值和最小值, 以确定划分点迭代的间隔
            for k in range(0, 6):
                s = minVal + k * (maxVal - minVal) / 5     # 划分点为s
                data1, data2 = self.divide(data, j, s)
                c1 = self.getCm(data1)
                c2 = self.getCm(data2)
                loss = self.getLoss(data1, c1) + self.getLoss(data2, c2)
                if loss < minLoss:
                    minLoss = loss
                    best_j = j
                    best_s = s
        return best_j, best_s, minLoss
    
    def buildingTree(self, data: [], features: [], depth: int, numThreshold=10) -> TreeNode:     
        '''
        按照CART算法构建决策树
        Args:
            data ([]): 该区域的样本在数据集中的索引
            features ([]): 可选的特征索引
            numThreshold (int): 样本数小于阈值则返回叶子节点
        '''
        if not data:
            return
        node = TreeNode()
        node.data = data.copy()
        node.c = self.getCm(data)
        # 当样本数量小于阈值, 或没有可用的特征时, 即为leaf node
        if len(data) <= numThreshold or depth >= self.maxDepth:
            return node
        # 否则, 划分子区域
        best_j, best_s, _ = self.featureSelection(features, data)
        data1, data2 = self.divide(data, best_j, best_s)
        node.j = best_j
        node.s = best_s
        # features.remove(best_j)
        node.left = self.buildingTree(data1, features.copy(), depth + 1, numThreshold)     # 注意应传入features.copy()
        node.right = self.buildingTree(data2, features.copy(), depth + 1, numThreshold)
        return node
    
    def cnt(self, node: TreeNode) -> int:
        """
        计算该子树的叶子节点数量
        """
        if not node:
            return 0
        if not node.left and not node.right:
            return 1
        return self.cnt(node.left) + self.cnt(node.right)

    def cutBranches(self, alpha=1) -> None:
        """
        对决策树进行剪枝
        Step1: 从生成算法产生的决策树T0底端开始不断剪枝, 直到T0的根节点, 形成一个子树序列{T0,...,Tn}
        Step2: 通过交叉验证法在独立的验证集上对子树序列进行预测, 从中选择最优子树
        Returns:
            bestNum(int): 该最优子树的节点个数
        """
        # 剪枝产生子树序列treeList
        treeList = [self.tree.copy()]
        # 当子树只剩下决策树的根节点时停止剪枝
        while treeList[-1].left or treeList[-1].right:
            tree = treeList[-1].copy()
            _, _, _, bestNode = self.dfs(tree)
            bestNode.left = None
            bestNode.right = None             # 在最优节点处进行剪枝
            treeList.append(tree)
        # 交叉验证选择最优子树T 
        minLoss = float('inf')
        bestTree = None
        for tree in treeList:  
            loss, _, _ = self.test(tree, self.validationData, self.validationLabel)
            num = self.cnt(tree)
            # print('Loss is', loss, "Num of node is", num)
            # print('----------------------') 
            loss += num * alpha
            
            if loss < minLoss:
                minLoss = loss
                bestTree = tree
                bestNum = num
        self.tree = bestTree
        return bestNum

    def dfs(self, node: TreeNode) -> (float, int, float, TreeNode):
        '''
        根据CART算法对决策树进行剪枝
        Args:
            node (TreeNode): 当前节点
        Returns:
            loss_subTree (float): 子树的损失函数
            numLeaf (int): 子树的叶节点个数
            minAlpha (float): 最小的alpha值, 即剪枝的最优点对应的alpha值
            bestNode (TreeNode): 剪枝的最优点
        '''
        if not node:
            return 0, 0, None, None
        loss = self.getLoss(node.data, node.c)
        if not node.left and not node.right:        # 当前节点即为叶子节点
            return loss, 1, None, None
        else:                                       
            tmp1, num1, alpha1, node1 = self.dfs(node.left)
            tmp2, num2, alpha2, node2 = self.dfs(node.right)
            loss_subTree = tmp1 + tmp2              # 以当前节点为子树的总损失
            numLeaf = num1 + num2                   # 当前子树的叶节点个数
            minAlpha = float('inf')
            bestNode = None
            if alpha1 and alpha1 < minAlpha:
                minAlpha = alpha1
                bestNode = node1
            if alpha2 and alpha2 < minAlpha:
                minAlpha = alpha2
                bestNode = node2
            if numLeaf != 1:
                alpha = (loss - loss_subTree) / (numLeaf - 1)
                if alpha < minAlpha:
                    minAlpha = alpha
                    bestNode = node
            return loss_subTree, numLeaf, minAlpha, bestNode
    
    def train(self) -> None:
        """
        训练模型
        """
        self.tree = self.buildingTree(list(range(len(self.trainData))), list(range(len(self.trainData[0]))), 0)
    
    def predict(self, data: [], tree=None) -> float:
        """
        使用训练好的回归树模型对数据进行预测
        Args:
            data([]): 带预测样本的特征值
            tree(TreeNode): 用于预测的树模型根节点, 默认为self.tree
        Returns:
            c(float): 预测值
        """
        if not tree:
            tree = self.tree
        
        def dfs(node: TreeNode) -> float:
            """
            预测时用于递归决策树的辅助函数
            """
            if not node.left and not node.right:
                return node.c
            if data[node.j] <= node.s:
                return dfs(node.left)
            else:
                return dfs(node.right)
        return dfs(tree)

    def test(self, tree=None, data=[], label=[]) -> float:
        """
        对模型进行测试
        Args:
            tree(TreeNode): 用于预测的树模型, 默认为self.tree
            data([]): 测试集样本特征
            label([]): 测试集样本标签
        Returns:
            loss(float): 测试损失函数
            R(float): 回归系数, 越接近1则效果越好
            accurate(float): 平均相对误差, 越接近0越好
        """
        if not tree:
            tree = self.tree
        if not data:
            data = self.testData
            label = self.testLabel
        loss = 0
        rand = 0
        accurate = 0
        average = sum(label) / len(label)
        res = []
        for i in range(len(data)):
            val = self.predict(data[i], tree=tree)
            res.append(val)
            accurate += abs(val - label[i]) / label[i]
            loss += (val - label[i]) ** 2
            rand += (average - label[i]) ** 2
        fileObject = open('res.txt', 'w')
        for i in res:
            fileObject.write(str(i))
            fileObject.write('\n')
        fileObject.close()
        R = 1 - loss / rand
        loss /= len(data)
        accurate /= len(data)
        return loss, R, accurate

    def showTree(self, node=None):         
        """
        按层序遍历打印决策树
        """    
        if not node:
            node = self.tree
        queue = [(node, 0)]
        head = 0
        while head < len(queue):
            node, depth = queue[head]
            if head > 0 and depth > queue[head - 1][1]:
                print()
            for i in [node.left, node.right]:
                if i:
                    queue.append((i, depth + 1))
            head += 1
            print('Node', node.j, node.s, node.c, end=' ')
        print()

    def getResidual(self, learningRate=0.2) -> []:
        """
        基于训练好的模型计算其对样本数据的残差, ri = yi - f(xi)
        """
        residual = []
        for i in range(len(self.trainData)):
            val = self.predict(self.trainData[i])
            residual.append((self.trainLabel[i] - learningRate * val))
        return residual


class GBDT:
    def __init__(self, M=10, learningRate=0.2, maxDepth=2):
        self.trees = []  
        self.M = M      # 树的数量
        self.trainData = None
        self.trainLabel = None
        self.validationData = None
        self.validationLabel = None
        self.testData = None
        self.testLabel = None
        self.maxDepth = maxDepth
        self.learningRate = learningRate

    def loadDataSet(self, path: str) -> None:
        """
        加载txt格式的数据集
        """   
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
                self.trainData = data
                self.trainLabel = label
            elif j == 1:
                self.validationData = data
                self.validationLabel = label
            else:
                self.testData = data
                self.testLabel = label

    def train(self):
        """
        训练过程
        """
        trainData = self.trainData.copy()
        trainLabel = self.trainLabel.copy()
        for i in range(self.M):
            tree = CART(trainData=trainData, trainLabel=trainLabel.copy(), maxDepth=self.maxDepth)
            tree.train()
            # tree.cutBranches()
            self.trees.append(tree)
            # tree.showTree()

            if i == 0:
                trainLabel = tree.getResidual(1)
            else:
                trainLabel = tree.getResidual(self.learningRate)  
                 

            print(i)
            # print(sum(trainLabel))

    def getResidual(self):
        """
        基于训练好的模型计算其对样本数据的残差, ri = yi - f(xi)
        """
        residual = []
        for i in range(len(self.trainData)):
            val = self.predict(self.trainData[i])
            residual.append((self.trainLabel[i] - val))
        return residual

    def predict(self, data: []) -> float:
        """
        基于训练好的模型输出预测结果
        """
        res = 0
        for i in range(len(self.trees)):
            if i == 0:
                res += self.trees[i].predict(data)
            else:
                res += self.trees[i].predict(data) * self.learningRate
        return res
    
    def test(self) -> float:
        """
        作用在测试集数据, 返回回归系数R
        Returns:
            loss(float): 损失函数
            R(float): 回归系数
            accurate(float): 平均相对误差
        """
        N = len(self.testData)
        loss = 0
        rand = 0
        average = sum(self.testLabel) / N
        accurate = 0
        for i in range(N):
            val = self.predict(self.testData[i])
            loss += (val - self.testLabel[i]) ** 2
            rand += (self.testLabel[i] - average) ** 2
            accurate += abs((val - self.testLabel[i]) / self.testLabel[i])
        R = 1 - loss / rand
        loss /= N
        accurate /= N
        return loss, R, accurate


if __name__ == "__main__":
    solver = GBDT(M=100, learningRate=0.2, maxDepth=5)
    solver.loadDataSet([r'housing\\data.txt', r'housing\\validate.txt', r'housing\\test.txt'])
    solver.train()
    loss, R, accurate = solver.test()
    print(loss, R, accurate)
    