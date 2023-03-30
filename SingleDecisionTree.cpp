#include "SingleDecisionTree.h"



#include <cmath>
#include <random>
#include <string>
/*
* @function: 构造函数
* @param: attriNum 有多少属性
* @param: classOneName 类别名
* @param k 随机森林随机性
* @param standard 划分的依据（信息增益、信息增益比、基尼系数）
*/
SingleDecisionTree::SingleDecisionTree(int attriNum,std::vector<std::string>& classNames,int k,int standard,bool isCut,int max_depth,double min_restrict):classNames(classNames)
{
	this->min_restrict = min_restrict;
	e.seed(time(0));
	this->max_depth = max_depth;
	this->attriNum = attriNum;
	this->isCut=isCut;
	this->k = k;
	if(k==0)
	{
		this->k = attriNum;
	}
	if(standard==0)
	{
		this->judgeStandard = GiniIndex;
	}else if(standard==1)
	{
		this->judgeStandard = GainRatio;
	}else if(standard==2)
	{
		this->judgeStandard = InfoGain;
	}else
	{
		this->judgeStandard = GiniIndex;
	}

}
SingleDecisionTree::~SingleDecisionTree()
{
	delete this->root;

}

/*
* @function: InfoEntropy() 计算信息熵
* @param: deciNum 有效样本数目（good）
* @param: sampleSum 样本总数
* @return infoEntropy 属性信息熵的值
*/
double SingleDecisionTree::InfoEntropy(int deciNum, int sampleSum)
{
	double infoEnt1, infoEnt2, infoEntropy;
	int undeciNum = sampleSum - deciNum;
	if (sampleSum == 0)
		return 0;
	double P1 = (double)deciNum / (double)sampleSum;
	double P2 = (double)undeciNum / (double)sampleSum;
	if (P1 == 0)
		infoEnt1 = 0;
	else
		infoEnt1 = P1 * log2(P1);
	if (P2 == 0)
		infoEnt2 = 0;
	else
		infoEnt2 = P2 * log2(P2);
	if (P1 == 0 && P2 == 0)
		infoEntropy = 0;
	else
		infoEntropy = -1 * (infoEnt1 + infoEnt2);
	return infoEntropy;
}

/*
* @function: InfoGain() 计算信息增益
* @param: deciNum 有效样本数目（good）
* @param: sampleSum 样本总数
* @param: attr1Num 特征属性分支1上有效样本数目
* @param: attr2Num 特征属性分支2上有效样本数目
* @param: attr1Sum 特征属性分支1上样本数目,可以直接求得特征属性分支2上的样本数目
* @return Gain_A 属性信息增益的值
*/
double SingleDecisionTree::InfoGain(int deciNum, int sampleSum, int attr1Num, int attr1Sum)
{
	int attr2Sum = sampleSum - attr1Sum;
	double InfoEntropy_D = InfoEntropy(deciNum, sampleSum);
	double InfoEntropy_A1 = InfoEntropy(attr1Num, attr1Sum);
	double InfoEntropy_A2 = InfoEntropy(deciNum-attr1Num, attr2Sum);//改过**************
	double A1 = (double)attr1Sum / (double)sampleSum;
	double A2 = (double)attr2Sum / (double)sampleSum;
	double InfoEntropy_A = A1 * InfoEntropy_A1 + A2 * InfoEntropy_A2;
	double Gain_A = InfoEntropy_D - InfoEntropy_A;
	return Gain_A;
}

/*
* @function: 计算信息增益比
* @param: deciNum 有效样本数目（good）
* @param: sampleSum 样本总数
* @param: attr1Num 特征属性分支1上有效样本数目
* @param: attr2Num 特征属性分支2上有效样本数目
* @param: attr1Sum 特征属性分支1上样本数目
* @return GainRatio_A 属性信息增益率的值
*/
double SingleDecisionTree::GainRatio(int deciNum, int sampleSum, int attr1Num, int attr1Sum)
{
	double IV_A = InfoEntropy(deciNum, sampleSum);//原来的信息熵
	double InfoGain_A = InfoGain(deciNum, sampleSum, attr1Num, attr1Sum);//分完的信息增益
	double GainRatio_A = InfoGain_A / IV_A;
	return GainRatio_A;
}

/*
* @function: 计算基尼系数
* @param: deciNum 有效样本数目（good）
* @param: sampleSum 样本总数
* @param: attr1Num 特征属性分支1上有效样本数目
* @param: attr2Num 特征属性分支2上有效样本数目
* @param: attr1Sum 特征属性分支1上样本数目
* @return 基尼系数
*/
double SingleDecisionTree::GiniIndex( int deciNum,int sampleSum, int attr1Num, int attr1Sum)
{
	//std::cout << deciNum << "   " << sampleSum << "   " << attr1Num << "  " << attr1Sum << std::endl;
	double gini =(double)attr1Sum/(double)sampleSum* (1 - pow(((double)attr1Num / (double)attr1Sum), 2) - pow(((double)(attr1Sum - attr1Num) / (double)attr1Sum), 2))+ (double)(sampleSum-attr1Sum)/ (double)sampleSum*(1-pow((double)(deciNum-attr1Num)/ (double)(sampleSum-attr1Sum),2)- pow((double)(sampleSum-attr1Sum-deciNum +attr1Num) / (double)(sampleSum - attr1Sum), 2));
	//std::cout << "基尼" << gini << std::endl;
	return gini;
}

/*2022/12/23/23:49
 *@function:生成决策树（递归）
 *@param: Node当前生成的节点的指针
 *@param: trainSet为训练数据集
 *@param: testSet为测试数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: colNum为数据的列数
 *@param: rowNum为数据的行数
 */
void SingleDecisionTree::decisionTreeCreate(TreeNode*& tree_node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet ,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool> choosable ,int depth,int attribute, double splitValue, int GreatorLess)
{
	
	tree_node = new TreeNode(this->classNames[0]);
	//生成拆分过的训练集
	std::list<int> newTrainSet = updateDateSet(trainSet, subTrainSet, trainClassifications, attribute, splitValue, sampleSum, classOneNum, GreatorLess);
//	std::cout << "得到" << newTrainSet.size() << "个" << std::endl;

	//更新
	sampleSum = 0;
	classOneNum = 0;
	for(std::list<int>::iterator iterator=newTrainSet.begin();iterator!=newTrainSet.end();++iterator)
	{
		sampleSum++;
		if(trainClassifications[trainSet[(*iterator)][0]] == this->classNames[0])
		{
			classOneNum++;
		}
	}

	//用该训练集生成新的节点，在节点中要寻找拆分属性和对应的拆分点
	InitDecisiNode(tree_node, trainSet,newTrainSet, trainClassifications, sampleSum, classOneNum, node_index,choosable,depth);
	if(tree_node->attribute_index==0)
	{
		this->root = tree_node;
		return;
	}
	depth = depth + 1;//深度加一
	decisionTreeCreate(tree_node->greaterChild, trainSet, newTrainSet, trainClassifications, sampleSum, classOneNum, node_index, choosable,depth,tree_node->attribute_index, tree_node->attrValue, 1);
	decisionTreeCreate(tree_node->lessChild, trainSet, newTrainSet, trainClassifications, sampleSum, classOneNum, node_index, choosable,depth,tree_node->attribute_index, tree_node->attrValue, 0);
	this->root = tree_node;
}

/*2022/12/24/00:24
 *@function:寻找二分值
 *@param: trainSet为训练数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: classOneNum 为与该决策树指定值一样的个数
 *@param: sampleSum 该次的样本数
 */
double* SingleDecisionTree::attrSplitValue(std::multiset<ItemLine>& AttrInfo, std::vector<std::string>& trainClassifications, int classOneNum, int sampleSum)
{
	//std::cout << "容量" << AttrInfo.size() << std::endl;
	std::set<double> attrSplitPoints;
	double gain_split[2] = { 0 };//1储存增益比  0储存二分点
	if(this->judgeStandard==GiniIndex)
	{
		gain_split[1] = 1;
	}
	double previous;
	for(std::multiset<ItemLine>::iterator it=AttrInfo.begin();it!=AttrInfo.end();++it)
	{
	
		if( it != AttrInfo.begin()&& previous != (*it).AttrValue)//防止原有值混进去
		{
			attrSplitPoints.insert((previous + (double)it->AttrValue) / (double)2);
		}
		previous = it->AttrValue;
	}

	for(std::set<double>::iterator it=attrSplitPoints.begin();it!=attrSplitPoints.end();++it)
	{
		int greaterClassOneNum = 0;
		int greaterNum = 0;
		for(std::multiset<ItemLine>::iterator m_iterator=AttrInfo.begin();m_iterator != AttrInfo.end();++m_iterator)
		{
			if(m_iterator->AttrValue>*it)
			{
				greaterNum++;
				if(trainClassifications[m_iterator->Index]==this->classNames[0])
				{
					greaterClassOneNum++;
				}
			}
		}
		if(greaterNum==0||greaterNum==sampleSum)
		{
			continue;
		}
		//根据划分的判定方式划分（基尼系数，信息增益比等）
		double judge = judgeStandard(classOneNum, sampleSum, greaterClassOneNum, greaterNum);
		if(judgeStandard==GiniIndex)
		{
			if (judge < gain_split[1])
			{
				gain_split[1] = judge;
				gain_split[0] = *it;
			}
		}else
		{
			if (judge > gain_split[1])
			{
				gain_split[1] = judge;
				gain_split[0] = *it;
			}
		}
	}
	return gain_split;
}
/*2022/12/24/11:02
 *@function:划分训练集
 *@param: trainSet为训练数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: attribute 为属性编号（对应训练集里列）
 *@param: splitValue为二分点
 *@param: classOneNum 为与该决策树指定值一样的个数
 *@param: sampleSum 该次的样本数
 *@param: GreatorLess 标记，表明划分小的还是大的1为大，0为小
 */
std::list<int> SingleDecisionTree::updateDateSet(std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet, std::vector<std::string>& trainClassifications, int attribute, double splitValue, int sampleNum, int classOneNum, int GreatorLess)
{
	std::list<int> vector;
	if(attribute==0)//0位为序号，不划分
	{
		return subTrainSet;
	}else
	{
		std::vector<double> temp;
		if(GreatorLess==1)
		{
			//将大的划分出来
			for(std::list<int>::iterator it=subTrainSet.begin();it!=subTrainSet.end();++it)
			{
			
					if (trainSet[(*it)][attribute] >splitValue)
					{
						vector.push_back((*it));
					}
			}
		}else
		{
			//将小的划分出来
			for (std::list<int>::iterator it = subTrainSet.begin();it != subTrainSet.end();++it)
			{

				if (trainSet[(*it)][attribute] < splitValue)
				{
					vector.push_back((*it));
				}
			}
		}
	}

	return vector;
}
/*2022/12/24/13:06
 *@function:生成树节点
 *@param: Node当前生成的节点的指针
 *@param: trainSet为训练数据集
 *@param: testSet为测试数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: sampleNum 当前训练集总数
 *@param: classOneNum 该类的数量
 *@param: node_index节点编号
 */
void SingleDecisionTree::InitDecisiNode(TreeNode* node, std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool>& choosable,int depth)
{

	node_index++;//编号加一
	node->node_index = node_index;
	if(classOneNum==0||classOneNum==sampleSum)//全部都归为一类了,那么该节点为叶子节点
	{
		
		node->attrValue = INIFINE;//默认值
		node->attribute_index = 0;//0位为序号位，说明没有最佳的属性
		node->classOneNum = classOneNum;//生成该点时classOne有多少个
		node->classTwoNum = sampleSum - classOneNum;//classTwo有多少个
		return;
	}
	//随机森林的决策树需要生成一个随机的
	std::vector<int> selectAttribute = randomSelectAttri(choosable);

	double maxGainRatio = -1;
	double minGini = 2;
	double bestSplit = 0;
	double bestAtrribute=0;
	std::multiset<ItemLine> item_lines;
	for(int i=0;i<selectAttribute.size();i++)//遍历，每一个属性
	{
		int cur = selectAttribute[i];//随机取到的值
		item_lines.clear();


		for(std::list<int>::iterator it=subTrainSet.begin();it!=subTrainSet.end();++it)
		{
			ItemLine item(trainSet[(*it)][0], trainSet[(*it)][cur]);
			item_lines.insert(item);
		}
		double* gain_split = attrSplitValue(item_lines, trainClassifications, classOneNum, sampleSum);

		//根据划分的方式来确定（基尼系数，信息增益比）
		if(this->judgeStandard==GiniIndex&&gain_split[1]<minGini)
		{
			minGini = gain_split[1];
			bestSplit = gain_split[0];
			bestAtrribute = cur;
		}else if(this->judgeStandard!=GiniIndex&&gain_split[1]>maxGainRatio)
		{
			maxGainRatio = gain_split[1];
			bestSplit = gain_split[0];
			bestAtrribute = cur;
		}
	}
	//std::cout << "拆分属性" << bestAtrribute << "拆分值" << bestSplit <<"基尼值"<<minGini << std::endl;
	node->classOneNum = classOneNum;
	node->classTwoNum = sampleSum - classOneNum;
	node->attribute_index = bestAtrribute;
	node->attrValue = bestSplit;
	node->greaterChild = nullptr;
	node->lessChild = nullptr;

	if (minGini<=min_restrict||(maxGainRatio>=0&&maxGainRatio<=min_restrict))//全部都归为一类了,那么该节点为叶子节点
	{
		choosable[bestAtrribute - 1] = false;
		node->attribute_index = 0;//默认值
		return;
	}
	if(depth>=max_depth)
	{
		node->attribute_index = 0;//默认值
		return;
	}

}


/*2023/1/1/11:02
 *@function:决策树可视化处理
 */
void SingleDecisionTree::showDecisionTree()
{
	std::cout << "*************************决策树结点表************************" << std::endl;
	std::cout << std::endl;
	std::cout.setf(std::ios::left);
	std::cout << std::setw(15) << "结点编号";
	std::cout << std::setw(15) << "特征属性";
	std::cout << std::setw(15) << "属性阈值";
	std::cout << std::setw(15) << "LessNode";
	std::cout << std::setw(15) << "LessNum";
	std::cout << std::setw(15) << "GreaterNode";
	std::cout << std::setw(15) << "GreaterNum";
	std::cout << std::setw(15) << "所属类别";
	std::cout << std::endl;
	// 先序遍历输出
	root->preOrder();
}


/*2022/12/24/21:44
 *@function:选择k个属性
 */
std::vector<int> SingleDecisionTree::randomSelectAttri(std::vector<bool>& choosable)
{
	std::uniform_int_distribution<int> u(0, attriNum - 1);
	int num=0;
	for(int i=0;i<choosable.size();i++)
	{
		if(choosable[i])
		{
			num++;
		}
	}
	num = num < k ? num : k;
	std::set<int> v;
	while(v.size()<num)
	{
		int index =u(e) + 1;

		if(!choosable[index-1])
		{
			continue;
		}
		v.insert(index);
	}
	std::vector<int> vector;
	vector.assign(v.begin(), v.end());
	
	return vector;
}

/*2022/12/24/11:02
 *@function:训练
 *@param: trainSet为训练数据集
 *@param: trainClassifications为训练集的分类结果
 */
void SingleDecisionTree::train(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet, std::vector<std::string>& classification)
{
	int depth = 0;

	TreeNode* tree_node = new TreeNode(this->getClassOne());
	int classOneNum = 0;
	int node_index = 0;
	std::vector<bool> choosable;
	for(int i=0;i<attriNum;i++)
	{
		choosable.push_back(true);
	}
	for (std::vector<std::vector<double>>::iterator it = trainSet.begin();it != trainSet.end();it++)
	{
		if (classification[(*it)[0]] == this->getClassOne())
		{
			classOneNum++;
		}
	}
	this->decisionTreeCreate(tree_node, trainSet, subTrainSet,classification, trainSet.size(), classOneNum, node_index,choosable,depth);
	if(isCut)
	{
		std::cout << "剪枝" << std::endl;
		cut(this->root);
	}
}


/*2022/12/24/11:02
 *@function:预测
 *@param: vector供预测的数据

 */
std::string SingleDecisionTree::forecast(std::vector<double>& vector)
{
	TreeNode* p = this->root;
	while (p->attribute_index != 0)		// 属性编号为0代表是叶子结点
	{
		if (vector[p->attribute_index] < p->attrValue)	// 小于分裂值进入左子树
			p = p->lessChild;
		else											// 大于分裂值进入右子树		
			p = p->greaterChild;
	}
	if (p->classOneNum <= p->classTwoNum)
		return classNames[1];
	else
		return classNames[0];
}


double SingleDecisionTree::cut(TreeNode* tree_node)
{
	if (tree_node->attribute_index == 0)//说明这是子节点
	{
		double et;
		if (tree_node->classOneNum > tree_node->classTwoNum)
		{
			et = tree_node->classTwoNum + 0.5;//错误率修正
		}
		else if (tree_node->classOneNum < tree_node->classTwoNum)
		{
			et = tree_node->classOneNum + 0.5;
		}
		else
		{
			et = 0;
		}
			return et;
	}
	double et;
	double ET=0.0;

	if(tree_node->classOneNum>=tree_node->classTwoNum)
	{
		et = tree_node->classTwoNum;
	}else
	{
		et = tree_node->classOneNum;
	}
	ET += cut(tree_node->greaterChild);
	ET += cut(tree_node->lessChild);

	double sum = tree_node->classOneNum + tree_node->classTwoNum;
	double SE = ET * (sum - ET) / sum;
	SE = sqrt(SE);
	if(et<=ET+SE)
	{
		delete tree_node->greaterChild;
		delete tree_node->lessChild;
		tree_node->greaterChild = nullptr;
		tree_node->lessChild = nullptr;
		tree_node->attribute_index = 0;
	}
	return ET;
	
}
