#pragma once
#include <random>
#include <set>
#include <vector>
# include <sstream>

#include "DecisionTree.h"
#include "TreeNode.h"

struct ItemLine//一个ItemLine数组储存的是以个属性的所有值
{
	int Index;						// 训练集样本的索引
	double AttrValue;				// 训练集样本对应属性值

	ItemLine(int Index, double AttrValue)
	{
		this->Index = Index;
		this->AttrValue = AttrValue;
	}

	// 运算符重载能对ItemLine按AttrValue的大小进行排序
	bool operator < (const ItemLine& n) const
	{
		return AttrValue < n.AttrValue;
	}

};

class SingleDecisionTree:public DecisionTree
{
public:
	SingleDecisionTree(int attriNum,std::vector<std::string>& classNames, int k=0,int standard=0,bool isCut=false,int max_depth=25,double min_restrict=0);
	virtual ~SingleDecisionTree();

	//训练
	virtual void train(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& classification);
	
	//决策树可视化
	virtual void showDecisionTree();


	//预测结果
	virtual std::string forecast(std::vector<double>& vector);



	static double InfoEntropy(int deciNum, int sampleSum);//计算信息熵
	static double InfoGain(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//计算信息增益
	static double GainRatio(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//计算信息增益比
	static double GiniIndex(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//计算基尼指数

	std::string getClassOne()
	{
		return this->classNames[0];
	}
private:
	//随机获取属性,获得一个节点编号的数组
	std::vector<int> randomSelectAttri(std::vector<bool>& choosable);
	//决策树的生成
	void decisionTreeCreate(TreeNode*& Node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet ,std::vector<std::string>& classification, int sampleSum, int classOneNum, int& node_index,std::vector<bool> choosable ,int depth,int attribute = 0, double splitValue = 0, int GreatorLess = 0);
	//找到二分点
	double* attrSplitValue(std::multiset<ItemLine>& AttrInfo, std::vector<std::string>& classification, int classOneNum, int sampleSum);
	//通过二分点和指定属性划分训练集
	std::list<int> updateDateSet(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& classification, int attribute, double splitValue, int sampleNum, int classOneNum, int GreatorLess);
	//生成树的节点，生成二分点和属性
	void InitDecisiNode(TreeNode* node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool>& choosable,int depth);
	//剪枝算法PEP
	double cut(TreeNode* tree_node);
	
	int attriNum;
	TreeNode* root;
	//std::string classOne;//储存类别的类型，只要不与之一致，都视为另一类
	std::vector<std::string> classNames;//储存类别的类型，在这个树中为两个
	int k;//随机性
	bool isCut;//是否需要剪枝
	int max_depth;//最大深度
	double min_restrict;
	double (*judgeStandard)(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//策略函数


	std::default_random_engine e;

};

