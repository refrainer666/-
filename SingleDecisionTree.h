#pragma once
#include <random>
#include <set>
#include <vector>
# include <sstream>

#include "DecisionTree.h"
#include "TreeNode.h"

struct ItemLine//һ��ItemLine���鴢������Ը����Ե�����ֵ
{
	int Index;						// ѵ��������������
	double AttrValue;				// ѵ����������Ӧ����ֵ

	ItemLine(int Index, double AttrValue)
	{
		this->Index = Index;
		this->AttrValue = AttrValue;
	}

	// ����������ܶ�ItemLine��AttrValue�Ĵ�С��������
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

	//ѵ��
	virtual void train(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& classification);
	
	//���������ӻ�
	virtual void showDecisionTree();


	//Ԥ����
	virtual std::string forecast(std::vector<double>& vector);



	static double InfoEntropy(int deciNum, int sampleSum);//������Ϣ��
	static double InfoGain(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//������Ϣ����
	static double GainRatio(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//������Ϣ�����
	static double GiniIndex(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//�������ָ��

	std::string getClassOne()
	{
		return this->classNames[0];
	}
private:
	//�����ȡ����,���һ���ڵ��ŵ�����
	std::vector<int> randomSelectAttri(std::vector<bool>& choosable);
	//������������
	void decisionTreeCreate(TreeNode*& Node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet ,std::vector<std::string>& classification, int sampleSum, int classOneNum, int& node_index,std::vector<bool> choosable ,int depth,int attribute = 0, double splitValue = 0, int GreatorLess = 0);
	//�ҵ����ֵ�
	double* attrSplitValue(std::multiset<ItemLine>& AttrInfo, std::vector<std::string>& classification, int classOneNum, int sampleSum);
	//ͨ�����ֵ��ָ�����Ի���ѵ����
	std::list<int> updateDateSet(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& classification, int attribute, double splitValue, int sampleNum, int classOneNum, int GreatorLess);
	//�������Ľڵ㣬���ɶ��ֵ������
	void InitDecisiNode(TreeNode* node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool>& choosable,int depth);
	//��֦�㷨PEP
	double cut(TreeNode* tree_node);
	
	int attriNum;
	TreeNode* root;
	//std::string classOne;//�����������ͣ�ֻҪ����֮һ�£�����Ϊ��һ��
	std::vector<std::string> classNames;//�����������ͣ����������Ϊ����
	int k;//�����
	bool isCut;//�Ƿ���Ҫ��֦
	int max_depth;//������
	double min_restrict;
	double (*judgeStandard)(int deciNum, int sampleSum, int attr1Num, int attr1Sum);//���Ժ���


	std::default_random_engine e;

};

