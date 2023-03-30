#pragma once
#include <map>
#include <string>
#include <vector>

#include "DecisionTree.h"
#include "SingleDecisionTree.h"

class MultiDecisionTree:public DecisionTree
{
public:
	MultiDecisionTree(int attriNum,std::vector<std::string>& classNames, int k=0,int standard=0,bool isCut=false,int max_depth=25,double min_restrict=0);
	virtual ~MultiDecisionTree();
	//ѵ��
	virtual void train(std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet, std::vector<std::string>& classification);

	//Ԥ��
	virtual std::string forecast(std::vector<double>& vector);
	//���������ӻ�
	virtual void showDecisionTree();
private:
	void removeAllClass(std::vector<std::vector<double>>& trainSet, std::list<int>& subTrainSet,std::string className, std::vector<std::string>& classification);
	std::vector<std::string> classNames;//����Ÿ����������
	std::vector<SingleDecisionTree*> subTrees;
	int max_depth;
};
