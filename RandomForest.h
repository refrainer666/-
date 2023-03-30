#pragma once
#include "DecisionTree.h"
#include "MultiDecisionTree.h"

class RandomForest
{
private:
	std::vector<DecisionTree*> forest;
	//std::string classOne;
	std::vector<std::string> classNames;
	int k;//随机性，一般是log2d
	int attriNum;//属性的个数
	double frac;//抽样的比例
	int treeNum;
	int standard;
	int max_depth;
	double min_restrict;
	std::default_random_engine e;
	//抽取训练集
	std::list<int> randomsubTrainSet(int top);
public:
	RandomForest(int attriNum,std::vector<std::string>& classNames,double frac, int k=0,int standard=0, int treeNum=10,int max_depth=25,double min_restrict=0);
	//读文件数据
	void readFile(std::string fileName, std::vector<std::vector<double>>& trainSet,std::vector<std::string>& trainClassification, int colNum,int& rowNum);
	//预处理
	int pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications, int colNum);
	//训练
	void train(std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification);
	//展示
	void show();
	//评估模型
	void EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification);
	//预测
	std::string forecast(std::vector<double>& vector);
};

