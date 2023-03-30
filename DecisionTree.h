#pragma once
#include <list>
#include <string>
#include <vector>
# define INIFINE 6554433
class DecisionTree
{
public:
	//预处理，包括读文件与划分测试集
	virtual int pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications, int colNum);
	//读文件数据
	virtual void readFile(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification, int colNum, int& rowNum);
	virtual ~DecisionTree() {};
	//训练
	virtual void train(std::vector<std::vector<double>>& trainSet,std::list<int> subTrain, std::vector<std::string>& classification)=0;
	//预测
	virtual std::string forecast(std::vector<double>& vector)=0;
	//展示
	virtual void showDecisionTree()=0;
	//评估模型
	double EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification);
};

