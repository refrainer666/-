#pragma once
#include <list>
#include <string>
#include <vector>
# define INIFINE 6554433
class DecisionTree
{
public:
	//Ԥ�����������ļ��뻮�ֲ��Լ�
	virtual int pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications, int colNum);
	//���ļ�����
	virtual void readFile(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification, int colNum, int& rowNum);
	virtual ~DecisionTree() {};
	//ѵ��
	virtual void train(std::vector<std::vector<double>>& trainSet,std::list<int> subTrain, std::vector<std::string>& classification)=0;
	//Ԥ��
	virtual std::string forecast(std::vector<double>& vector)=0;
	//չʾ
	virtual void showDecisionTree()=0;
	//����ģ��
	double EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification);
};

