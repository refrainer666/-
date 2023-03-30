#pragma once
#include "DecisionTree.h"
#include "MultiDecisionTree.h"

class RandomForest
{
private:
	std::vector<DecisionTree*> forest;
	//std::string classOne;
	std::vector<std::string> classNames;
	int k;//����ԣ�һ����log2d
	int attriNum;//���Եĸ���
	double frac;//�����ı���
	int treeNum;
	int standard;
	int max_depth;
	double min_restrict;
	std::default_random_engine e;
	//��ȡѵ����
	std::list<int> randomsubTrainSet(int top);
public:
	RandomForest(int attriNum,std::vector<std::string>& classNames,double frac, int k=0,int standard=0, int treeNum=10,int max_depth=25,double min_restrict=0);
	//���ļ�����
	void readFile(std::string fileName, std::vector<std::vector<double>>& trainSet,std::vector<std::string>& trainClassification, int colNum,int& rowNum);
	//Ԥ����
	int pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications, int colNum);
	//ѵ��
	void train(std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification);
	//չʾ
	void show();
	//����ģ��
	void EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification);
	//Ԥ��
	std::string forecast(std::vector<double>& vector);
};

