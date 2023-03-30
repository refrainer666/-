#include "RandomForest.h"
#include <windows.h>
#include <map>
#include <random>

RandomForest::RandomForest(int attriNum,std::vector<std::string>& classNames,double frac,int k,int standard,int treeNum,int max_depth,double min_restrict):classNames(classNames)
{
	this->min_restrict = min_restrict;
	this->attriNum = attriNum;
	this->frac=frac;
	this->k = k;
	this->standard = standard;
	if (k == 0)
	{
		this->k = attriNum;
	}
	this->treeNum = treeNum;
	this->max_depth = max_depth;
}
/*2022/12/23/21:08
 *@function:预处理，将文件读出的数据分为测试集合训练集
 *@param: fileName为数据来源文件
 *@param: trainSet为训练数据集
 *@param: testSet为测试数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: testClassifications为测试集的分类结果（用于测试准确性）
 *@param: colNum为数据的列数
 *@param: rowNum为数据的行数
 */
int RandomForest::pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications, int colNum)
{
	int rowNum = 0;
	readFile(fileName, trainSet, trainClassifications, colNum, rowNum);
	int testNum = rowNum * 0.3;//百分之三十用来测试
	for (int i = rowNum - 1;i >= rowNum - testNum;i--)
	{
		testSet.push_back(trainSet.back());
		trainSet.pop_back();
	}
	return trainSet.size();
}


/*2022/12/23/21:09
 *@function:读取文件
 *@param: fileName为数据来源文件
 *@param: trainSet为训练数据集
 *@param: trainClassifications为训练集的分类结果
 *@param: colNum为数据的列数
 *@param: rowNum为数据的函数，用引用想在该函数外获取该值
 */
void RandomForest::readFile(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification, int colNum, int& rowNum)
{
	int index = 0;
	rowNum = 0;
	std::ifstream ifs;
	ifs.open(fileName, std::ios::in);
	std::string temp;
	while (std::getline(ifs, temp))
	{
		++rowNum;
		std::vector<double> tempVector;
		std::string tempClass;
		std::istringstream istringstream(temp);
		std::string number;
		tempVector.push_back(index++);
		for (int i = 0;i < colNum;i++)
		{
			std::getline(istringstream, number, ',');
			tempVector.push_back(atof(number.c_str()));
		}
		trainSet.push_back(tempVector);
		tempVector.clear();
		std::getline(istringstream, tempClass);
		trainClassification.push_back(tempClass);
	}
}
/*2022/12/24/21:08
 *@function:根据比例抽抽取训练集
 *@param: trainSet为训练数据集（总的）
 */
std::list<int> RandomForest::randomsubTrainSet(int top)
{

	std::uniform_int_distribution<int> u(0, top-1);
	std::list<int> subTrainSet;
	for(int i=0;i<top*frac;i++)
	{
		int index = u(e);
		subTrainSet.push_back(index);
	}
	
	
	return subTrainSet;
}

/*2022/12/24/21:
 *@function:根据比例抽抽取训练集
 *@param: trainSet为训练数据集（总的）
 */
void RandomForest::train(std::vector<std::vector<double>>& trainSet,std::vector<std::string>& trainClassification)
{
	std::default_random_engine e;
	e.seed(time(0));
	for(int i=0;i<this->treeNum;i++)
	{
		std::list<int> temp = this->randomsubTrainSet(trainSet.size());
		DecisionTree* decision_tree = new MultiDecisionTree(attriNum, classNames, k,this->standard,false,max_depth,min_restrict);
		decision_tree->train(trainSet,temp, trainClassification);
		this->forest.push_back(decision_tree);
	//	Sleep(1000);
	}
}
void RandomForest::show()
{
	std::cout << "该森林由以下树构成" << std::endl;
	for(int i=0;i<forest.size();i++)
	{
		forest[i]->showDecisionTree();
		std::cout << std::endl;
	}
}

void RandomForest::EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification)
{
	std::cout << "该森林各个树准确率" << std::endl;
	for(int i=0;i<forest.size();i++)
	{
		std::cout << forest[i]->EvaluateModel(testSet, classification) << std::endl;
	}
}

std::string RandomForest::forecast(std::vector<double>& vector)
{
	std::map<std::string,int> vote;
	for(std::vector<DecisionTree*>::iterator iterator=forest.begin();iterator!=forest.end();++iterator)
	{
		std::string result = (*iterator)->forecast(vector);
		vote[result] = vote.find(result) == vote.end() ? 1 : vote[result] + 1;
	}
	std::string result;
	int max = 0;
	//std::cout << "投票结果" << std::endl;
	for(std::map<std::string,int>::iterator iterator=vote.begin();iterator!=vote.end();++iterator)
	{
	//	std::cout << iterator->first <<"有" << iterator->second << "票" << std::endl;
		if(iterator->second>max)
		{
			max = iterator->second;
			result = iterator->first;
		}
	}
	return result;
}
