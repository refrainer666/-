#include "DecisionTree.h"

#include "SingleDecisionTree.h"
/*2022/12/23/21:08
 *@function:用测试集评估模型
 *@param: testSet测试集
 *@param: classification结果集
 */
double DecisionTree::EvaluateModel(std::vector<std::vector<double>>& testSet, std::vector<std::string>& classification)
{
	int rightNum = 0;
	for (std::vector<std::vector<double>>::iterator iterator = testSet.begin();iterator != testSet.end();++iterator)
	{
		if (this->forecast(*iterator) == classification[(*iterator)[0]])
		{
			rightNum++;
		}
	}

	return (double)rightNum / (double)testSet.size();
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
int DecisionTree::pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications,int colNum)
{
	int rowNum=0;
	readFile(fileName, trainSet, trainClassifications,colNum,rowNum);
	int testNum = rowNum * 0.3;//百分之三十用来测试
	for(int i=rowNum-1;i>=rowNum-testNum;i--)
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
void DecisionTree::readFile(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::string>& trainClassification, int colNum,int& rowNum)
{
	int index = 0;
	rowNum = 0;
	std::ifstream ifs;
	ifs.open(fileName, std::ios::in);
	std::string temp;
	while(std::getline(ifs,temp))
	{
		++rowNum;
		std::vector<double> tempVector;
		std::string tempClass;
		std::istringstream istringstream(temp);
		std::string number;
		tempVector.push_back(index++);
		for(int i=0;i<colNum;i++)
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
