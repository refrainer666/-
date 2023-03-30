#include "DecisionTree.h"

#include "SingleDecisionTree.h"
/*2022/12/23/21:08
 *@function:�ò��Լ�����ģ��
 *@param: testSet���Լ�
 *@param: classification�����
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
 *@function:Ԥ�������ļ����������ݷ�Ϊ���Լ���ѵ����
 *@param: fileNameΪ������Դ�ļ�
 *@param: trainSetΪѵ�����ݼ�
 *@param: testSetΪ�������ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: testClassificationsΪ���Լ��ķ����������ڲ���׼ȷ�ԣ�
 *@param: colNumΪ���ݵ�����
 *@param: rowNumΪ���ݵ�����
 */
int DecisionTree::pretreatment(std::string fileName, std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet, std::vector<std::string>& trainClassifications,int colNum)
{
	int rowNum=0;
	readFile(fileName, trainSet, trainClassifications,colNum,rowNum);
	int testNum = rowNum * 0.3;//�ٷ�֮��ʮ��������
	for(int i=rowNum-1;i>=rowNum-testNum;i--)
	{
		testSet.push_back(trainSet.back());
		trainSet.pop_back();
	}
	return trainSet.size();
}

/*2022/12/23/21:09
 *@function:��ȡ�ļ�
 *@param: fileNameΪ������Դ�ļ�
 *@param: trainSetΪѵ�����ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: colNumΪ���ݵ�����
 *@param: rowNumΪ���ݵĺ��������������ڸú������ȡ��ֵ
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
