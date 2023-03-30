#include "SingleDecisionTree.h"



#include <cmath>
#include <random>
#include <string>
/*
* @function: ���캯��
* @param: attriNum �ж�������
* @param: classOneName �����
* @param k ���ɭ�������
* @param standard ���ֵ����ݣ���Ϣ���桢��Ϣ����ȡ�����ϵ����
*/
SingleDecisionTree::SingleDecisionTree(int attriNum,std::vector<std::string>& classNames,int k,int standard,bool isCut,int max_depth,double min_restrict):classNames(classNames)
{
	this->min_restrict = min_restrict;
	e.seed(time(0));
	this->max_depth = max_depth;
	this->attriNum = attriNum;
	this->isCut=isCut;
	this->k = k;
	if(k==0)
	{
		this->k = attriNum;
	}
	if(standard==0)
	{
		this->judgeStandard = GiniIndex;
	}else if(standard==1)
	{
		this->judgeStandard = GainRatio;
	}else if(standard==2)
	{
		this->judgeStandard = InfoGain;
	}else
	{
		this->judgeStandard = GiniIndex;
	}

}
SingleDecisionTree::~SingleDecisionTree()
{
	delete this->root;

}

/*
* @function: InfoEntropy() ������Ϣ��
* @param: deciNum ��Ч������Ŀ��good��
* @param: sampleSum ��������
* @return infoEntropy ������Ϣ�ص�ֵ
*/
double SingleDecisionTree::InfoEntropy(int deciNum, int sampleSum)
{
	double infoEnt1, infoEnt2, infoEntropy;
	int undeciNum = sampleSum - deciNum;
	if (sampleSum == 0)
		return 0;
	double P1 = (double)deciNum / (double)sampleSum;
	double P2 = (double)undeciNum / (double)sampleSum;
	if (P1 == 0)
		infoEnt1 = 0;
	else
		infoEnt1 = P1 * log2(P1);
	if (P2 == 0)
		infoEnt2 = 0;
	else
		infoEnt2 = P2 * log2(P2);
	if (P1 == 0 && P2 == 0)
		infoEntropy = 0;
	else
		infoEntropy = -1 * (infoEnt1 + infoEnt2);
	return infoEntropy;
}

/*
* @function: InfoGain() ������Ϣ����
* @param: deciNum ��Ч������Ŀ��good��
* @param: sampleSum ��������
* @param: attr1Num �������Է�֧1����Ч������Ŀ
* @param: attr2Num �������Է�֧2����Ч������Ŀ
* @param: attr1Sum �������Է�֧1��������Ŀ,����ֱ������������Է�֧2�ϵ�������Ŀ
* @return Gain_A ������Ϣ�����ֵ
*/
double SingleDecisionTree::InfoGain(int deciNum, int sampleSum, int attr1Num, int attr1Sum)
{
	int attr2Sum = sampleSum - attr1Sum;
	double InfoEntropy_D = InfoEntropy(deciNum, sampleSum);
	double InfoEntropy_A1 = InfoEntropy(attr1Num, attr1Sum);
	double InfoEntropy_A2 = InfoEntropy(deciNum-attr1Num, attr2Sum);//�Ĺ�**************
	double A1 = (double)attr1Sum / (double)sampleSum;
	double A2 = (double)attr2Sum / (double)sampleSum;
	double InfoEntropy_A = A1 * InfoEntropy_A1 + A2 * InfoEntropy_A2;
	double Gain_A = InfoEntropy_D - InfoEntropy_A;
	return Gain_A;
}

/*
* @function: ������Ϣ�����
* @param: deciNum ��Ч������Ŀ��good��
* @param: sampleSum ��������
* @param: attr1Num �������Է�֧1����Ч������Ŀ
* @param: attr2Num �������Է�֧2����Ч������Ŀ
* @param: attr1Sum �������Է�֧1��������Ŀ
* @return GainRatio_A ������Ϣ�����ʵ�ֵ
*/
double SingleDecisionTree::GainRatio(int deciNum, int sampleSum, int attr1Num, int attr1Sum)
{
	double IV_A = InfoEntropy(deciNum, sampleSum);//ԭ������Ϣ��
	double InfoGain_A = InfoGain(deciNum, sampleSum, attr1Num, attr1Sum);//�������Ϣ����
	double GainRatio_A = InfoGain_A / IV_A;
	return GainRatio_A;
}

/*
* @function: �������ϵ��
* @param: deciNum ��Ч������Ŀ��good��
* @param: sampleSum ��������
* @param: attr1Num �������Է�֧1����Ч������Ŀ
* @param: attr2Num �������Է�֧2����Ч������Ŀ
* @param: attr1Sum �������Է�֧1��������Ŀ
* @return ����ϵ��
*/
double SingleDecisionTree::GiniIndex( int deciNum,int sampleSum, int attr1Num, int attr1Sum)
{
	//std::cout << deciNum << "   " << sampleSum << "   " << attr1Num << "  " << attr1Sum << std::endl;
	double gini =(double)attr1Sum/(double)sampleSum* (1 - pow(((double)attr1Num / (double)attr1Sum), 2) - pow(((double)(attr1Sum - attr1Num) / (double)attr1Sum), 2))+ (double)(sampleSum-attr1Sum)/ (double)sampleSum*(1-pow((double)(deciNum-attr1Num)/ (double)(sampleSum-attr1Sum),2)- pow((double)(sampleSum-attr1Sum-deciNum +attr1Num) / (double)(sampleSum - attr1Sum), 2));
	//std::cout << "����" << gini << std::endl;
	return gini;
}

/*2022/12/23/23:49
 *@function:���ɾ��������ݹ飩
 *@param: Node��ǰ���ɵĽڵ��ָ��
 *@param: trainSetΪѵ�����ݼ�
 *@param: testSetΪ�������ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: colNumΪ���ݵ�����
 *@param: rowNumΪ���ݵ�����
 */
void SingleDecisionTree::decisionTreeCreate(TreeNode*& tree_node, std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet ,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool> choosable ,int depth,int attribute, double splitValue, int GreatorLess)
{
	
	tree_node = new TreeNode(this->classNames[0]);
	//���ɲ�ֹ���ѵ����
	std::list<int> newTrainSet = updateDateSet(trainSet, subTrainSet, trainClassifications, attribute, splitValue, sampleSum, classOneNum, GreatorLess);
//	std::cout << "�õ�" << newTrainSet.size() << "��" << std::endl;

	//����
	sampleSum = 0;
	classOneNum = 0;
	for(std::list<int>::iterator iterator=newTrainSet.begin();iterator!=newTrainSet.end();++iterator)
	{
		sampleSum++;
		if(trainClassifications[trainSet[(*iterator)][0]] == this->classNames[0])
		{
			classOneNum++;
		}
	}

	//�ø�ѵ���������µĽڵ㣬�ڽڵ���ҪѰ�Ҳ�����ԺͶ�Ӧ�Ĳ�ֵ�
	InitDecisiNode(tree_node, trainSet,newTrainSet, trainClassifications, sampleSum, classOneNum, node_index,choosable,depth);
	if(tree_node->attribute_index==0)
	{
		this->root = tree_node;
		return;
	}
	depth = depth + 1;//��ȼ�һ
	decisionTreeCreate(tree_node->greaterChild, trainSet, newTrainSet, trainClassifications, sampleSum, classOneNum, node_index, choosable,depth,tree_node->attribute_index, tree_node->attrValue, 1);
	decisionTreeCreate(tree_node->lessChild, trainSet, newTrainSet, trainClassifications, sampleSum, classOneNum, node_index, choosable,depth,tree_node->attribute_index, tree_node->attrValue, 0);
	this->root = tree_node;
}

/*2022/12/24/00:24
 *@function:Ѱ�Ҷ���ֵ
 *@param: trainSetΪѵ�����ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: classOneNum Ϊ��þ�����ָ��ֵһ���ĸ���
 *@param: sampleSum �ôε�������
 */
double* SingleDecisionTree::attrSplitValue(std::multiset<ItemLine>& AttrInfo, std::vector<std::string>& trainClassifications, int classOneNum, int sampleSum)
{
	//std::cout << "����" << AttrInfo.size() << std::endl;
	std::set<double> attrSplitPoints;
	double gain_split[2] = { 0 };//1���������  0������ֵ�
	if(this->judgeStandard==GiniIndex)
	{
		gain_split[1] = 1;
	}
	double previous;
	for(std::multiset<ItemLine>::iterator it=AttrInfo.begin();it!=AttrInfo.end();++it)
	{
	
		if( it != AttrInfo.begin()&& previous != (*it).AttrValue)//��ֹԭ��ֵ���ȥ
		{
			attrSplitPoints.insert((previous + (double)it->AttrValue) / (double)2);
		}
		previous = it->AttrValue;
	}

	for(std::set<double>::iterator it=attrSplitPoints.begin();it!=attrSplitPoints.end();++it)
	{
		int greaterClassOneNum = 0;
		int greaterNum = 0;
		for(std::multiset<ItemLine>::iterator m_iterator=AttrInfo.begin();m_iterator != AttrInfo.end();++m_iterator)
		{
			if(m_iterator->AttrValue>*it)
			{
				greaterNum++;
				if(trainClassifications[m_iterator->Index]==this->classNames[0])
				{
					greaterClassOneNum++;
				}
			}
		}
		if(greaterNum==0||greaterNum==sampleSum)
		{
			continue;
		}
		//���ݻ��ֵ��ж���ʽ���֣�����ϵ������Ϣ����ȵȣ�
		double judge = judgeStandard(classOneNum, sampleSum, greaterClassOneNum, greaterNum);
		if(judgeStandard==GiniIndex)
		{
			if (judge < gain_split[1])
			{
				gain_split[1] = judge;
				gain_split[0] = *it;
			}
		}else
		{
			if (judge > gain_split[1])
			{
				gain_split[1] = judge;
				gain_split[0] = *it;
			}
		}
	}
	return gain_split;
}
/*2022/12/24/11:02
 *@function:����ѵ����
 *@param: trainSetΪѵ�����ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: attribute Ϊ���Ա�ţ���Ӧѵ�������У�
 *@param: splitValueΪ���ֵ�
 *@param: classOneNum Ϊ��þ�����ָ��ֵһ���ĸ���
 *@param: sampleSum �ôε�������
 *@param: GreatorLess ��ǣ���������С�Ļ��Ǵ��1Ϊ��0ΪС
 */
std::list<int> SingleDecisionTree::updateDateSet(std::vector<std::vector<double>>& trainSet,std::list<int> subTrainSet, std::vector<std::string>& trainClassifications, int attribute, double splitValue, int sampleNum, int classOneNum, int GreatorLess)
{
	std::list<int> vector;
	if(attribute==0)//0λΪ��ţ�������
	{
		return subTrainSet;
	}else
	{
		std::vector<double> temp;
		if(GreatorLess==1)
		{
			//����Ļ��ֳ���
			for(std::list<int>::iterator it=subTrainSet.begin();it!=subTrainSet.end();++it)
			{
			
					if (trainSet[(*it)][attribute] >splitValue)
					{
						vector.push_back((*it));
					}
			}
		}else
		{
			//��С�Ļ��ֳ���
			for (std::list<int>::iterator it = subTrainSet.begin();it != subTrainSet.end();++it)
			{

				if (trainSet[(*it)][attribute] < splitValue)
				{
					vector.push_back((*it));
				}
			}
		}
	}

	return vector;
}
/*2022/12/24/13:06
 *@function:�������ڵ�
 *@param: Node��ǰ���ɵĽڵ��ָ��
 *@param: trainSetΪѵ�����ݼ�
 *@param: testSetΪ�������ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 *@param: sampleNum ��ǰѵ��������
 *@param: classOneNum ���������
 *@param: node_index�ڵ���
 */
void SingleDecisionTree::InitDecisiNode(TreeNode* node, std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet,std::vector<std::string>& trainClassifications, int sampleSum, int classOneNum, int& node_index,std::vector<bool>& choosable,int depth)
{

	node_index++;//��ż�һ
	node->node_index = node_index;
	if(classOneNum==0||classOneNum==sampleSum)//ȫ������Ϊһ����,��ô�ýڵ�ΪҶ�ӽڵ�
	{
		
		node->attrValue = INIFINE;//Ĭ��ֵ
		node->attribute_index = 0;//0λΪ���λ��˵��û����ѵ�����
		node->classOneNum = classOneNum;//���ɸõ�ʱclassOne�ж��ٸ�
		node->classTwoNum = sampleSum - classOneNum;//classTwo�ж��ٸ�
		return;
	}
	//���ɭ�ֵľ�������Ҫ����һ�������
	std::vector<int> selectAttribute = randomSelectAttri(choosable);

	double maxGainRatio = -1;
	double minGini = 2;
	double bestSplit = 0;
	double bestAtrribute=0;
	std::multiset<ItemLine> item_lines;
	for(int i=0;i<selectAttribute.size();i++)//������ÿһ������
	{
		int cur = selectAttribute[i];//���ȡ����ֵ
		item_lines.clear();


		for(std::list<int>::iterator it=subTrainSet.begin();it!=subTrainSet.end();++it)
		{
			ItemLine item(trainSet[(*it)][0], trainSet[(*it)][cur]);
			item_lines.insert(item);
		}
		double* gain_split = attrSplitValue(item_lines, trainClassifications, classOneNum, sampleSum);

		//���ݻ��ֵķ�ʽ��ȷ��������ϵ������Ϣ����ȣ�
		if(this->judgeStandard==GiniIndex&&gain_split[1]<minGini)
		{
			minGini = gain_split[1];
			bestSplit = gain_split[0];
			bestAtrribute = cur;
		}else if(this->judgeStandard!=GiniIndex&&gain_split[1]>maxGainRatio)
		{
			maxGainRatio = gain_split[1];
			bestSplit = gain_split[0];
			bestAtrribute = cur;
		}
	}
	//std::cout << "�������" << bestAtrribute << "���ֵ" << bestSplit <<"����ֵ"<<minGini << std::endl;
	node->classOneNum = classOneNum;
	node->classTwoNum = sampleSum - classOneNum;
	node->attribute_index = bestAtrribute;
	node->attrValue = bestSplit;
	node->greaterChild = nullptr;
	node->lessChild = nullptr;

	if (minGini<=min_restrict||(maxGainRatio>=0&&maxGainRatio<=min_restrict))//ȫ������Ϊһ����,��ô�ýڵ�ΪҶ�ӽڵ�
	{
		choosable[bestAtrribute - 1] = false;
		node->attribute_index = 0;//Ĭ��ֵ
		return;
	}
	if(depth>=max_depth)
	{
		node->attribute_index = 0;//Ĭ��ֵ
		return;
	}

}


/*2023/1/1/11:02
 *@function:���������ӻ�����
 */
void SingleDecisionTree::showDecisionTree()
{
	std::cout << "*************************����������************************" << std::endl;
	std::cout << std::endl;
	std::cout.setf(std::ios::left);
	std::cout << std::setw(15) << "�����";
	std::cout << std::setw(15) << "��������";
	std::cout << std::setw(15) << "������ֵ";
	std::cout << std::setw(15) << "LessNode";
	std::cout << std::setw(15) << "LessNum";
	std::cout << std::setw(15) << "GreaterNode";
	std::cout << std::setw(15) << "GreaterNum";
	std::cout << std::setw(15) << "�������";
	std::cout << std::endl;
	// ����������
	root->preOrder();
}


/*2022/12/24/21:44
 *@function:ѡ��k������
 */
std::vector<int> SingleDecisionTree::randomSelectAttri(std::vector<bool>& choosable)
{
	std::uniform_int_distribution<int> u(0, attriNum - 1);
	int num=0;
	for(int i=0;i<choosable.size();i++)
	{
		if(choosable[i])
		{
			num++;
		}
	}
	num = num < k ? num : k;
	std::set<int> v;
	while(v.size()<num)
	{
		int index =u(e) + 1;

		if(!choosable[index-1])
		{
			continue;
		}
		v.insert(index);
	}
	std::vector<int> vector;
	vector.assign(v.begin(), v.end());
	
	return vector;
}

/*2022/12/24/11:02
 *@function:ѵ��
 *@param: trainSetΪѵ�����ݼ�
 *@param: trainClassificationsΪѵ�����ķ�����
 */
void SingleDecisionTree::train(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet, std::vector<std::string>& classification)
{
	int depth = 0;

	TreeNode* tree_node = new TreeNode(this->getClassOne());
	int classOneNum = 0;
	int node_index = 0;
	std::vector<bool> choosable;
	for(int i=0;i<attriNum;i++)
	{
		choosable.push_back(true);
	}
	for (std::vector<std::vector<double>>::iterator it = trainSet.begin();it != trainSet.end();it++)
	{
		if (classification[(*it)[0]] == this->getClassOne())
		{
			classOneNum++;
		}
	}
	this->decisionTreeCreate(tree_node, trainSet, subTrainSet,classification, trainSet.size(), classOneNum, node_index,choosable,depth);
	if(isCut)
	{
		std::cout << "��֦" << std::endl;
		cut(this->root);
	}
}


/*2022/12/24/11:02
 *@function:Ԥ��
 *@param: vector��Ԥ�������

 */
std::string SingleDecisionTree::forecast(std::vector<double>& vector)
{
	TreeNode* p = this->root;
	while (p->attribute_index != 0)		// ���Ա��Ϊ0������Ҷ�ӽ��
	{
		if (vector[p->attribute_index] < p->attrValue)	// С�ڷ���ֵ����������
			p = p->lessChild;
		else											// ���ڷ���ֵ����������		
			p = p->greaterChild;
	}
	if (p->classOneNum <= p->classTwoNum)
		return classNames[1];
	else
		return classNames[0];
}


double SingleDecisionTree::cut(TreeNode* tree_node)
{
	if (tree_node->attribute_index == 0)//˵�������ӽڵ�
	{
		double et;
		if (tree_node->classOneNum > tree_node->classTwoNum)
		{
			et = tree_node->classTwoNum + 0.5;//����������
		}
		else if (tree_node->classOneNum < tree_node->classTwoNum)
		{
			et = tree_node->classOneNum + 0.5;
		}
		else
		{
			et = 0;
		}
			return et;
	}
	double et;
	double ET=0.0;

	if(tree_node->classOneNum>=tree_node->classTwoNum)
	{
		et = tree_node->classTwoNum;
	}else
	{
		et = tree_node->classOneNum;
	}
	ET += cut(tree_node->greaterChild);
	ET += cut(tree_node->lessChild);

	double sum = tree_node->classOneNum + tree_node->classTwoNum;
	double SE = ET * (sum - ET) / sum;
	SE = sqrt(SE);
	if(et<=ET+SE)
	{
		delete tree_node->greaterChild;
		delete tree_node->lessChild;
		tree_node->greaterChild = nullptr;
		tree_node->lessChild = nullptr;
		tree_node->attribute_index = 0;
	}
	return ET;
	
}
