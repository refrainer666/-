# include "MultiDecisionTree.h"
MultiDecisionTree::MultiDecisionTree(int attriNum, std::vector<std::string>& classNames, int k,int standard,bool isCut,int max_depth,double min_restrict)
{
	for(std::vector<std::string>::iterator it=classNames.begin();it!=classNames.end();++it)
	{
		std::vector<std::string> subClassNames;
		subClassNames.push_back(*it);
		subClassNames.push_back("not" + *it);
		SingleDecisionTree* decision_tree = new SingleDecisionTree(attriNum, subClassNames, k,standard,isCut,max_depth,min_restrict);
		this->subTrees.push_back(decision_tree);
		
	}
}
void MultiDecisionTree::train(std::vector<std::vector<double>>& trainSet, std::list<int> subTrainSet, std::vector<std::string>& classification)
{
	for(std::vector<SingleDecisionTree*>::iterator it=this->subTrees.begin();it!=this->subTrees.end();++it)
	{
		(*it)->train(trainSet,subTrainSet, classification);
		this->removeAllClass(trainSet,subTrainSet, (*it)->getClassOne(), classification);
	}
}
void MultiDecisionTree::removeAllClass(std::vector<std::vector<double>>& trainSet, std::list<int>& subTrainSet,std::string className, std::vector<std::string>& classification)
{

	for(std::list<int>::iterator it=subTrainSet.begin();it!=subTrainSet.end();)
	{
		if (classification[trainSet[(*it)][0]] == className)
		{
			it=subTrainSet.erase(it);
		}else
		{
			++it;
		}
		//else
		//{
		//	++it;
		//}
	}
}

std::string MultiDecisionTree::forecast(std::vector<double>& vector)
{
	for (std::vector<SingleDecisionTree*>::iterator it = this->subTrees.begin();it != this->subTrees.end();++it)
	{
		std::string re_string=(*it)->forecast(vector);
		if(re_string==(*it)->getClassOne())
		{
			return re_string;
		}
	}
}
MultiDecisionTree::~MultiDecisionTree()
{
	for (std::vector<SingleDecisionTree*>::iterator it = this->subTrees.begin();it != this->subTrees.end();++it)
	{
		delete (*it);
	}
}

void MultiDecisionTree::showDecisionTree()
{
	for (std::vector<SingleDecisionTree*>::iterator it = this->subTrees.begin();it != this->subTrees.end();++it)
	{
		 (*it)->showDecisionTree();
	}
}
