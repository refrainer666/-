#pragma once
#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>

struct TreeNode
{
	std::string classOneName;
	int node_index;					// 结点编号
	int attribute_index;				// 属性编号
	double attrValue;			//属性的分裂值
	int classOneNum;				// 分类一数量，默认为符合
	int classTwoNum;				// 分类二数量
	TreeNode* greaterChild;		// 大于分裂值的子节点
	TreeNode* lessChild;		// 小于分裂值的子节点
	TreeNode(std::string classOneName):classOneName(classOneName)
	{
		node_index = 0;
		attribute_index = 0;
		attrValue = 0;
		classOneNum = 0;
		classTwoNum = 0;
		greaterChild = NULL;
		lessChild = NULL;
	}
	~TreeNode();
	//打印
	friend std::ostream& operator<<(std::ostream& cout, TreeNode& tree_node);
	// 决策树的遍历
	void preOrder();

};
