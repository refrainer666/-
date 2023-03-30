#pragma once
#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>

struct TreeNode
{
	std::string classOneName;
	int node_index;					// �����
	int attribute_index;				// ���Ա��
	double attrValue;			//���Եķ���ֵ
	int classOneNum;				// ����һ������Ĭ��Ϊ����
	int classTwoNum;				// ���������
	TreeNode* greaterChild;		// ���ڷ���ֵ���ӽڵ�
	TreeNode* lessChild;		// С�ڷ���ֵ���ӽڵ�
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
	//��ӡ
	friend std::ostream& operator<<(std::ostream& cout, TreeNode& tree_node);
	// �������ı���
	void preOrder();

};
