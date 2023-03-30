#include "TreeNode.h"
#define INIFINE 6554433
std::ostream& operator<<(std::ostream& cout, TreeNode& tree_node)
{
	{
		// ��������
		cout << std::setw(15) << tree_node.node_index;
		// �������������
		if (tree_node.attribute_index != 0)
			cout << "����" << std::setw(12) << tree_node.attribute_index;
		else
			cout << std::setw(15) << " ";
		// ������������ֵ
		if (tree_node.attrValue != INIFINE)
			cout << std::setw(15) << std::setprecision(6) << tree_node.attrValue;
		else
			cout << std::setw(15) << " ";
		// ���С����ֵ�Ľ����
		if (tree_node.lessChild) {
			cout << std::setw(15) << tree_node.lessChild->node_index;
		}
		else {
			cout << std::setw(15) << " ";
		}
		cout << std::setw(15) << tree_node.classOneNum;
		// ���������ֵ�Ľ����
		if (tree_node.greaterChild) {
			cout << std::setw(15) << tree_node.greaterChild->node_index;

		}
		else {
			cout << std::setw(15) << " ";
		}
		cout << std::setw(15) << tree_node.classTwoNum;
			// ��������������
		if (tree_node.classOneNum == 0)
			cout << std::setw(15) << "Not" << tree_node.classOneName;
		else if (tree_node.classTwoNum == 0)
			cout << std::setw(15) <<tree_node.classOneName;
		cout << std::endl;
	}
	return cout;
}
void TreeNode::preOrder()
{
	std::cout << *this;
	if(this->lessChild)
	{
		this->lessChild->preOrder();
	}
	if(this->greaterChild)
	{
		this->greaterChild->preOrder();
	}
}

TreeNode::~TreeNode()
{
	if(this->greaterChild)
	{
		delete this->greaterChild;
	}
	if(this->lessChild)
	{
		delete this->lessChild;
	}
}

