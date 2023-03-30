#include "DecisionTree.h"
#include "MultiDecisionTree.h"
#include "RandomForest.h"

int main()
{
	bool flag = true;
	while(flag)
	{
		clock_t start = clock();
		int sampleNum=0;
		std::vector<std::string> v;
		int max_depth=0;
		int k=0;
		double min_restrict=0;
		int colNum=0;
		std::string filename=" ";
		//Ч���ã�80����
		//����26��������ݼ���ʼ��,�ļ��� ��ʮ����������һʮ������.CSV







		//���������������ݼ���ʼ�����ļ��� ������������ʮ������.data
	
		//Ч���ã�80����
		//����2��������ݼ���ʼ��,�ļ��� �����������.CSV


		//Ч���ã�90����
		//������ɢ���ݼ���ʼ�����ļ��� ��ɢ��������ʮ������.CSV
	

		//Ч������,50����
		//����7��������ݼ���ʼ��,�ļ��� �߷���ʮһ����.CSV
		v.clear();
		system("cls");
		std::cout << "����������ѡ��" << std::endl;
		std::cout << "1.������36��������" << std::endl;
		std::cout << "2.������8��������" << std::endl;
		std::cout << "3.������34��������" << std::endl;
		std::cout << "4.�߷���11��������(һ�����ӣ�" << std::endl;
		std::cout << "5.��ʮ������617�������ԣ�һ�þ�����Ҫ���������ң�ɭ�ִ��Ҫһ��Сʱ��" << std::endl;
		std::cout << "6.������4��������" << std::endl;
		std::cout << "0.�˳�" << std::endl;
		int choice;
		std::cin >> choice;
		switch (choice)
		{
		case 1:
				colNum = 36;
				max_depth = 30;
				k = log2(colNum);
				min_restrict = 0;
				v.push_back("1");
				v.push_back("-1");
				 filename = "��ɢ��������ʮ������.CSV";
				break;
		case 2:
			 colNum = 8;
			max_depth = 50;
			min_restrict = 0;
			k = log2(colNum);
			for (int i = 0;i <= 1;i++)
			{
				v.push_back(std::to_string(i));
			}
			 filename = "�����������.CSV";
			break;
		case 3:
			colNum = 34;
			max_depth = 30;
			k = log2(colNum);
			min_restrict = 0;
			v.push_back("g");
			v.push_back("b");
			filename = "������������ʮ������.data";
			break;
		case 4:
			colNum = 11;
			max_depth = 30;
			min_restrict = 0;
			k = log2(colNum);
			for (int i = 3;i <= 9;i++)
			{
				v.push_back(std::to_string(i));
			}
			filename = "�߷���ʮһ����.CSV";
			break;
		case 5:
			 colNum = 617;
			max_depth = 10;
			min_restrict = 0.05;
			k = log2(colNum);
			for (int i = 1;i <= 26;i++)
			{
				v.push_back(std::to_string(i));
			}
			filename = "��ʮ����������һʮ������.CSV";
			break;
		case 6:
			colNum = 4;
			max_depth = 50;
			min_restrict = 0;
			k = log2(colNum);
			for (int i = 0;i <= 2;i++)
			{
				v.push_back(std::to_string(i));
			}
			filename = "iris.CSV";
			break;
		case 0:
			flag = false;;
			return 0;
		}

		std::vector<std::vector<double>> trainSet;
		std::vector<std::vector<double>> testSet;
		std::list<int> subTrainSet;
		std::vector<std::string> trainClassifications;
		RandomForest forest(colNum, v, 0.7, k, 0, 10, max_depth, min_restrict);
		if ((sampleNum = forest.pretreatment(filename, trainSet, testSet, trainClassifications, colNum)) != 0)
		{
			for (int i = 0;i < trainSet.size();i++)
			{
				subTrainSet.push_back(i);
			}
			forest.train(trainSet, trainClassifications);
			//forest.show();
			forest.show();
			forest.EvaluateModel(testSet, trainClassifications);
			int rightNum = 0;
			for (int i = 0;i < testSet.size();i++)
			{
				std::string forcast = forest.forecast(testSet[i]);
				if (forcast == trainClassifications[testSet[i][0]])
				{
					rightNum++;
				}
			}
			std::cout << "׼ȷ��Ϊ" << (double)rightNum / (double)testSet.size();
			clock_t end = clock();
			std::cout << "������" << (double)(end - start) / CLOCKS_PER_SEC << "��" << std::endl;
		}
		else
		{
			std::cout << "��ȡ����" << std::endl;
		}
		system("pause");
	}
}
