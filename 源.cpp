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
		//效果好，80左右
		//这是26分类的数据集初始化,文件名 二十六分类六百一十七特征.CSV







		//这是连续数据数据集初始化，文件名 连续二分类三十四特征.data
	
		//效果好，80左右
		//这是2分类的数据集初始化,文件名 二分类八特征.CSV


		//效果好，90左右
		//这是离散数据集初始化，文件名 离散二分类三十六特征.CSV
	

		//效果不好,50左右
		//这是7分类的数据集初始化,文件名 七分类十一特征.CSV
		v.clear();
		system("cls");
		std::cout << "请输入您的选择" << std::endl;
		std::cout << "1.二分类36特征测试" << std::endl;
		std::cout << "2.二分类8特征测试" << std::endl;
		std::cout << "3.二分类34特征测试" << std::endl;
		std::cout << "4.七分类11特征测试(一两分钟）" << std::endl;
		std::cout << "5.二十六分类617特征测试（一棵决策树要六分钟左右，森林大概要一个小时）" << std::endl;
		std::cout << "6.三分类4特征测试" << std::endl;
		std::cout << "0.退出" << std::endl;
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
				 filename = "离散二分类三十六特征.CSV";
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
			 filename = "二分类八特征.CSV";
			break;
		case 3:
			colNum = 34;
			max_depth = 30;
			k = log2(colNum);
			min_restrict = 0;
			v.push_back("g");
			v.push_back("b");
			filename = "连续二分类三十四特征.data";
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
			filename = "七分类十一特征.CSV";
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
			filename = "二十六分类六百一十七特征.CSV";
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
			std::cout << "准确率为" << (double)rightNum / (double)testSet.size();
			clock_t end = clock();
			std::cout << "花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << std::endl;
		}
		else
		{
			std::cout << "读取错误" << std::endl;
		}
		system("pause");
	}
}
