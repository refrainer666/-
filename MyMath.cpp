#include "MyMath.h"

#include <cmath>
/*
* @function: log_2() ����һ����ֵ����2Ϊ�׵Ķ���
* @param: n ����
* @return: log2(n)��ֵ
*/
double MyMath::log_2(double n)
{
	return log10(n) / log10(2.0);
}