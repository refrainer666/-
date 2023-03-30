#include "MyMath.h"

#include <cmath>
/*
* @function: log_2() 计算一个数值得以2为底的对数
* @param: n 真数
* @return: log2(n)的值
*/
double MyMath::log_2(double n)
{
	return log10(n) / log10(2.0);
}