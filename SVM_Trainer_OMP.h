#pragma once
#include "SVM_Trainer.h"
class SVM_Trainer_OMP :	public SVM_Trainer
{
protected:
	static const double XX[2], YY[2];

	long num_threads_;
	double  *yg1_, *yg2_, *bba_;
	double **g_omp_, **obj_min_omp_;
	int **index_max_omp_, **index_min_omp_;


public:
	SVM_Trainer_OMP();
	virtual ~SVM_Trainer_OMP();

	//virtual bool SelectWorkingSetOMP(const double C, int &first, int &second);

protected:
	virtual void Init();
	void InitOMP();
	virtual double Select1stElement(const double C, int &first);
	virtual double Select2ndElement(const double C, const int i, const double g_max, int& j);

	//double Select1stElement0(const double C, int &first);
	//double Select2ndElement1(const double C, const int i, const double g_max, int& j);
	void InitSelect1stElement(const double C);
	void InitSelect2ndElement(const double C, const double g_max, const int first);

	void SetElementValuesOMP(double** g, const double x);
	double GetMaxFromOMP(int &first, double** g_max_omp, int** index_max_omp);
	double GetMinFromOMP(int &first, double** g_min_omp, int** index_min_omp);

	double GetSecondFromOMP(int& j);
};
