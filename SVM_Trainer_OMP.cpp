#include "stdafx.h"
#include "SVM_Trainer_OMP.h"
#include <math.h>
#include <omp.h>
#include "PublicFunctions.h"
#include "DataSet.h"
//#include "mkl.h"
#include "ParticleSwarmOptimizer.h"
#include <chrono>

const double SVM_Trainer_OMP::XX[2] = { FLT_MAX,  0 };
const double SVM_Trainer_OMP::YY[2] = { DBL_MAX, 0 };

SVM_Trainer_OMP::SVM_Trainer_OMP()
{
}


SVM_Trainer_OMP::~SVM_Trainer_OMP()
{
	Public::Release2dArray(g_omp_, num_threads_);
	Public::Release2dArray(obj_min_omp_, num_threads_);
	Public::Release2dArray(index_max_omp_, num_threads_);
	Public::Release2dArray(index_min_omp_, num_threads_);
	Public::ReleaseArray(yg1_);
	Public::ReleaseArray(yg2_);
	Public::ReleaseArray(bba_);
}

void SVM_Trainer_OMP::Init()
{
	SVM_Trainer::Init();
	InitOMP();
}

void SVM_Trainer_OMP::InitOMP()
{
	const long cache_line_size = 64;
	num_threads_ = omp_get_num_procs();
	g_omp_ = NULL;
	obj_min_omp_ = NULL;
	index_max_omp_ = index_min_omp_ = NULL;
	Public::Alloc2dArray(g_omp_, num_threads_, cache_line_size);
	Public::Alloc2dArray(obj_min_omp_, num_threads_, cache_line_size);
	Public::Alloc2dArray(index_max_omp_, num_threads_, cache_line_size);
	Public::Alloc2dArray(index_min_omp_, num_threads_, cache_line_size);
	yg1_ = new double[num_train_samples_];
	yg2_ = new double[num_train_samples_];
	bba_ = new double[num_train_samples_];
}


double SVM_Trainer_OMP::GetMaxFromOMP(int &i_max, double** g_max_omp, int** index_max_omp)
{
	double g_max = -FLT_MAX;
	for (int i = 0; i < num_threads_; i++)
	{
		if (g_max < g_max_omp[i][0])
		{
			g_max = g_max_omp[i][0];
			i_max = index_max_omp[i][0];
		}
	}
	return g_max;
}

double SVM_Trainer_OMP::GetMinFromOMP(int &i_min, double** g_min_omp, int** index_min_omp)
{
	double g_min = FLT_MAX;
	for (int i = 0; i < num_threads_; i++)
	{
		if (g_min > g_min_omp[i][0])
		{
			g_min = g_min_omp[i][0];
			i_min = index_min_omp[i][0];
		}
	}
	return g_min;
}

double SVM_Trainer_OMP::GetSecondFromOMP(int& second)
{
	double g_min = FLT_MAX, obj_min = FLT_MAX;
	for (int i = 0; i<num_threads_; i++)
	{
		if (g_min > g_omp_[i][0])
		{
			g_min = g_omp_[i][0];
		}
		if (obj_min > obj_min_omp_[i][0])
		{
			obj_min = obj_min_omp_[i][0];
			second = index_max_omp_[i][0];
		}
	}
	return g_min;
}



double SVM_Trainer_OMP::Select1stElement(const double C, int &first)
{
	SetElementValuesOMP(g_omp_, -FLT_MAX );
#pragma omp parallel for
	for (int t = 0; t<num_train_samples_; t++)
	{
		if (((train_set_->labels(t) == 1) && (alpha_[t] < C)) || ((train_set_->labels(t) == -1) && (alpha_[t] > 0)))
		{
			int thread_num = omp_get_thread_num();
			//double yg = -train_set_->labels(t) * gradient_[t];
			double yg = (train_set_->labels(t) == -1) ? gradient_[t] : -gradient_[t];
			if (yg  > g_omp_[thread_num][0])
			{
				index_max_omp_[thread_num][0] = t;
				g_omp_[thread_num][0] = yg;
			}
		}
	}
	return GetMaxFromOMP(first, g_omp_, index_max_omp_);
}

double SVM_Trainer_OMP::Select2ndElement(const double C, const int first, const double g_max, int& second)
{
	SetElementValuesOMP(g_omp_, FLT_MAX * 2.0);
	SetElementValuesOMP(obj_min_omp_, FLT_MAX * 2.0);
#pragma omp parallel for
	for (int t = 0; t < num_train_samples_; t++)
	{
		if (((train_set_->labels(t) == 1) && (alpha_[t] > 0)) || ((train_set_->labels(t) == -1) && (alpha_[t] < C)))
		{
			int thread_num = omp_get_thread_num();
			//double yg = -train_set_->labels(t) * gradient_[t];
			double yg = (train_set_->labels(t) == -1) ? gradient_[t] : -gradient_[t];
			g_omp_[thread_num][0] = (yg < g_omp_[thread_num][0]) ? yg : g_omp_[thread_num][0];
			double b = g_max - yg;
			if (b > 0)
			{
				double bba = -b * b / GetAij(first, t);
				if (bba < obj_min_omp_[thread_num][0])
				{
					index_max_omp_[thread_num][0] = t;
					obj_min_omp_[thread_num][0] = bba;
				}
			}
		}
	}
	return GetSecondFromOMP(second);
}

void SVM_Trainer_OMP::SetElementValuesOMP(double** g, const double x)
{
	for (int i = 0; i < num_threads_; i++)
	{
		g[i][0] = x;
	}
}

//bool SVM_Trainer_OMP::SelectWorkingSetOMP(const double C, int &first, int &second)
//{
//	double** g_max_omp = g_omp_, **g_min_omp = obj_min_omp_;
//	SetElementValuesOMP(g_max_omp, -FLT_MAX);
//	SetElementValuesOMP(g_min_omp,   FLT_MAX);
//#pragma omp parallel for
//	for (int t = 0; t<num_train_samples_/2; t++)
//	{
//		int thread_num = omp_get_thread_num();
//		double yg = -train_set_->labels(t) * gradient_[t];
//		if  (alpha_[t] < C)
//		{
//			if (yg  > g_max_omp[thread_num][0])
//			{
//				index_max_omp_[thread_num][0] = t;
//				g_max_omp[thread_num][0] = yg;
//			}
//		}
//		if (alpha_[t] > 0)
//		{
//			if (yg < g_min_omp[thread_num][0])
//			{
//				index_min_omp_[thread_num][0] = t;
//				g_min_omp[thread_num][0] = yg;
//			}
//		}
//	}
//#pragma omp parallel for
//	for (int t = num_train_samples_/2; t<num_train_samples_; t++)
//	{
//		int thread_num = omp_get_thread_num();
//		double yg = -train_set_->labels(t) * gradient_[t];
//		if (alpha_[t] > 0)
//		{
//			if (yg  > g_max_omp[thread_num][0])
//			{
//				index_max_omp_[thread_num][0] = t;
//				g_max_omp[thread_num][0] = yg;
//			}
//		}
//		if (alpha_[t] < C)
//		{
//			if (yg < g_min_omp[thread_num][0])
//			{
//				index_min_omp_[thread_num][0] = t;
//				g_min_omp[thread_num][0] = yg;
//			}
//		}
//	}
//	double g_max = GetMaxFromOMP(first, g_max_omp, index_max_omp_);
//	double g_min = GetMinFromOMP(second, g_min_omp, index_min_omp_);
//	q_matrix_i_ = cov_matrix_ + (long long)first * (long long)num_train_samples_;
//	q_matrix_j_ = cov_matrix_ + (long long)second * (long long)num_train_samples_;
//	return ((g_max - g_min) > EPSILON_);
//}

void SVM_Trainer_OMP::InitSelect1stElement(const double C)
{
#pragma omp parallel for
	for (int i = 0; i<num_train_samples_; i++)
	{
		int kkt = ((train_set_->labels(i) == 1) && (alpha_[i] < C)) || ((train_set_->labels(i) == -1) && (alpha_[i] > 0));
		yg1_[i] = yg_[i] - XX[kkt];
	}
}

//double SVM_Trainer_OMP::Select1stElement0(const double C, int &first)
//{
//	//InitSelectWorkingSet();
//	//InitSelect1stElement(C);
//	SetElementValuesOMP(g_omp_, -FLT_MAX * 2.0);
//#pragma omp parallel for
//	for (int t = 0; t<num_train_samples_; t++)
//	{
//		yg_[t] = -train_set_->labels(t) * gradient_[t];
//		int kkt = ((train_set_->labels(t) == 1) && (alpha_[t] < C)) || ((train_set_->labels(t) == -1) && (alpha_[t] > 0));
//		yg1_[t] = yg_[t] - XX[kkt];
//
//		int thread_num = omp_get_thread_num();
//		if (yg1_[t]  > g_omp_[thread_num][0])
//		{
//			index_max_omp_[thread_num][0] = t;
//			g_omp_[thread_num][0] = yg1_[t];
//		}
//	}
//	return GetMaxFromOMP(first, g_omp_, index_max_omp_);
//}

void SVM_Trainer_OMP::InitSelect2ndElement(const double C, const double g_max, const int first)
{
#pragma omp parallel for
	for (int i = 0; i < num_train_samples_; i++)
	{
		int kkt = ((train_set_->labels(i) == 1) && (alpha_[i] > 0)) || ((train_set_->labels(i) == -1) && (alpha_[i] < C));
		yg2_[i] = yg_[i] + XX[kkt];
		double c = 0, d = 0;
		double b = g_max - yg2_[i];
		c = 2 * b + 9;
		//bba_[i] = -b * b / GetAij(first, i) + YY[b > 0];
		bba_[i] = -b * b / GetAij(first, i);
		d = 3 * c + 2 * b;
		bba_[i] += YY[b > 0];
	}
}

//double SVM_Trainer_OMP::Select2ndElement1(const double C, const int i, const double g_max, int& j)
//{
//	InitSelect2ndElement(C, g_max, i);
//	SetElementValuesOMP(g_omp_, FLT_MAX * 2.0);
//	SetElementValuesOMP(obj_min_omp_, FLT_MAX * 2.0);
//#pragma omp parallel for
//	for (int t = 0; t < num_train_samples_; t++)
//	{
//		int thread_num = omp_get_thread_num();
//		g_omp_[thread_num][0] = (yg2_[t] < g_omp_[thread_num][0]) ? yg2_[t] : g_omp_[thread_num][0];
//
//		double b = g_max - yg2_[t];
//		double bba = -b * b / GetAij(i, t) + YY[b > 0];
//		if (bba < obj_min_omp_[thread_num][0])
//		{
//			index_max_omp_[thread_num][0] = t;
//			obj_min_omp_[thread_num][0] = bba;
//		}
//	}
//	return GetSecondFromOMP(j);
//}
