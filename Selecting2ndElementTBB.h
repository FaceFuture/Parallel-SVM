#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

using namespace tbb;

class Selecting2ndElementTBB
{
private:
	static const double TAU_;

	double *alpha_, *yg_, *q_matrix_i_;
	int *labels_;
	int idx_w1_, idx_w2_;
	double g_max_, g_min_, obj_min_, C_;


public:
	Selecting2ndElementTBB()
	{
		alpha_ = yg_ = q_matrix_i_ = NULL;
		labels_ = NULL;
		idx_w1_ = idx_w2_ = -1;
		g_min_ = obj_min_ = C_ = FLT_MAX;
		g_max_ = -FLT_MAX;
	}
	Selecting2ndElementTBB(double *alpha, int *labels, double *yg, double *q_matrix_i, const double C, const double g_max, const int idx_w1)
	{
		SetValues(alpha, labels, yg, q_matrix_i, C, g_max, idx_w1);
	}
	Selecting2ndElementTBB(Selecting2ndElementTBB& s2e, split ) :
		alpha_(s2e.alpha()), yg_(s2e.yg()), q_matrix_i_(s2e.q_matrix_i()), labels_(s2e.labels()), idx_w1_(s2e.idx_w1()), idx_w2_(-1), g_max_(s2e.g_max()), 
		g_min_(FLT_MAX), obj_min_(FLT_MAX), C_(s2e.C())
	{}

	~Selecting2ndElementTBB(){}

	const int idx_w1() const
	{
		return idx_w1_;
	}
	const int idx_w2() const
	{
		return idx_w2_;
	}

	const double g_max() const
	{
		return g_max_;
	}
	const double g_min() const
	{
		return g_min_;
	}
	const double obj_min() const
	{
		return obj_min_;
	}
	int* labels() const
	{
		return labels_;
	}
	double *alpha() const
	{
		return alpha_;
	}
	double *yg() const
	{
		return yg_;
	}
	double *q_matrix_i() const
	{
		return q_matrix_i_;
	}
	double C() const
	{
		return C_;
	}

	void SetValues(double *alpha, int *labels, double *yg, double *q_matrix_i, const double C, const double g_max, const int idx_w1)
	{
		alpha_ = alpha;
		labels_ = labels;
		yg_ = yg;
		q_matrix_i_ = q_matrix_i;
		C_ = C;
		g_max_ = g_max;
		idx_w1_ = idx_w1;
		g_min_ = obj_min_ = FLT_MAX;
		idx_w2_ = -1;
	}

	double GetAij(const int i, const int j)
	{
		double a =  2 - 2 * labels_[i] * labels_[j] * q_matrix_i_[j];  
		return (a<=0) ? TAU_ : a;
	}

	void operator()(const blocked_range<int>& r)
	{
		for(int i=r.begin(); i != r.end(); ++i)
		{
			if (   (  (labels_[i] == 1)  && (alpha_[i] > 0) )     ||    
				(  (labels_[i] == -1) && (alpha_[i] < C_) )   )
			{
				g_min_ = (yg_[i] <= g_min_) ? yg_[i] : g_min_;
				double b = g_max_ - yg_[i];
				if (b > 0)
				{	
					double a = GetAij(idx_w1_, i);
					double bba = - b * b / a;
					if (bba <= obj_min_)
					{
						idx_w2_ = i;
						obj_min_ = bba;
					}
				}
			}
		}
	}

	void join( const Selecting2ndElementTBB& y)
	{
		if (y.g_min() < g_min_)
		{
			g_min_ = y.g_min();
		}
		if (y.obj_min() < obj_min_)
		{
			obj_min_ = y.obj_min();
			idx_w2_ = y.idx_w2();
		}
	}
};

const double Selecting2ndElementTBB::TAU_ = 1e-12; 