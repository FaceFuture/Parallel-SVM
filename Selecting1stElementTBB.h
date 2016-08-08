#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

using namespace tbb;

class Selecting1stElementTBB
{
private:
	double *alpha_, *yg_;
	int *labels_;
	int idx_w1_;
	double g_max_, C_;

public:
	Selecting1stElementTBB()
	{
		alpha_ = yg_ = NULL;
		labels_ = NULL;
		idx_w1_ = -1;
		g_max_ = C_ = -FLT_MAX;
	}
	Selecting1stElementTBB(double *alpha, int *labels, double *yg, const double C)
	{
		SetValues(alpha, labels, yg, C);
	}
	Selecting1stElementTBB(Selecting1stElementTBB& s1e, split ) :
		 alpha_(s1e.alpha()), yg_(s1e.yg()), labels_(s1e.labels()), idx_w1_(-1), g_max_(-FLT_MAX), C_(s1e.C())
	{}

	~Selecting1stElementTBB(){}

	const int idx_w1() const
	{
		return idx_w1_;
	}
	const double g_max() const
	{
		return g_max_;
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
	 double C() const
	 {
		 return C_;
	 }

	void SetValues(double *alpha, int *labels, double *yg, const double C)
	{
		alpha_ = alpha;
		labels_ = labels;
		yg_ = yg;
		C_ = C;
		g_max_ = -FLT_MAX;
		idx_w1_ = -1;
	}

	void operator()(const blocked_range<int>& r)
	{
		for(int i=r.begin(); i != r.end(); ++i)
		{
			if (   (  (labels_[i] == 1)  && (alpha_[i] < C_) )     ||    
				(  (labels_[i] == -1) && (alpha_[i] > 0) )   )
			{
				if  (  yg_[i]  > g_max_  )
				{
					idx_w1_ = i;
					g_max_ = yg_[i];
				}
			}
		}
	}

	void join( const Selecting1stElementTBB& y)
	{
		if (y.g_max() > g_max_ )
		{
			g_max_ = y.g_max();
			idx_w1_ = y.idx_w1();
		}
	}
};

