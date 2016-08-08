#include "stdafx.h"
#include "SVM_Trainer_TBB.h"
#include "Selecting1stElementTBB.h"
#include "Selecting2ndElementTBB.h"
#include "DataSet.h"

SVM_Trainer_TBB::SVM_Trainer_TBB()
{
	select_1st_element_tbb_ = new Selecting1stElementTBB();
	select_2nd_element_tbb_ = new Selecting2ndElementTBB();
}


SVM_Trainer_TBB::~SVM_Trainer_TBB()
{
	delete select_1st_element_tbb_;
	delete select_2nd_element_tbb_;
}


double SVM_Trainer_TBB::Select1stElement(const double C, int &first)
{
	InitSelectWorkingSet();
	select_1st_element_tbb_->SetValues(alpha_, (int *) ( train_set_->labels() ), yg_, C);
	tbb::parallel_reduce(tbb::blocked_range<int>(0, num_train_samples_), *select_1st_element_tbb_, tbb::auto_partitioner());
	first = select_1st_element_tbb_->idx_w1();	
	return select_1st_element_tbb_->g_max();
}

double SVM_Trainer_TBB::Select2ndElement(const double C, const int idx_w1, const double g_max, int& idx_w2)
{
	select_2nd_element_tbb_->SetValues(alpha_, (int *) train_set_->labels(), yg_, q_matrix_i_, C, g_max, idx_w1);
	tbb::parallel_reduce(tbb::blocked_range<int>(0, num_train_samples_), *select_2nd_element_tbb_, tbb::auto_partitioner());
	idx_w2 = select_2nd_element_tbb_->idx_w2();	
	return select_2nd_element_tbb_->g_min();
}
