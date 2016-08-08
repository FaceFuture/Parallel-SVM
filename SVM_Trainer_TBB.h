#pragma once
#include "SVM_Trainer.h"
class Selecting1stElementTBB;
class Selecting2ndElementTBB;


class SVM_Trainer_TBB : public SVM_Trainer
{
protected:
	Selecting1stElementTBB* select_1st_element_tbb_;
	Selecting2ndElementTBB* select_2nd_element_tbb_;

public:
	SVM_Trainer_TBB();
	virtual ~SVM_Trainer_TBB();


	virtual double Select1stElement(const double C, int &first);
	virtual double Select2ndElement(const double C, const int i, const double g_max, int& j);
};

