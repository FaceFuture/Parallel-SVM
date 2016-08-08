// TrainSVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <conio.h>

#include <omp.h>
#include "SVM_Trainer_OMP.h"
#include "SVM_Trainer_TBB.h"
#include "PublicFunctions.h"

int _tmain(int argc, _TCHAR* argv[])
{
	time_t t1=0, t2=0;	time(&t1);
	char* filename_trainset = "G:/data/train_set6W.dat";
	char* filename_crossset = "G:/data/test_set.dat";
	////SVM_Trainer* svm_trainer = new SVM_Trainer(filename_trainset, filename_crossset);
	////time(&t2);	Public::ShowTime((t2-t1)/60.0, "Load Train set："); printf_s("\n\n"); time(&t1);
	////svm_trainer->CalculateRBF_CovMatrixMKL("G:/cov_matrix_6w.dat"); 
	////svm_trainer->CalculateRBF_Train_CV_MatrixMKL("G:/train6w_cv.dat");
	////time(&t2);	Public::ShowTime((t2-t1)/60.0, "CalculateRBF_CovMatrixMKL："); printf_s("\n\n"); time(&t1); exit(0);


	//////svm_trainer->CalculateRBF_PackedCovMatrixMKL("G:/p_cov.dat"); 
	//////svm_trainer->CalculateRBF_Train_CV_MatrixMKL("G:/t_cv.dat");
	//////exit(0);
	////svm_trainer->LoadPackedCovMatrix("G:/packed_cov_matrix_8w_.dat");
	////svm_trainer->LoadTrain_CV_Matrix("G:/train_cv_matrix_8w_.dat");
	////time(&t2);	Public::ShowTime((t2-t1)/60.0, "计算训练与测试的核矩阵并保存："); printf_s("\n\n"); time(&t1);
	////delete svm_trainer;

	//kmp_set_blocktime(1024 * 1024 * 1024);

	SVM_Trainer* svm_trainer = new SVM_Trainer_OMP();
	svm_trainer->LoadLabels(filename_trainset, filename_crossset);
	svm_trainer->LoadCovMatrix("G:/data/cov_matrix_6w.dat");
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "LoadCovMatrix："); printf_s("\n\n"); time(&t1);
	svm_trainer->LoadTrain_CV_Matrix("G:/data/train6w_cv.dat");
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "LoadTrain_CV_Matrix："); printf_s("\n\n"); time(&t1);

	//svm_trainer->SMO_PSO();
	//svm_trainer->SMO_NestPSO();
	svm_trainer->SMO_Grid();

	delete svm_trainer;
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "SMO："); printf_s("\n\n"); time(&t1);	
	return 0;
}

//time(&t2);	Public::ShowTime((t2-t1)/60.0, "载入数据："); printf_s("\n\n"); time(&t1);
//svm_trainer->CalculateRBF_PackedCovMatrixMKL();
//time(&t2);	Public::ShowTime((t2-t1)/60.0, "计算对称核矩阵并保存："); printf_s("\n\n"); time(&t1);
//svm_trainer->CalculateRBF_Train_CV_MatrixMKL();