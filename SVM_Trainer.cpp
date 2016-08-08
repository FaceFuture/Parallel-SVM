#include "StdAfx.h"
#include <math.h>
#include <omp.h>
#include "SVM_Trainer.h"
#include "PublicFunctions.h"
#include "DataSet.h"
#include "mkl.h"
#include "ParticleSwarmOptimizer.h"
#include <chrono>
#include <iostream>
#include <time.h>

const double SVM_Trainer::EPSILON_ = 1e-5; //1e-3;
const double SVM_Trainer::TAU_ = 1e-12; 


SVM_Trainer::SVM_Trainer(void)
{
	train_set_ = new DataSet<double>();
	cross_validation_set_ =  new DataSet<double>();
	cov_matrix_ = train_cv_matrix_ = alpha_ = gradient_ = NULL;
	support_vectors_index_ = NULL;	
	num_train_samples_ = num_cv_samples_ = num_features_ = num_support_vectors_ = 0;
	rho_ = gamma_ = 0.0;
}
SVM_Trainer::SVM_Trainer(const char* filename_train_set, const char* filename_cross_validation_set)
{
	train_set_ = new DataSet<double>(filename_train_set);
	cross_validation_set_ =  new DataSet<double>(filename_cross_validation_set);
	cov_matrix_ = train_cv_matrix_ = NULL;	
	Init();
}

SVM_Trainer::~SVM_Trainer(void)
{
	if (train_set_ != NULL)
	{
		delete train_set_;
	}
	//if (cross_validation_set_ != NULL)
	//{
	//	delete cross_validation_set_;
	//}
	Public::ReleaseArray(cov_matrix_);
	Public::ReleaseArray(train_cv_matrix_);
	//Public::ReleaseArray(alpha_);

	//Public::ReleaseArray(gradient_);
	//Public::ReleaseArray(support_vectors_index_);
	//Public::ReleaseArray(output_filename_);

	//Public::ReleaseArray(yg_);
}


void SVM_Trainer::LoadLabelsData(const char* filename_train_set, const char* filename_cross_validation_set)
{
	train_set_->LoadDataLabels(filename_train_set);
	cross_validation_set_->LoadDataLabels(filename_cross_validation_set);
	Init();
}

void SVM_Trainer::LoadLabels(const char* filename_train_set, const char* filename_cross_validation_set)
{
	train_set_->LoadLabels(filename_train_set);
	cross_validation_set_->LoadLabels(filename_cross_validation_set);
	Init();
}

void SVM_Trainer::Init()
{
	num_train_samples_ = train_set_->num_samples();
	num_cv_samples_ = cross_validation_set_->num_samples();
	num_features_ = train_set_->num_features();

	num_support_vectors_ = 0;
	rho_ = gamma_ = 0.0;
	alpha_ = new double[num_train_samples_];
	gradient_ = new double[num_train_samples_];
	support_vectors_index_ = new long[num_train_samples_];
	q_matrix_i_ = q_matrix_j_ = NULL;

	yg_ = new double[num_train_samples_];

	SetOutputFilename();
}

void SVM_Trainer::SetOutputFilename()
{
	const time_t t = time(NULL);
	struct tm* current_time = new struct tm;
	localtime_s(current_time, &t);
	output_filename_ = new char[256];
	sprintf_s(output_filename_, 256, "G:/svm_results_%d_%d_%d_%d.txt", 
		current_time->tm_mon + 1, current_time->tm_mday, current_time->tm_hour, current_time->tm_min);
	delete current_time;
}

void SVM_Trainer::CalculateRBF_CovMatrixMKL(char* filename_cov_matrix)
{	
	time_t t1=0, t2=0;	time(&t1);
	Public::AllocArray(cov_matrix_, (long long)num_train_samples_ * (long long)num_train_samples_);
	cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, num_train_samples_, num_features_, 1.0, train_set_->data(), num_features_,
		0.0, cov_matrix_, num_train_samples_); 
	Public::CopyUpperTriangularMatrix2Lower(cov_matrix_, num_train_samples_);
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "cblas_dsyrk£º"); printf_s("\n\n"); time(&t1);
	CalculateCovMatrixRBF(cov_matrix_);
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "CalculateRBF_CovMatrix£º"); printf_s("\n\n"); time(&t1);
	Public::SaveMatrix(cov_matrix_, num_train_samples_, num_train_samples_, filename_cov_matrix);
	time(&t2);	Public::ShowTime((t2-t1)/60.0, "SaveMatrix cov_matrix_£º"); printf_s("\n\n"); time(&t1);
}

void SVM_Trainer::LoadCovMatrix(char* filename)
{
	Public::LoadMatrix(cov_matrix_, filename);
}

void SVM_Trainer::LoadTrain_CV_Matrix(char* filename)
{
	//Public::LoadMatrix(train_cv_matrix_, num_cv_samples_, num_train_samples_, filename);
	Public::LoadMatrix(train_cv_matrix_, filename);
}

void SVM_Trainer::ExpLongVector(double* long_vector, const long long size)
{
#pragma omp parallel for
	for (long long i = 0; i < size; i++)
	{
		long_vector[i] = exp(long_vector[i]);
	}
}

void SVM_Trainer::PowerLongVector(double* long_vector, const long long size, const double gamma)
{
#pragma omp parallel for
	for (long long i = 0; i < size; i++)
	{
		long_vector[i] = pow(long_vector[i], gamma);
	}
}

void SVM_Trainer::ExpLongVectorMKL(double* long_vector, const long long size)
{
	long long NUM_SEGMENT = 32;
	long long nn = size / NUM_SEGMENT;
#pragma omp parallel for
	for (long long i=0; i<nn; i++)
	{
		vmdExp(NUM_SEGMENT, long_vector + i*(long long)NUM_SEGMENT, long_vector + i*(long long)NUM_SEGMENT, VML_HA);
	}
	long long xx =  nn * NUM_SEGMENT;
	int size_left = (int)(size - xx);	//printf_s("size_left = %ld, xx = %lld\n", size_left, xx);
	if (size_left > 0)
	{
		vmdExp(size_left, long_vector + xx, long_vector + xx, VML_HA);
	}
}

void SVM_Trainer::PowerLongVectorMKL(double* long_vector, const long long size, const double gamma)
{
	long long NUM_SEGMENT = 512;
	long long nn = size / NUM_SEGMENT;
#pragma omp parallel for
	for (long long i=0; i<nn; i++)
	{
		vdPowx(NUM_SEGMENT, long_vector + i*NUM_SEGMENT, gamma, long_vector + i*NUM_SEGMENT);
	}
	long long xx =  nn * NUM_SEGMENT;
	int size_left = (int)(size - xx);	//printf_s("size_left = %ld, xx = %lld\n", size_left, xx);
	if (size_left > 0)
	{
		vdPowx(size_left, long_vector + xx, gamma, long_vector + xx);
	}
}

void SVM_Trainer::CalculateCovMatrix(double* cov_matrix)
{
	double* diag_cov = new double[num_train_samples_];
#pragma omp parallel for
	for (long long i = 0; i < num_train_samples_; i++)
	{
		diag_cov[i] = cov_matrix[i + i*num_train_samples_];
	}
#pragma omp parallel for
	for (long long i = 0; i < num_train_samples_; i++)
	{
		for (long long j = 0; j < num_train_samples_; j++)
		{
			cov_matrix[j + i*num_train_samples_] = -diag_cov[i] - diag_cov[j] + 2 * cov_matrix[j + i*num_train_samples_];
		}
	}
	delete[] diag_cov;
}


void SVM_Trainer::CalculateCovMatrixRBF(double* cov_matrix)
{
	CalculateCovMatrix(cov_matrix);
	const long long size_cov_matrix = (long long)num_train_samples_ * (long long)num_train_samples_; 
	ExpLongVector(cov_matrix, size_cov_matrix);
}


void SVM_Trainer::CalculateRBF_Train_CV_MatrixMKL(char* filename_train_cv_matrix)
{
	Public::AllocArray(train_cv_matrix_, (long long)num_train_samples_ * (long long)num_cv_samples_);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_cv_samples_, num_train_samples_, num_features_, 1.0, 
						cross_validation_set_->data(), num_features_, train_set_->data(), num_features_, 0.0, train_cv_matrix_, num_train_samples_);
	CalculateRBF_Train_CV_Matrix(train_cv_matrix_);
	Public::SaveMatrix(train_cv_matrix_, num_cv_samples_, num_train_samples_, filename_train_cv_matrix);
	Public::ReleaseArray(train_cv_matrix_);
}

void SVM_Trainer::CalculateRBF_Train_CV_Matrix(double* train_cv_matrix)
{
	double* train_dot_products = new double [num_train_samples_];
	double* cv_dot_products = new double [num_cv_samples_];
	MultiDotProduct(train_set_->data(), num_train_samples_, num_features_, train_dot_products);
	MultiDotProduct(cross_validation_set_->data(), num_cv_samples_, num_features_, cv_dot_products);
#pragma omp parallel for
	for (long long i=0; i<num_cv_samples_; i++)
	{
		for (long long j=0; j<num_train_samples_; j++)
		{
			train_cv_matrix[i*num_train_samples_ + j] = - cv_dot_products[i] - train_dot_products[j] + 2*train_cv_matrix[i*num_train_samples_ + j];
		}
	}
	ExpLongVectorMKL(train_cv_matrix, (long long)num_train_samples_ * num_cv_samples_);
	delete[] cv_dot_products;
	delete[] train_dot_products;
}

void SVM_Trainer::MultiDotProduct(const double* samples, const long num_samples, const long num_features, double* dot_products)
{
#pragma omp parallel for
	for (int i=0; i<num_samples; i++)
	{
		dot_products[i] = cblas_ddot(num_features, samples + (long long)i*num_features, 1, samples + (long long)i*num_features, 1);
	}
}

void SVM_Trainer::InitGradient()
{
#pragma omp parallel for
	for (int i=0; i<num_train_samples_; i++)
	{
		gradient_[i] = -1.0;
	}
}

double SVM_Trainer::Select1stElement(const double C, int &first)
{
	double g_max = -FLT_MAX;
	for (int t=0; t<num_train_samples_; t++)
	{
		if (   (  (train_set_->labels(t) == 1)  && (alpha_[t] < C) )     ||    
			   (  (train_set_->labels(t) == -1) && (alpha_[t] > 0) )   )
		{
			double yg = -train_set_->labels(t) * gradient_[t];
			if  (  yg  >= g_max  )
			{
				first = t;
				g_max = yg;
			}
		}
	}
	return g_max;
}


double SVM_Trainer::GetAij(const int i, const int j)
{
	//double a =  PackedCovMatrix(i, i) + PackedCovMatrix(j, j) - 2*train_set_->labels(i)*train_set_->labels(j) * PackedCovMatrix(i, j);  
	//double a =  2 - 2*train_set_->labels(i)*train_set_->labels(j) * PackedCovMatrix(i, j);  
	double a =  2 - 2*train_set_->labels(i)*train_set_->labels(j) * q_matrix_i_[j];  
	return (a < TAU_) ? TAU_ : a;
}

double SVM_Trainer::Select2ndElement(const double C, const int i, const double g_max, int& j)
{
	double g_min = FLT_MAX, obj_min = FLT_MAX;
	for (int t=0; t<num_train_samples_; t++)
	{
		if (  ( (train_set_->labels(t) == 1)  && (alpha_[t] > 0) )    ||   
			  ( (train_set_->labels(t) == -1) && (alpha_[t] < C) )   )
		{
			double yg = -train_set_->labels(t) * gradient_[t];
			double b = g_max - yg;
			g_min = (yg <= g_min) ? yg : g_min;
			if (b > 0)
			{
				double bba = -b * b / GetAij(i, t);
				if (bba <= obj_min)
				{
					j = t;
					obj_min = bba;
				}
			}
		}
	}
	return g_min;
}



void SVM_Trainer::MultiplyLabels2Qmatrix() 
{
#pragma omp parallel for
	for (long long i=0; i<num_train_samples_; i++)
	{
		for (long long j=0; j<num_train_samples_; j++)
		{
			cov_matrix_[j + i*num_train_samples_] *= train_set_->labels(i)*train_set_->labels(j);
		}
	}
}

void SVM_Trainer::MultiplyLabels2QmatrixMKL() 
{
	long num_samples_class1 = train_set_->num_samples_class1();
#pragma omp parallel for
	for (long long  i=0; i<num_samples_class1; i++)
	{
		cblas_dscal(num_train_samples_-num_samples_class1, -1, cov_matrix_ + i*num_train_samples_ + num_samples_class1, 1);
	}
#pragma omp parallel for
	for (long long i=num_samples_class1; i<num_train_samples_; i++)
	{
		cblas_dscal(num_samples_class1, -1, cov_matrix_ + i*num_train_samples_, 1);
	}
}

void SVM_Trainer::SMO_Pre(const double gamma)
{
	PowerLongVectorMKL(cov_matrix_, (long long)num_train_samples_ * (long long)num_train_samples_, gamma / gamma_);
	MultiplyLabels2QmatrixMKL();
	PowerLongVectorMKL(train_cv_matrix_, (long long)num_train_samples_ * (long long)num_cv_samples_, gamma / gamma_);
	gamma_ = gamma;	
}

bool SVM_Trainer::SelectWorkingSet(const double C, int &first, int &second)
{	
	double g_max = Select1stElement(C, first);
	q_matrix_i_ = cov_matrix_ + (long long)first * (long long)num_train_samples_;
	double g_min = Select2ndElement(C, first, g_max, second);
	q_matrix_j_ = cov_matrix_ + (long long)second * (long long)num_train_samples_;
	return ((g_max - g_min) > EPSILON_);
}

void SVM_Trainer::SMO_Main( const double C)
{
	memset(alpha_, 0, num_train_samples_*sizeof(double) );
	InitGradient();
	int first = 0, second = 0;
	while (SelectWorkingSet(C, first, second))  //?????????????????????????????????????????????
	{
		double old_alpha_i = alpha_[first];
		double old_alpha_j = alpha_[second];
		UpdateAlpha(C, first, second);
		UpdateGradientMKL(alpha_[first] - old_alpha_i, alpha_[second] - old_alpha_j);
	}
	ComputeRho(C);
	GetSupportVectorsIndex();
}


void SVM_Trainer::SMO_Grid()
{
	FILE* fid = NULL;
	time_t t1=0, t2=0;	time(&t1);
	printf_s("num_cv_samples_=%d, num_train_samples_=%d\n", num_cv_samples_, num_train_samples_);
	gamma_ = 1.0;
	const int nn_C = 32, nn_gamma = 8;
	const double c_min = 2.32, c_max = 2.48;
	const double gamma_min = -9.5535, gamma_max = -9.4547;
	const double cc_step = (c_max - c_min)/(nn_C - 1), gg_step = (gamma_max - gamma_min)/(nn_gamma - 1);
	double *rates = new double [nn_C];

	for (int i = 0; i<nn_gamma; i++)
	{
		double gg = pow(2.0, gamma_min + gg_step*i);
		SMO_Pre(gg);
		for (int j=0; j<nn_C; j++)
		{
			double cc = pow(2.0, c_min + j*cc_step);
			SMO_Main(cc);
			rates[j] = Test();
		}
		MultiplyLabels2QmatrixMKL();
		time(&t2); Public::ShowTime((t2 - t1) / 60.0, nn_gamma, i + 1);
		fopen_s(&fid, output_filename_, "at");
		for (int j=0; j<nn_C; j++)
		{
			fprintf_s(fid, "ng=%3d, gamma=%12.8lf, C=%12.8lf, test rate = %12.8lf \n", i, gg, pow(2.0, j*cc_step - 1), rates[j]);		
		}
		fclose(fid);
	}

	delete[] rates;
}


// assure 0 < ahpha < C
void SVM_Trainer::ProjectAlpha2FeasibleRegion(const double C, double& alpha)
{
	alpha = (alpha > C) ? C : ( (alpha < 0 ) ? 0 : alpha );  // assure:  0 <= ahpha <= C
}

void SVM_Trainer::UpdateAlpha(const double C, const int i, const int j)
{
	const int yi = train_set_->labels(i);
	const int yj = train_set_->labels(j);
	double b_a =  (- yi * gradient_[i]  +  yj * gradient_[j]) / GetAij(i, j);
	double sum = yi*alpha_[i] + yj*alpha_[j];
	alpha_[i] += yi * b_a;
	alpha_[j] -= yj * b_a;
	ProjectAlpha2FeasibleRegion(C, alpha_[i]);
	alpha_[j] = yj * ( sum - yi*alpha_[i] );
	ProjectAlpha2FeasibleRegion(C, alpha_[j]);
	alpha_[i] = yi * ( sum - yj*alpha_[j] );
}

void SVM_Trainer::UpdateGradient(const double delta_ahpha_i, const double delta_ahpha_j)
{
#pragma omp parallel for
	for (int t=0; t<num_train_samples_; t++)
	{
		//gradient_[t] += PackedCovMatrix(t, i) * delta_ahpha_i + PackedCovMatrix(t, j) * delta_ahpha_j;
		gradient_[t] += q_matrix_i_[t] * delta_ahpha_i + q_matrix_j_[t] * delta_ahpha_j;
	}
}

void SVM_Trainer::UpdateGradientMKL(const double delta_ahpha_i, const double delta_ahpha_j)  //***********
{
	cblas_daxpy(num_train_samples_, delta_ahpha_i, q_matrix_i_, 1, gradient_, 1);
	cblas_daxpy(num_train_samples_, delta_ahpha_j, q_matrix_j_, 1, gradient_, 1);
}

void SVM_Trainer::ComputeRho(const double C)
{
	long num = 0;
	double rho = 0.0;
#pragma omp parallel for reduction(+ : num, rho)
	for (int i=0; i<num_train_samples_; i++)
	{
		if (alpha_[i] > 0  &&  alpha_[i] < C)
		{
			num ++;
			rho += train_set_->labels(i) * gradient_[i];
		}
	}
	rho_ = rho / num;
}

void SVM_Trainer::GetSupportVectorsIndex()
{
	num_support_vectors_ = 0;
	for (int i=0; i<num_train_samples_; i++)
	{
		if (alpha_[i] > 0)
		{
			support_vectors_index_[num_support_vectors_] = i;
			num_support_vectors_ ++;
		}
	}
}

double SVM_Trainer::Test()
{
	long num_corrects = 0;
	for (long long i=0; i<num_cv_samples_; i++)
	{
		double g = -rho_;
		for (int n_sv=0; n_sv<num_support_vectors_; n_sv++)
		{
			long sv_idx = support_vectors_index_[n_sv];
			g += train_set_->labels(sv_idx) * alpha_[sv_idx] * train_cv_matrix_[i*num_train_samples_ + sv_idx];
		}
		num_corrects += ( g*cross_validation_set_->labels(i) > 0 );
	}
	return num_corrects/(double)num_cv_samples_;
}


void SVM_Trainer::SMO_NestPSO()
{
	time_t t1=0, t2=0, t3=0;	time(&t1);
	printf_s("num_cv_samples_=%d, num_train_samples_=%d\n", num_cv_samples_, num_train_samples_);
	gamma_ = 1.0;
	//tbb::task_scheduler_init inittbb;
	const long num_particles = 16, dim_particle=1, num_generations = 120;
	const double pso_gamma_max = -5.5, pso_gamma_min = -10.5;
	const double pso_C_max = 25, pso_C_min = 2.5;
	ParticleSwarmOptimizer* pso_gamma = new ParticleSwarmOptimizer( num_particles, dim_particle, num_generations, &pso_gamma_max, &pso_gamma_min);
	ParticleSwarmOptimizer* pso_C = new ParticleSwarmOptimizer( 10, dim_particle, 10, &pso_C_max, &pso_C_min);

	unsigned int seed = 0;
	unsigned int nnn=0;
	pso_gamma->InitRandom(seed);		
	time(&t3); 
	for (int g_gamma=0; g_gamma<num_generations; g_gamma++)
	{
		for (int np_gamma=0; np_gamma<num_particles; np_gamma++)
		{
			double gg = pow(2.0, pso_gamma->particles(np_gamma, 0));				printf_s("g_gamma=%ld, np_gamma=%ld, gamma=%8.5lf\n", g_gamma, np_gamma, gg);
			SMO_Pre(gg);
			SMO_PSO_C_Evolve(pso_C, seed + nnn);	
			pso_gamma->fitnesses(np_gamma, pso_C->gbest_fitnesses(0));
			MultiplyLabels2Qmatrix();
			time(&t2); Public::ShowTime( (t2-t3)/60.0, num_generations*num_particles,  ++nnn);
		}
		pso_gamma->GetBest();
		PrintOutPSO_Gamma(pso_gamma);	
		pso_gamma->Save(NULL);
		pso_gamma->UpdateVelocitiesParticles();
	}
	delete pso_gamma;
	delete pso_C;
}

void SVM_Trainer::SMO_PSO_C_Once(int np_C, ParticleSwarmOptimizer* pso_C)
{
	//double cc = pow(2.0, pso_C->particles(np_C, 0));
	double cc = pso_C->particles(np_C, 0);
	SMO_Main(cc);
	double rate = Test();
	pso_C->fitnesses(np_C, rate);
}

void SVM_Trainer::SMO_PSO_C_Evolve(ParticleSwarmOptimizer* pso_C, unsigned int seed )
{
	pso_C->InitRandom(seed);
	for (int g_C=0; g_C<pso_C->num_generations(); g_C++)
	{
		printf_s("gc=%ld, npc=", g_C);
		for (int np_C=0; np_C<pso_C->num_particles(); np_C++)
		{					
			printf_s("%ld, %8.5lf; ", np_C, pso_C->particles(np_C, 0));
			SMO_PSO_C_Once(np_C, pso_C);
		}
		printf_s("\n");
		pso_C->GetBest();
		PrintOutPSO_C(pso_C);
		//pso_C->Save("G:/pso_C.dat");
		pso_C->UpdateVelocitiesParticles();	
	}
}

void SVM_Trainer::SMO_PSO()
{
	printf_s("num_cv_samples_=%d, num_train_samples_=%d\n", num_cv_samples_, num_train_samples_);
	gamma_ = 1.0;
	const long num_particles = 32, dim_particle=2, num_generations = 500;
	const double pso_max[] = {-4.0, 16.0}, pso_min[] = {-11.0, 0.5};
	ParticleSwarmOptimizer* pso = new ParticleSwarmOptimizer( num_particles, dim_particle, num_generations, pso_max, pso_min);
	unsigned int seed = 19750815 + 256;
	pso->InitRandom(seed);
	pso->Load("G:/PSO_data_05_13_15_01_13.dat");

	time_t t1=0, t2=0;	time(&t1);		
	for (int i_g=0; i_g<num_generations; i_g++)
	{
		for (int np=0; np<num_particles; np++)
		{
			double gg = pow( 2.0, pso->particles(np, 0) );
			double cc = pso->particles(np, 1);
			SMO_Pre(gg);
			SMO_Main(cc);
			double rate = Test();
			pso->fitnesses(np, rate);
			MultiplyLabels2Qmatrix();
		}
		pso->GetBest();
		PrintOutPSO(pso);	
		pso->Save(NULL);
		pso->UpdateVelocitiesParticles();
		time(&t2); Public::ShowTime( (t2-t1)/60.0, num_generations,  i_g+1);
	}
	delete pso;
}

void SVM_Trainer::PrintOutPSO(ParticleSwarmOptimizer* pso)
{
	FILE* fid = NULL; 
	fopen_s(&fid, output_filename_, "at");
	for (int i=0; i<pso->num_particles(); i++)
	{
		fprintf_s( fid, "np = %3d, gamma = %12.8lf, C = %12.8lf, test rate = %8.6lf\n", i, pow(2.0, pso->particles(i, 0)), pso->particles(i, 1), pso->fitnesses(i) );
	}
	for (int i=0; i<pso->num_particles(); i++)
	{
		fprintf_s( fid, " @@@@@@@@ n_best=%3d, gamma=%12.8lf, C = %12.8lf, test rate = %8.6lf @@@@@@@@ \n",
			i, pow(2.0, pso->gbest(i, 0)), pso->gbest(i, 1), pso->gbest_fitnesses(i) );
	}
	fclose(fid);
}


void SVM_Trainer::PrintOutPSO_Gamma(ParticleSwarmOptimizer* pso_gamma)
{
	FILE* fid = NULL; 
	fopen_s(&fid, output_filename_, "at");
	for (int i=0; i<pso_gamma->num_particles(); i++)
	{
		fprintf_s( fid, " $$$$$ np_gamma = %3d, gamma = %12.8lf, test rate = %8.6lf $$$$$ \n", i, pow(2.0, pso_gamma->particles(i, 0)), pso_gamma->fitnesses(i) );
	}
	fprintf_s(fid, "+++++++++ global best gamma =%12.8lf, global test rate = %8.6lf +++++++++\n", pow(2.0, pso_gamma->gbest(0, 0)), pso_gamma->gbest_fitnesses(0));
	for (int i=0; i<pso_gamma->num_particles(); i++)
	{
		fprintf_s( fid, " @@@@@@@@ n_best=%3d, gamma=%12.8lf, test rate = %8.6lf @@@@@@@@ \n",
			i, pow(2.0, pso_gamma->gbest(i, 0)), pso_gamma->gbest_fitnesses(i) );
	}
	fclose(fid);
}

void SVM_Trainer::PrintOutPSO_C(ParticleSwarmOptimizer* pso_C)
{
	FILE* fid = NULL; 
	fopen_s(&fid, output_filename_, "at");
	fprintf_s(fid, " ******************************************************************************************************************************* \n");
	for (int np_C=0; np_C<pso_C->num_particles(); np_C++)
	{
		fprintf_s( fid, "np_C=%3d, gamma=%12.8lf, C=%12.8lf, test rate = %8.6lf \n",np_C, gamma_, pso_C->particles(np_C, 0), pso_C->fitnesses(np_C) );
	}
	fprintf_s(fid, "+++++++++ global best: C=%12.8lf, global test rate = %8.6lf +++++++++\n", pso_C->gbest(0, 0), pso_C->gbest_fitnesses(0));
	for (int i=0; i<pso_C->num_particles(); i++)
	{
		fprintf_s( fid, " ========= n_best=%3d, gamma=%12.8lf, C=%12.8lf, test rate = %8.6lf =========\n",
			i, gamma_, pso_C->gbest(i, 0), pso_C->gbest_fitnesses(i) );
	}
	fclose(fid);
}


void SVM_Trainer::InitSelectWorkingSet()
{
#pragma omp parallel for
	for (int i = 0; i < num_train_samples_; i++)
	{
		yg_[i] = -train_set_->labels(i) * gradient_[i];
	}
}


//
//
//void SVM_Trainer::SMO_Packed_Pre(const double gamma)
//{
//	time_t t1=0, t2=0;	time(&t1);
//	long long size = (long long)num_train_samples_ * ( (long long)num_train_samples_ + 1) / 2;
//	gamma_ = 1.0;
//	PowerLongVector(packed_cov_matrix_, size, gamma/gamma_);
//	MultiplyLabels2PackedQmatrix();
//	PowerLongVector(train_cv_matrix_, (long long)num_train_samples_ * (long long)num_cv_samples_, gamma/gamma_);
//	gamma_ = gamma;
//	time(&t2);	Public::ShowTime((t2-t1)/60.0, "SMO_Packed_Pre£º"); printf_s("\n\n"); time(&t1);
//}
//
//void SVM_Trainer::SMO_Packed( const double C, const double gamma)
//{
//	time_t t1=0, t2=0;	time(&t1);
//	printf_s("num_cv_samples_=%d, num_train_samples_=%d\n", num_cv_samples_, num_train_samples_);
//	SMO_Packed_Pre(gamma);
//	time(&t2);	Public::ShowTime((t2-t1)/60.0, "SMO_Packed_Pre£º"); printf_s("\n\n"); time(&t1);
//	SMO_PackedMain(C);
//	time(&t2);	Public::ShowTime((t2-t1)/60.0, "SMO_PackedMain£º"); printf_s("\n\n"); time(&t1);
//}
//
//void SVM_Trainer::SMO_PackedMain( const double C)
//{
//	memset(alpha_, 0, num_train_samples_*sizeof(double) );
//	InitGradient();
//	int i = 0, j = 0, loop_times = 0;
//	while(SelectWorkingSetPacked(C, i, j))
//	{
//		double old_alpha_i = alpha_[i];
//		double old_alpha_j = alpha_[j];
//		UpdateAlpha(C, i, j);
//		UpdateGradient(alpha_[i]-old_alpha_i, alpha_[j]-old_alpha_j, i, j);
//		loop_times++;
//	}
//	ComputeRho(C);
//	GetSupportVectorsIndex();
//	printf_s("loop_times=%d, num_support_vectors_=%d\n", loop_times, num_support_vectors_);
//}

//void SVM_Trainer::SavePackedCovMatrix(const long num_samples, char* filename_packed_cov_matrix)
//{
//	if (num_samples%2 == 1)
//	{
//		Public::SaveMatrix(cov_matrix_, num_samples, (num_samples+1)/2, filename_packed_cov_matrix);  // cov_matrix_[i, j] == packed_cov_matrix[j + (long long)i*(i+1)/2]
//	}
//	else
//	{
//		Public::SaveMatrix(cov_matrix_, num_samples+1, num_samples/2, filename_packed_cov_matrix);  // cov_matrix_[i, j] == packed_cov_matrix[j + (long long)i*(i+1)/2]
//	}	
//}


//
//void SVM_Trainer::CalculateRBF_PackedCovMatrixMKL(char* filename_packed_cov_matrix)
//{
//	Public::AllocArray(cov_matrix_, (long long)num_train_samples_ * (long long)num_train_samples_);
//	cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, num_train_samples_, num_features_, 1.0, train_set_->data(), num_features_,
//							0.0, cov_matrix_, num_train_samples_); 
//	LAPACKE_dtrttp(LAPACK_COL_MAJOR, 'U', num_train_samples_, cov_matrix_, num_train_samples_, cov_matrix_);
//	CalculateRBF_PackedCovMatrix(cov_matrix_);
//	SavePackedCovMatrix(num_train_samples_, filename_packed_cov_matrix);
//}

//// cov_matrix_[i, j] == packed_cov_matrix[j + (long long)i*(i+1)/2]
//void SVM_Trainer::LoadPackedCovMatrix(char* filename)
//{
//	Public::LoadMatrix(packed_cov_matrix_, filename);
//}//
//void SVM_Trainer::CalculateRBF_PackedCovMatrix(double* packed_cov_matrix)
//{
//	double* diag_cov = new double [num_train_samples_];
//#pragma omp parallel for
//	for (long long i=0; i<num_train_samples_; i++)
//	{
//		diag_cov[i] = packed_cov_matrix[i + i*(i+1)/2];
//	}
//	CalculateRBF_PackedCovMatrixSub(diag_cov, packed_cov_matrix);
//	const long long size_packed_cov_matrix = (long long)num_train_samples_ * (long long)(num_train_samples_+1) /2; 
//	ExpLongVector(packed_cov_matrix, size_packed_cov_matrix);
//	delete[] diag_cov;	
//}
//void SVM_Trainer::CalculateRBF_PackedCovMatrixSub(const double* diag_cov, double* packed_cov_matrix)
//{
//#pragma omp parallel for
//	for (int i=0; i<num_train_samples_; i++)
//	{
//		long long xxi = (long long)i*(i+1)/2;
//		for (int j=0; j<=i; j++)
//		{
//			packed_cov_matrix[j + xxi] = - diag_cov[i] - diag_cov[j] + 2*packed_cov_matrix[j + xxi];
//		}
//	}
//}

//
//double SVM_Trainer::PackedCovMatrix(const int i, const int j)
//{
//	double q_ij = 0;
//	if (i >= j)
//	{
//		q_ij = packed_cov_matrix_[j + (long long)i*(i+1)/2]; 
//	}
//	else
//	{
//		q_ij = packed_cov_matrix_[i + (long long)j*(j+1)/2]; 
//	}
//	return q_ij;
//}//
//bool SVM_Trainer::SelectWorkingSetPacked(const double C, int &first, int &second)
//{
//	double g_max = SelectFirstElement(C, first);
//	GetQmatrixRowFromePacked(q_matrix_i_, first);
//	double g_min =	SelectSecondElement(C, first, g_max, second);
//	GetQmatrixRowFromePacked(q_matrix_j_, second);
//	return ( (g_max - g_min) > EPSILON_);
//}

//
//void SVM_Trainer::GetQmatrixRowFromePacked(double *q_matrix_i, const int i)
//{
//	const double *src = packed_cov_matrix_ +  (long long)i * (long long)(i+1) / 2;	
//#pragma omp parallel for
//	for (long j=0; j<=i; j++)
//	{
//		q_matrix_i[j] = src[j];
//	}
//#pragma omp parallel for
//	for (long long j=i+1; j<num_train_samples_; j++)
//	{
//		q_matrix_i[j] = packed_cov_matrix_[i + j*(j+1)/2];
//	}
//}


//void SVM_Trainer::MultiplyLabels2PackedQmatrix() 
//{
//	//#pragma omp parallel for
//	for (long long i=0; i<num_train_samples_; i++)
//	{
//		for (long long j=0; j<=i; j++)
//		{
//			packed_cov_matrix_[j + i*(i+1)/2] *= train_set_->labels(i)*train_set_->labels(j);
//		}
//	}
//}