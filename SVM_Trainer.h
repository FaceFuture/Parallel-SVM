#pragma once

template < typename Real> class DataSet;
class ParticleSwarmOptimizer;

class SVM_Trainer
{
protected:
	static const double EPSILON_, TAU_;
	char* output_filename_;

	long num_train_samples_, num_cv_samples_, num_features_;
	long num_support_vectors_;
	double rho_, gamma_;
	DataSet<double> *train_set_;
	DataSet<double> *cross_validation_set_;
	double *cov_matrix_, *train_cv_matrix_;
	double *alpha_, *gradient_;
	long *support_vectors_index_;
	double *q_matrix_i_, *q_matrix_j_;

	double  *yg_;

public:
	SVM_Trainer(void);
	SVM_Trainer(const char* filename_train_set, const char* filename_cross_validation_set);
	virtual ~SVM_Trainer(void);

	//-------------------------------------------------------------------------------
	void LoadLabelsData(const char* filename_train_set, const char* filename_cross_validation_set);
	void LoadLabels(const char* filename_train_set, const char* filename_cross_validation_set);

	void CalculateRBF_CovMatrixMKL(char* filename_cov_matrix);
	void CalculateRBF_Train_CV_MatrixMKL(char* filename_train_cv_matrix);
	void LoadCovMatrix(char* filename);
	void LoadTrain_CV_Matrix(char* filename);

	void SMO_Grid();
	void SMO_NestPSO();
	void SMO_PSO();
	double Test();


protected:
	void SMO_PSO_C_Once(int np_C, ParticleSwarmOptimizer* pso_C);

	void ExpLongVector(double* long_vector, const long long size);
	void PowerLongVector(double* long_vector, const long long size, const double gamma);
	void ExpLongVectorMKL(double* long_vector, const long long size);
	void PowerLongVectorMKL(double* long_vector, const long long size, const double gamma);

	void CalculateRBF_Train_CV_Matrix(double* train_cv_matrix);
	void MultiDotProduct(const double* samples, const long num_samples, const long num_features, double* dot_products);
	virtual void Init();
	void SetOutputFilename();
	void InitGradient();
	virtual double Select1stElement(const double C, int &first);
	virtual double Select2ndElement(const double C, const int first, const double g_max, int& second);
	bool SelectWorkingSet(const double C, int &first, int &second);
	//virtual bool SelectWorkingSetOMP(const double C, int &first, int &second){}

	void UpdateAlpha(const double C, const int first, const int second);
	void UpdateGradient(const double delta_ahpha_i, const double delta_ahpha_j);
	inline double GetAij(const int i, const int j);
	inline void ProjectAlpha2FeasibleRegion(const double C, double& alpha);
	void ComputeRho(const double C);
	void GetSupportVectorsIndex();

	void CalculateCovMatrixRBF(double* cov_matrix);
	void CalculateCovMatrix(double* cov_matrix);
	void SMO_Pre(const double gamma);
	void MultiplyLabels2Qmatrix();

	void SMO_Main( const double C );


	void SMO_PSO_C_Evolve(ParticleSwarmOptimizer* pso_C, unsigned int seed);
	void PrintOutPSO_C(ParticleSwarmOptimizer* pso_C);
	void PrintOutPSO_Gamma(ParticleSwarmOptimizer* pso_gamma);
	void PrintOutPSO(ParticleSwarmOptimizer* pso);
	void MultiplyLabels2QmatrixMKL();
	inline void UpdateGradientMKL(const double delta_ahpha_i, const double delta_ahpha_j);
	void InitSelectWorkingSet();
};




// void CalculateRBF_PackedCovMatrixMKL(char* filename_packed_cov_matrix);
// void LoadPackedCovMatrix(char* filename);
// void SMO_Packed(const double C, const double gamma);	
//void CalculateRBF_PackedCovMatrix(double* packed_cov_matrix);
//void CalculateRBF_PackedCovMatrixSub(const double* diag_cov, double* packed_cov_matrix);
//void SavePackedCovMatrix(const long num_samples, char* filename_packed_cov_matrix);
//bool SelectWorkingSetPacked(const double C, int &first, int &second);
//inline double PackedCovMatrix(const int i, const int j);
//void MultiplyLabels2PackedQmatrix();
//void GetQmatrixRowFromePacked(double *q_matrix_i, const int i);
//void SMO_Packed_Pre(const double gamma);
//void SMO_PackedMain( const double C);


