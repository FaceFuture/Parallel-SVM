#pragma once
class ParticleSwarmOptimizer
{
private:
	static const double W_MAX_, W_MIN_, C1_, C2_;
	long num_particles_, dim_particle_, num_generations_, i_generation_;
	double *particles_, *velocities_, *max_value_particles_, *min_value_particles_, *rand1_, *rand2_;
	double *pbest_, *gbest_;
	double *fitnesses_,  *pbest_fitnesses_, *gbest_fitnesses_;
	char* output_filename_;

public:
	ParticleSwarmOptimizer(void);
	ParticleSwarmOptimizer(const long num_particles, const long dim_particle, const long num_generations, const double* max_values, const double* min_values);
	~ParticleSwarmOptimizer(void);

	void Init(const long num_particles, const long dim_particle, const long num_generations, const double* max_values, const double* min_values);
	void SetMaxMinValues(const double* max_values, const double* min_values);
	void InitRandomMKL(const unsigned int SEED = 19750815);
	void InitRandom(const unsigned int SEED);
	void Evolve();
	void UpdateVelocitiesParticles();
	void GetBest();

	double rand1(const long i_particles, const long i_dim, const long i_generations)
	{
		return rand1_[i_generations*num_particles_*dim_particle_ + i_dim*num_particles_ + i_particles];
	}

	double particles(const long i_particles, const long i_dim)
	{
		return particles_[i_dim*num_particles_ + i_particles];
	}

	double* fitnesses()
	{
		return fitnesses_;
	}
	double fitnesses(int i)
	{
		return fitnesses_[i];
	}
	void fitnesses(int i, double x)
	{
		fitnesses_[i] = x;
	}
	double gbest(const long i_particles, const long i_dim)
	{
		return gbest_[i_dim*num_particles_ + i_particles];
	}
	double gbest_fitnesses(const long i_particles)
	{
		return gbest_fitnesses_[i_particles];
	}

	const long num_particles()
	{
		return num_particles_;
	}
	const long num_generations()
	{
		return num_generations_;
	}

	void Save(char* output_filename);
	void Load(char* output_filename);
private:
	void GetParticleBest();
	void GetGlobalBest();
	void PushBackGlobalBest(const int i);
	void GetGlobalBestI(const int i, const int idx_max, const double pbest_fitnesses_max);
	void SetOutputFilename();
};

