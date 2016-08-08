#include "StdAfx.h"
#include "ParticleSwarmOptimizer.h"
#include "PublicFunctions.h"
#include "mkl.h"

const double ParticleSwarmOptimizer::W_MAX_ = 0.9;
const double ParticleSwarmOptimizer::W_MIN_ = 0.4;
const double ParticleSwarmOptimizer::C1_ = 2.0;
const double ParticleSwarmOptimizer::C2_ = 2.0;

ParticleSwarmOptimizer::ParticleSwarmOptimizer(void)
{
	num_particles_ = dim_particle_ = num_generations_ = i_generation_ = 0;
	particles_ = velocities_ = pbest_ = gbest_ = fitnesses_ = pbest_fitnesses_ = gbest_fitnesses_ = NULL;
	max_value_particles_ = min_value_particles_ = rand1_ = rand2_ = NULL;
}

ParticleSwarmOptimizer::ParticleSwarmOptimizer( const long num_particles, const long dim_particle, const long num_generations, const double* max_values, const double* min_values)
{
	Init(num_particles, dim_particle, num_generations, max_values, min_values);
}


ParticleSwarmOptimizer::~ParticleSwarmOptimizer(void)
{
	Public::ReleaseArray(particles_);
	Public::ReleaseArray(velocities_);
	Public::ReleaseArray(pbest_);
	Public::ReleaseArray(gbest_);
	Public::ReleaseArray(fitnesses_);
	Public::ReleaseArray(pbest_fitnesses_);
	Public::ReleaseArray(gbest_fitnesses_);
	Public::ReleaseArray(max_value_particles_);
	Public::ReleaseArray(min_value_particles_);
	Public::ReleaseArray(rand1_);
	Public::ReleaseArray(rand2_);
	Public::ReleaseArray(output_filename_);
}

void ParticleSwarmOptimizer::Init( const long num_particles, const long dim_particle, const long num_generations, const double* max_values, const double* min_values )
{
	i_generation_ = 0;
	num_particles_ = num_particles;
	dim_particle_ = dim_particle;
	num_generations_ = num_generations;
	particles_ = velocities_ = pbest_ = gbest_ = fitnesses_ = pbest_fitnesses_ = gbest_fitnesses_ = NULL;
	max_value_particles_ = min_value_particles_ = rand1_ = rand2_ = NULL;

	Public::AllocArray(particles_, dim_particle_ * num_particles_);
	Public::AllocArray(velocities_, dim_particle_ * num_particles_);
	Public::AllocArray(pbest_, dim_particle_ * num_particles_);
	Public::AllocArray(gbest_, dim_particle_ * num_particles_);
	Public::AllocArray(fitnesses_, num_particles_);
	Public::AllocArray(pbest_fitnesses_, num_particles_);
	Public::AllocArray(gbest_fitnesses_, num_particles_);
	Public::AllocArray(max_value_particles_, dim_particle_);
	Public::AllocArray(min_value_particles_, dim_particle_);
	Public::AllocArray(rand1_, num_generations_ * dim_particle_ * num_particles_);
	Public::AllocArray(rand2_, num_generations_ * dim_particle_ * num_particles_);

	SetOutputFilename();
	SetMaxMinValues(max_values, min_values);
}

void ParticleSwarmOptimizer::SetOutputFilename()
{
	const time_t t = time(NULL);
	struct tm* current_time = new struct tm;
	localtime_s(current_time, &t);
	output_filename_ = new char[256];
	sprintf_s(output_filename_, 256, "G:/PSO_data_%02d_%02d_%02d_%02d_%02d.dat",
		current_time->tm_mon + 1, current_time->tm_mday, current_time->tm_hour, current_time->tm_min, current_time->tm_sec);
	delete current_time;
}

void ParticleSwarmOptimizer::Load(char* output_filename)
{
	FILE* fid = NULL;
	fopen_s(&fid, output_filename, "rb");
	fread(&num_particles_, sizeof(long), 1, fid);
	fread(&dim_particle_, sizeof(long), 1, fid);
	fread(&num_generations_, sizeof(long), 1, fid);
	fread(&i_generation_, sizeof(long), 1, fid);

	fread(particles_, sizeof(double), dim_particle_ * num_particles_, fid);
	fread(velocities_, sizeof(double), dim_particle_ * num_particles_, fid);
	fread(pbest_, sizeof(double), dim_particle_ * num_particles_, fid);
	fread(gbest_, sizeof(double), dim_particle_ * num_particles_, fid);
	fread(fitnesses_, sizeof(double), num_particles_, fid);
	fread(pbest_fitnesses_, sizeof(double), num_particles_, fid);
	fread(gbest_fitnesses_, sizeof(double), num_particles_, fid);
	fread(max_value_particles_, sizeof(double), dim_particle_, fid);
	fread(min_value_particles_, sizeof(double), dim_particle_, fid);
	fread(rand1_, sizeof(double), num_generations_ * dim_particle_ * num_particles_, fid);
	fread(rand2_, sizeof(double), num_generations_ * dim_particle_ * num_particles_, fid);

	fclose(fid);
}

void ParticleSwarmOptimizer::Save(char* output_filename)
{
	FILE* fid = NULL;
	if (output_filename == NULL)
	{
		fopen_s(&fid, output_filename_, "wb");
	}
	else
	{
		fopen_s(&fid, output_filename, "wb");
	}
	fwrite(&num_particles_, sizeof(long), 1, fid);
	fwrite(&dim_particle_, sizeof(long), 1, fid);
	fwrite(&num_generations_, sizeof(long), 1, fid);
	fwrite(&i_generation_, sizeof(long), 1, fid);

	fwrite(particles_, sizeof(double), dim_particle_ * num_particles_, fid);
	fwrite(velocities_, sizeof(double), dim_particle_ * num_particles_, fid);
	fwrite(pbest_, sizeof(double), dim_particle_ * num_particles_, fid);
	fwrite(gbest_, sizeof(double), dim_particle_ * num_particles_, fid);
	fwrite(fitnesses_, sizeof(double), num_particles_, fid);
	fwrite(pbest_fitnesses_, sizeof(double), num_particles_, fid);
	fwrite(gbest_fitnesses_, sizeof(double), num_particles_, fid);
	fwrite(max_value_particles_, sizeof(double), dim_particle_, fid);
	fwrite(min_value_particles_, sizeof(double), dim_particle_, fid);
	fwrite(rand1_, sizeof(double), num_generations_ * dim_particle_ * num_particles_, fid);
	fwrite(rand2_, sizeof(double), num_generations_ * dim_particle_ * num_particles_, fid);

	fclose(fid);
}

void ParticleSwarmOptimizer::SetMaxMinValues(const double* max_values, const double* min_values)
{
	memcpy_s( max_value_particles_, sizeof(max_value_particles_[0])*dim_particle_, max_values, sizeof(max_values[0])*dim_particle_ );
	memcpy_s( min_value_particles_, sizeof(min_value_particles_[0])*dim_particle_, min_values, sizeof(min_values[0])*dim_particle_ );
}

void ParticleSwarmOptimizer::InitRandomMKL(const unsigned int SEED)
{
	VSLStreamStatePtr stream;
	vslNewStream(&stream,VSL_BRNG_SFMT19937, SEED);
	for (int i=0; i<dim_particle_; i++)
	{
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, num_particles_, particles_ + i*num_particles_, min_value_particles_[i], max_value_particles_[i]);
	}
	const double K = 0.1;
	for (int i=0; i<dim_particle_; i++)
	{
		double v_max = (max_value_particles_[i] - min_value_particles_[i]) * K;
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, num_particles_, velocities_ + i*num_particles_, -v_max, v_max);
	}
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, num_generations_ * dim_particle_ * num_particles_, rand1_, 0.0, 1.0);
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, num_generations_ * dim_particle_ * num_particles_, rand2_, 0.0, 1.0);

	vslDeleteStream( &stream );
	//i_generation_ = 0;
	//memset(pbest_fitnesses_, 0, num_particles_ * sizeof(pbest_fitnesses_[0]));
	//memset(gbest_fitnesses_, 0, num_particles_ * sizeof(gbest_fitnesses_[0]));
}

void ParticleSwarmOptimizer::InitRandom(const unsigned int SEED)
{
	i_generation_ = 0;
	srand(SEED);
	for (int i = 0; i < dim_particle_; i++)
	{
		double diff = max_value_particles_[i] - min_value_particles_[i];
		for (int j = 0; j < num_particles_; j++)
		{
			particles_[i*num_particles_ + j] = rand() / double(RAND_MAX) * diff + min_value_particles_[i];
		}
	}
	const double K = 0.1;
	for (int i = 0; i < dim_particle_; i++)
	{
		double v_max = (max_value_particles_[i] - min_value_particles_[i]) * K;
		for (int j = 0; j < num_particles_; j++)
		{
			velocities_[i*num_particles_ + j] = rand() / double(RAND_MAX) * 2 * v_max - v_max;
		}
	}
}

void ParticleSwarmOptimizer::UpdateVelocitiesParticles()
{
	double w = W_MAX_ - (W_MAX_ - W_MIN_) * i_generation_ / num_generations_;
	int n_size = dim_particle_ * num_particles_;

	for (int i=0; i<n_size; i++)
	{
		int i_dim = i/num_particles_;
		velocities_[i] = w*velocities_[i] + C1_*rand() / double(RAND_MAX) * (pbest_[i] - particles_[i])
															+ C2_*rand() / double(RAND_MAX) * (gbest_[i_dim] - particles_[i]);
		particles_[i] += velocities_[i];
		while ( (particles_[i] > max_value_particles_[i_dim]) || (particles_[i] < min_value_particles_[i_dim]) )
		{
			if (particles_[i] > max_value_particles_[i_dim])
			{
				particles_[i] = 2*max_value_particles_[i_dim] - particles_[i];
			}
			if (particles_[i] < min_value_particles_[i_dim])
			{
				particles_[i] = 2*min_value_particles_[i_dim] - particles_[i];
			}
		}
	}
	i_generation_++;
}

void ParticleSwarmOptimizer::GetParticleBest()
{
	for (int i=0; i<num_particles_; i++)
	{
		if (fitnesses_[i] > pbest_fitnesses_[i])
		{
			pbest_fitnesses_[i] = fitnesses_[i];
			for (int j=0; j<dim_particle_; j++)
			{
				pbest_[j*num_particles_+i] = particles_[j*num_particles_+i];
			}
		}
	}
}

void ParticleSwarmOptimizer::PushBackGlobalBest(const int i)
{
	for (int j=num_particles_-1; j>i; j--)
	{
		gbest_fitnesses_[j] = gbest_fitnesses_[j-1];
//#pragma unroll
		for (int k=0; k<dim_particle_; k++)
		{
			gbest_[k*num_particles_+j] = gbest_[k*num_particles_+j-1];
		}
	}
}

void ParticleSwarmOptimizer::GetGlobalBestI(const int i, const int idx_max, const double fitnesses_max)
{
	gbest_fitnesses_[i] = fitnesses_max;
//#pragma unroll
	for (int k=0; k<dim_particle_; k++)
	{
		gbest_[k*num_particles_+i] = particles_[k*num_particles_+ idx_max];
	}
}

void ParticleSwarmOptimizer::GetGlobalBest()
{
	for (int j=0; j<num_particles_; j++)
	{
		for (int i=0; i<num_particles_; i++)
		{
			if (fitnesses_[j] > gbest_fitnesses_[i])
			{
				PushBackGlobalBest(i);
				GetGlobalBestI(i, j, fitnesses_[j]);
				break;
			}
		}
	}
	
}

void ParticleSwarmOptimizer::GetBest()
{
	GetParticleBest();
	GetGlobalBest();
}

void ParticleSwarmOptimizer::Evolve()
{
	GetBest();
	UpdateVelocitiesParticles();
}
