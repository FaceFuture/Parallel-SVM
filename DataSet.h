#pragma once
#include <stdlib.h>
#include "PublicFunctions.h"

template <typename data_type>
class DataSet
{
private:
	long num_features_, num_samples_, num_samples_class1_;
	long long data_size_;
	long *labels_;
	data_type *data_;

public:
	DataSet(void)  :  num_features_(0), num_samples_(0), num_samples_class1_(0), labels_(NULL), data_(NULL)
	{	
	}
	DataSet(const char* fileName) : num_features_(0), num_samples_(0), num_samples_class1_(0), labels_(NULL), data_(NULL)
	{
		LoadDataLabels(fileName);
	}
	~DataSet(void)
	{
		Clear();
	}

	//----------------------------------------------------------------------------------------------------------------------------------------------------------
    const long num_features(void) const
    {
        return num_features_;
    }
	void num_features(const long num)
	{
		num_features_ = num;
	}

    const long num_samples(void) const
    {
        return num_samples_;
    }
	void num_samples(const long num)
	{
		num_samples_ = num;
	}

	const long num_samples_class1(void) const
	{
		return num_samples_class1_;
	}

    const long labels(const long i) const
    {
        return labels_[i];
    }
    const long* labels() const
    {
        return labels_;
    }

	const data_type* data(void) const
	{
		return (const data_type*)data_;
	}

	data_type* data(const long n_sample) const
	{
		return (data_type*)(data_ + (MKL_INT64)n_sample * (MKL_INT64)num_features_);
	}
	
	//----------------------------------------------------------------------------------------------------------------------------------------------------------
	void LoadLabels(const char* fileName)
	{
		FILE* fid = NULL;
		if (0 == fopen_s(&fid, fileName, "rb"))
		{
			fread_s(&num_features_, sizeof(long), sizeof(long), 1, fid);
			fread_s(&num_samples_, sizeof(long), sizeof(long), 1, fid);
			Public::AllocArray(labels_, num_samples_);
			fread_s(labels_, sizeof(long)*num_samples_, sizeof(long), num_samples_, fid);
			fclose(fid);
			GetSamplesNumberClass1();
		}		
		else
		{
			printf_s("%s open error!\n", fileName);
		}
	}

	void GetSamplesNumberClass1()
	{
		for (int i=1; i<num_samples_; i++)
		{
			if (labels_[i] != labels_[i-1])
			{
				num_samples_class1_ = i;
				printf_s("num_samples_class1_ = %ld\n", num_samples_class1_);
				break;
			}
		}
	}


	void LoadDataLabels(const char* fileName)
	{
		FILE* fid = NULL;
		if (0 == fopen_s(&fid, fileName, "rb"))
		{
			fread_s(&num_features_, sizeof(long), sizeof(long), 1, fid);
			fread_s(&num_samples_, sizeof(long), sizeof(long), 1, fid);
			Public::AllocArray(labels_, num_samples_);
			fread_s(labels_, sizeof(long)*num_samples_, sizeof(long), num_samples_, fid);
			//fread_s(data_, data_size_, sizeof(char), data_size_, fid);  //  若sizeof(data_type)*data_size_ 超过2G，会不会出问题？
			data_size_ = (long long)num_samples_ * (long long)num_features_;
			Public::AllocArray(data_, data_size_);
			Public::LoadData((char*)data_, data_size_*sizeof(data_type), fid);
			fclose(fid);
			GetSamplesNumberClass1();
		}		
		else
		{
			printf_s("%s open error!\n", fileName);
		}
	}

	void SaveDataLabels(const char* fileName)
	{
		FILE* fid = NULL;
		if (0 == fopen_s(&fid, fileName, "wb"))
		{
			fwrite(&num_features_, sizeof(long), 1, fid);
			fwrite(&num_samples_, sizeof(long), 1, fid);
			fwrite(labels_, sizeof(long), num_samples_, fid);
			//fread_s(data_, data_size_, sizeof(char), data_size_, fid);  //  若sizeof(data_type)*data_size_ 超过2G，会不会出问题？
			Public::SaveData((char*)data_, data_size_, fid);
			fclose(fid);
		}		
		else
		{
			printf_s("%s open error!\n", fileName);
		}
	}

private:
	void Clear(void)
	{       
		Public::ReleaseArray(labels_);
		Public::ReleaseArray(data_);
		num_features_ = num_samples_ = 0;
		data_size_ = 0;
	}

	void Alloc(void)
	{		
		if (labels_ != NULL)
		{
			delete[] labels_;
		}
		labels_ = new long [num_samples_];
		
		if (data_ != NULL)
		{
			delete[] data_;
		}
		data_ = (data_type*) (new unsigned char [data_size_]);
	}
};