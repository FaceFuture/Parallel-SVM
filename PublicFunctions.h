#pragma once

#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>
using namespace std;

class Public
{
public:
    Public(void);
    ~Public(void);

    static void ShowTime(const double theTime, char* timeDescription)
    {
		if (theTime < 1)
		{
			printf_s("%s%6.3lf秒 ", timeDescription, theTime*60);
		}
        else if (theTime < 60)
        {
            printf_s("%s%6.3lf分钟 ", timeDescription, theTime);
        }
        else if (theTime < 60*24)
        {
			long num_hours = (long)(theTime/60);
			printf_s("%s%2d小时%6.3lf分钟.", timeDescription, num_hours, theTime-num_hours*60.0);
        }
        else
        {
			long num_days = (long)(theTime/60/24);
			long num_hours = (long)(theTime/60 - num_days*24);
			if (num_hours > 0)
			{
				printf_s("%s%2d天%2d小时%6.3lf分钟.", timeDescription, num_days, num_hours, theTime - num_days*24*60.0 - num_hours*60.0);
			}
			else
			{
				printf_s("%s%2d天%6.3lf分钟.", timeDescription, num_days, theTime - num_days * 24 * 60.0 - num_hours*60.0);
			}
        }
    }

    static void ShowTime(const time_t &start, const long total, const long past)
    {
        time_t now = 0;
        time( &now );
		double usedTime = (now-start)/60.0;
		double leftTime = usedTime*(total-past)/(past);
		printf_s("循环次数%3d/%d, ", past, total);
		ShowTime(usedTime, "已用时间");
		ShowTime(leftTime, "剩余时间");  
        printf_s("\n");
    }

	static void ShowTime(const double usedTime, const long total, const long past)
	{
			double leftTime = usedTime*(total-past)/(past);
			printf_s("循环次数%3d/%d, ", past, total);
			ShowTime(usedTime, "已用时间");
			ShowTime(leftTime, "剩余时间");  
			printf_s("\n");
	}

	static void ShowTime(const time_t &t1, time_t &t2, time_t &t3, long num_class, const long num_samples_this_class, char * filename ) 
	{
		time(&t3); 
		if ( ((long)(t3-t2)) > 60*5)
		{
			t2 = t3;
			printf("nClass=%5ld，num_samples=%5ld，", num_class, num_samples_this_class);
			Public::ShowTime((t3-t1)/60.0, "已用时间 ");	printf("\n");
			FILE* fid = NULL;
			fopen_s(&fid, filename, "at");
			fprintf(fid, "nClass=%5ld，num_samples=%5ld，已用时间%6.2lf小时\n", num_class, num_samples_this_class, (t3-t1)/3600.0);
			fclose(fid);
		}
	}
	
	template <class T>
	static void ReleaseArray(T* &arrayX)
	{
		if (arrayX != NULL)
		{
			delete[] arrayX;
			arrayX = NULL;
		}
	}

	template <class T>
	static void AllocArray(T* &arrayX, const long long dim1)
	{
		ReleaseArray(arrayX);
		arrayX = new T [dim1];
		memset(arrayX, 0, sizeof(T)*dim1);
	}

	template <class T>
	static void Alloc2dArray(T** &array2d, const long dim1, const long long dim2)
	{
		Release2dArray(array2d, dim1);
		array2d = new T* [dim1];
		for (long i=0; i<dim1; i++)
		{
			array2d[i] = new T [dim2];
			memset(array2d[i], 0, sizeof(T)*dim2);
		}
	}

	template <class T>
	static void Alloc3dArray(T*** &array3d, const long dim1, const long dim2, const long dim3)
	{
		Release3dArray(array3d, dim1, dim2);
		array3d = new T** [dim1];
		for (long i=0; i<dim1; i++)
		{
			Alloc2dArray(array3d[i], dim2, dim3);
		}
	}

	template <class T>
	static void Release2dArray(T** &arrayX, const long num)
	{
		if (arrayX != NULL)
		{
			for (long i=0; i<num; i++)
			{
				ReleaseArray(arrayX[i]);
			}
			ReleaseArray(arrayX);
		}
	}

	template <class T>
	static void Release3dArray(T*** &arrayX, const long dim1, long dim2)
	{
		if (arrayX != NULL)
		{
			for (long i=0; i<dim1; i++)
			{
				Release2dArray(arrayX[i], dim2);
			}
			ReleaseArray(arrayX);
		}
	}
	template <class T>
	static const double Distance(const T x1, const T y1, const T x2, const T y2)
	{
		double distX = (double)x1 - (double)x2;
		double distY = (double)y1 - (double)y2;
		return sqrt( distX*distX + distY*distY );
	}

	static void PrintRate(double* rates, const long n, char* filename)
	{
		//char* filename = "E:/rates.txt";
		FILE* fid = NULL;
		fopen_s(&fid, filename, "at");
		for (long i=0; i<n; i++)
		{
			fprintf_s(fid, "%12.6lf", rates[i]);
		}
		fprintf_s(fid, "\n");
		fclose(fid);
	}


template <typename T>
	static void Normalize01(T* feature, const long num_dims)  // DownSampling::Normalize01 很重要！！！ 少了，识别率大大降低！（日記2012.3.20）
	{
		T min_value = *min_element(feature, feature+num_dims);
		T max_value = *max_element(feature, feature+num_dims);
		double span = max_value - min_value;
		if (span == 0.0)
		{
			printf_s("span == 0, min_value=%lf, max_value=%lf, num_dims=%ld", min_value, max_value, num_dims);
			exit(0);
		}
		double inv_span = 1.0 / span;
		for (long i=0; i<num_dims; i++)
		{
			feature[i] = (feature[i] - min_value)  * inv_span;
		}
	}

template <typename T>
	static void Normalize256(const T* feature, const long num_dims, char* data)
	{
		T min_value = *min_element(feature, feature+num_dims);
		T max_value = *max_element(feature, feature+num_dims);
		double span = max(max_value, -min_value);
		if (span == 0.0)
		{
			printf_s("ExtractDatasetWithMultiLDA::Normalize256 span == 0, min_value=%lf, max_value=%lf, num_dims=%ld", min_value, max_value, num_dims);
			exit(0);
		}
		double inv_span = 1.0 / span;
		for (long i=0; i<num_dims; i++)
		{		
			const long n = 128;	//(256*256);
			long xxx =  (long) floorl(feature[i] * inv_span * n) ;
			xxx =  (xxx < n) ? xxx : n-1;
			data[i] = (char)xxx;
		}
	}

	static void LoadData(char* start, const long long size, FILE* fid)
	{
		//fread(matrix, sizeof(TypeX), size_x, fid);
		const long long buff_size = 1024 * 1024 * 4; //4M
		const long long nn = size / buff_size;
		const long long left_size = size - nn*buff_size;
		for (long long i=0; i<nn; i++, start+=buff_size)
		{
			fread(start, sizeof(char), buff_size, fid);
		}		
		fread(start, sizeof(char), left_size, fid);
	}


	static void SaveData(char* start, const long long size, FILE* fid)
	{		
		const long long buff_size = 1024 * 1024 * 64; //64M
		const long long nn = size / buff_size;
		const long long left_size = size - nn*buff_size;
		//printf_s("size=%lld, buff_size=%lld, nn=%lld, left_size=%lld\n\n", size, buff_size, nn, left_size);
		for (long long i=0; i<nn; i++, start+=buff_size)
		{
			fwrite(start, sizeof(char), buff_size, fid);
		}		
		fwrite(start, sizeof(char), left_size, fid);
	}

	template <typename TypeX>
	static void SaveMatrix(const TypeX* matrix, const long num_rows, const long num_cols, char* filename)
	{
		FILE* fid = NULL;
		fopen_s(&fid, filename, "wb");
		fwrite(&num_rows, sizeof(long), 1, fid);
		fwrite(&num_cols, sizeof(long), 1, fid);
		//fwrite(matrix, sizeof(TypeX), (long long)num_rows * (long long)num_cols, fid);
		const long long size = (long long)num_rows * (long long)num_cols * (long long)( sizeof(TypeX) );
		SaveData((char*)matrix, size, fid);
		fclose(fid);
	}

	template <typename TypeX>
	static void SaveMatrix3d(TypeX** matrix,const long num_cols,  const long num_rows, const long num_dim3, char* filename)
	{
		FILE* fid = NULL;
		fopen_s(&fid, filename, "wb");
		fwrite(&num_cols, sizeof(long), 1, fid);
		fwrite(&num_rows, sizeof(long), 1, fid);
		fwrite(&num_dim3, sizeof(long), 1, fid);
		for (long i=0; i<num_dim3; i++)
		{
			fwrite(matrix[i], sizeof(TypeX), (long long)num_rows * (long long)num_cols, fid);
		}
		fclose(fid);
	}

	template <typename TypeX>
	static void SaveMatrix2d(TypeX** matrix, const long long num_cols, const long num_rows, char* filename)
	{
		FILE* fid = NULL;
		fopen_s(&fid, filename, "wb");
		fwrite(&num_cols, sizeof(long), 1, fid);
		fwrite(&num_rows, sizeof(long), 1, fid);
		for (int i=0; i<num_rows; i++)
		{
			fwrite(matrix[i], sizeof(TypeX), num_cols, fid);
		}
		fclose(fid);
	}


	template <typename TypeX>
	static void LoadMatrix(TypeX* &matrix,  long &num_rows, long &num_cols, char* filename)
	{
		FILE* fid = NULL;
		if ( 0 == fopen_s(&fid, filename, "rb") )
		{
			fread(&num_rows, sizeof(long), 1, fid);
			fread(&num_cols, sizeof(long), 1, fid);
			printf_s("num_rows=%ld, num_cols=%ld\n", num_rows, num_cols);
			const long long size = (long long)num_rows * (long long)num_cols * (long long) ( sizeof(TypeX) );
			if (matrix == NULL)
			{
				matrix = new TypeX [(long long)num_rows * (long long)num_cols];
			}
			LoadData((char*)matrix, size, fid);
			fclose(fid);
		}
		else
		{
			printf_s("%s file read error!\n", filename); exit(0);
		}
	}


	template <typename TypeX>
	static void LoadMatrix(TypeX* &matrix, char* filename)
	{
		long num_cols = 0, num_rows = 0;
		LoadMatrix(matrix, num_rows, num_cols, filename);
	}

	template <typename TypeX>
	static void LoadPartialMatrix(TypeX* &matrix,  long num_rows, long num_partial_cols, char* filename)
	{
		FILE* fid = NULL;
		fopen_s(&fid, filename, "rb");
		const long long size = (long long)num_rows * (long long)num_partial_cols * (long long) ( sizeof(TypeX) );
		if (matrix == NULL)
		{
			matrix = new TypeX [(long long)num_rows * (long long)num_partial_cols];
		}
		LoadData((char*)matrix, size, fid);
		fclose(fid);
	}

	template <typename TypeX>
	static void LoadData2d(TypeX** &matrix, const long dim1, const long long dim2, FILE* fid)
	{
		Public::Alloc2dArray(matrix, dim1, dim2);
		for (long i=0; i<dim1; i++)
		{
			fread(matrix[i], sizeof(TypeX), dim2, fid);
		}
	}

	template <typename TypeX>
	static void LoadMatrix2d(TypeX** &matrix, long &num_cols, long &num_rows, char* filename)
	{
		FILE* fid = NULL;
		if ( 0 != fopen_s(&fid, filename, "rb") )
		{
			printf_s("%s was not opened\n", filename);
			exit(0);
		}
		fread(&num_cols, sizeof(long), 1, fid);
		fread(&num_rows, sizeof(long), 1, fid);
		LoadData2d(matrix, num_rows, num_cols, fid);
		fclose(fid);
	}

	template <typename TypeX>
	static void LoadMatrix3d(TypeX** &matrix, long &num_cols, long &num_rows, long &num_dim3, char* filename)
	{
		FILE* fid = NULL;
		if ( 0 != fopen_s(&fid, filename, "rb") )
		{
			printf_s("%s was not opened\n", filename);
			exit(0);
		}
		fread(&num_cols, sizeof(long), 1, fid);
		fread(&num_rows, sizeof(long), 1, fid);
		fread(&num_dim3, sizeof(long), 1, fid);
		LoadData2d(matrix, num_dim3, (long long)num_rows * (long long)num_cols, fid);
		fclose(fid);
	}

	template <typename TypeX>
	static void CopyUpperTriangularMatrix2Lower( TypeX* cov_matrix, const long num_dims)
	{
		for (long long i=0; i<num_dims-1; i++)
		{
			for (long long j=i+1; j<num_dims; j++)
			{
				cov_matrix[i*num_dims + j] = cov_matrix[j*num_dims + i];
			}
		}
	}

	template <typename TypeX>
	static void CopyLowerTriangularMatrix2Upper( TypeX* cov_matrix, const long num_dims)
	{
		for (long long i=0; i<num_dims-1; i++)
		{
			for (long long j=i+1; j<num_dims; j++)
			{
				 cov_matrix[j*num_dims + i] = cov_matrix[i*num_dims + j];
			}
		}
	}

};
