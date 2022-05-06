#include <iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <pthread.h>
#include <semaphore.h>

#define n 4096
#define thread_count 4
using namespace std;

static float A[n][n];
int id[thread_count];
sem_t sem_parent;//主线程
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;
long long head, tail, freq;

void init()//初始化
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            A[i][j]=i+j;
        }

    }
}

void printA()
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            cout<<A[i][j]<<" ";
        cout<<endl;
    }
}

void * processbycol(void * ID)
{
    int * threadid= (int*)ID;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        for(int i=begin;i<end;i++)
        {
            A[k][i]=A[k][i]/A[k][k];
        }
        for(int i=k+1;i<n;i++)
        {
            for(int j=begin;j<end;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_col()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbycol,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
        //所有子线程执行完毕
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;
        pthread_barrier_wait(&childbarrier_col);
    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbyrow(void * ID)
{
    int* threadid= (int*)ID;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        for(int i=begin;i<end;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);//唤醒主线程，信号量
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_row()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1;
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbyrow,(void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for(int i=0;i<thread_count;i++)
        {
            sem_wait(&sem_parent);
        }
    }
    pthread_barrier_wait(&childbarrier_row);
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbyrow_SSE(void * ID)
{
    int* threadid= (int*)ID;
    __m128 t1,t2,t3;//四位单精度构成的向量
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%4;//预处理数量
        int begincol=k+1+preprocessnumber;
        for(int i=begin;i<end;i++)
        {
            for(int j=k+1;j<preprocessnumber;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
        for(int i=begin;i<end;i++)
        {
            float head1[4]={A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm_loadu_ps(head1);
            for(int j=begincol;j<n;j+=4)
            {
                t1=_mm_loadu_ps(A[k]+j);
                t2=_mm_loadu_ps(A[i]+j);
                t1=_mm_mul_ps(t1,t3);
                t2=_mm_sub_ps(t2,t1);
                _mm_store_ss(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);//唤醒主线程，信号量
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_row_SSE()
{
    pthread_t threadID[thread_count];
    __m128 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        int preprocessnumber=(n-k-1)%4;//预处理数量
        int begin=k+1+preprocessnumber;
        float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=_mm_loadu_ps(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=begin;j<n;j++)
        {
            //A[k][j]=A[k][j]/A[k][k];
            t1=_mm_loadu_ps(A[k]+j);
            t1=_mm_div_ps(t1,t2);
            _mm_store_ss(A[k]+j,t1);
        }
        A[k][k]=1;
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbyrow_SSE,(void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
    }
    pthread_barrier_wait(&childbarrier_row);
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbycol_SSE(void * ID)
{
    int * threadid= (int*)ID;
    __m128 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%4;//预处理数量
        int beginrow=k+1+preprocessnumber;
        float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=_mm_loadu_ps(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=beginrow;j<end;j+=4)
        {
            t1=_mm_loadu_ps(A[k]+j);
            t1=_mm_div_ps(t1,t2);
            _mm_store_ss(A[k]+j,t1);
        }
        t1=_mm_setzero_ps();
        t2=_mm_setzero_ps();

        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<k+1+preprocessnumber;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }

        for(int i=k+1;i<n;i++)
        {
            float head1[4]={A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm_loadu_ps(head1);
            for(int j=beginrow;j<end;j+=4)
            {
                t1=_mm_loadu_ps(A[k]+j);
                t2=_mm_loadu_ps(A[i]+j);
                t1=_mm_mul_ps(t1,t3);
                t2=_mm_sub_ps(t2,t1);
                _mm_store_ss(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_col_SSE()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbycol_SSE,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
        //所有子线程执行完毕
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;
        pthread_barrier_wait(&childbarrier_col);
    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbyrow_AVX(void * ID)
{
    int* threadid= (int*)ID;
    __m256 t1,t2,t3;//四位单精度构成的向量
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%8;//预处理数量
        int begincol=k+1+preprocessnumber;
        for(int i=begin;i<end;i++)
        {
            for(int j=k+1;j<preprocessnumber;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
        for(int i=begin;i<end;i++)
        {
            float head1[8]={A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm256_loadu_ps(head1);
            for(int j=begincol;j<n;j+=8)
            {
                t1=_mm256_loadu_ps(A[k]+j);
                t2=_mm256_loadu_ps(A[i]+j);
                t1=_mm256_mul_ps(t1,t3);
                t2=_mm256_sub_ps(t2,t1);
                _mm256_storeu_ps(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);//唤醒主线程，信号量
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_row_AVX()
{
    pthread_t threadID[thread_count];
    __m256 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        int preprocessnumber=(n-k-1)%8;//预处理数量
        int begin=k+1+preprocessnumber;
        float head[8]={A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=_mm256_loadu_ps(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=begin;j<n;j++)
        {
            t1=_mm256_loadu_ps(A[k]+j);
            t1=_mm256_div_ps(t1,t2);
            _mm256_storeu_ps(A[k]+j,t1);
        }
        A[k][k]=1;
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbyrow_AVX,(void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
    }
    pthread_barrier_wait(&childbarrier_row);
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbycol_AVX(void * ID)
{
    int * threadid= (int*)ID;
    __m256 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%8;//预处理数量
        int beginrow=k+1+preprocessnumber;
        float head[8]={A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=_mm256_loadu_ps(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=beginrow;j<end;j+=8)
        {
            t1=_mm256_loadu_ps(A[k]+j);
            t1=_mm256_div_ps(t1,t2);
            _mm256_storeu_ps(A[k]+j,t1);
        }
        t1=_mm256_setzero_ps();
        t2=_mm256_setzero_ps();

        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<k+1+preprocessnumber;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }

        for(int i=k+1;i<n;i++)
        {
            float head1[4]={A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm256_loadu_ps(head1);
            for(int j=beginrow;j<end;j+=8)
            {
                t1=_mm256_loadu_ps(A[k]+j);
                t2=_mm256_loadu_ps(A[i]+j);
                t1=_mm256_mul_ps(t1,t3);
                t2=_mm256_sub_ps(t2,t1);
                _mm256_storeu_ps(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_col_AVX()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbycol_AVX,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
        //所有子线程执行完毕
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;
        pthread_barrier_wait(&childbarrier_col);
    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void normal_gauss()
{
    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
    }
}

int main()
{
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    pthread_barrier_init(&childbarrier_row, NULL,thread_count+1);
    pthread_barrier_init(&childbarrier_col,NULL, thread_count+1);
    sem_init(&sem_parent, 0, 0);//信号量初始化
    for(int i=0;i<thread_count;i++)
        id[i]=i;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_col();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_col："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_row();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_row："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    normal_gauss();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"normal_gauss："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_row_SSE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_row_SSE："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_col_SSE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_col_SSE："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_row_AVX();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_row_AVX："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_col_AVX();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"pthread_row_AVX："<<(tail-head)*1000.0/freq<<"ms"<< endl;

    sem_destroy(&sem_parent);
    pthread_barrier_destroy(&childbarrier_col);
    pthread_barrier_destroy(&childbarrier_row);
	return 0;
}
