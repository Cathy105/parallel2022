#include <iostream>
#include <pthread.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <semaphore.h>

#if def_OPENMP
#include<omp.h>
#endif

#define n 2048
#define thread_count 4

using namespace std;

static float A[n][n];//数组
int id[thread_count];
long long head, tail, freq;
sem_t sem_parent;//主线程
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;

void init()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = i + j;
}
void printA()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}
void omp_gauss()
{
#pragma omp parallel num_threads(thread_count)\
    shared(A)
    for (int k = 0; k < n; k++)
    {
#pragma omp for
        //先做完除法
        for (int i = k + 1; i < n; i++)
        {
            A[k][i] = A[k][i] / A[k][k];
        }
        //自动隐式的同步
        A[k][k] = 1;
#pragma omp for
        //更新正方形
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

            A[i][k] = 0;
        }
        //隐式同步
    }
}
void normal_gausseliminate()
{
    for (int k = 0; k < n; k++)
    {

        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//按行划分pthread:
void* dealwithbyrow(void* ID)//处理高斯消去的每一个square线程函数
{
    int* threadid = (int*)ID;
    for (int k = 0; k < n; k++)
    {
        int begin = k + 1 + *threadid * ((n - k - 1) / thread_count);
        int end = begin + (n - k - 1) / thread_count;
        if (end > n)
            end = n;
        for (int i = begin; i < end; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        sem_post(&sem_parent);//唤醒主线程，信号量
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}
void Gauss_pthread_row()
{
    pthread_t threadID[thread_count];//存线程号
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;

        if (k == 0)//若未创建线程则创建线程
        {
            for (int i = 0; i < thread_count; i++)
            {
                pthread_create(&threadID[i], NULL, dealwithbyrow, (void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for (int i = 0; i < thread_count; i++)//
        {
            sem_wait(&sem_parent);
        }

    }
    pthread_barrier_wait(&childbarrier_row);
    for (int i = 0; i < thread_count; i++)
    {
        pthread_join(threadID[i], NULL);
    }
    return;
}
//按列划分pthread
void* dealwithbycol(void* ID)
{
    int* threadid = (int*)ID;
    for (int k = 0; k < n; k++)
    {
        int begin = k + 1 + *threadid * ((n - k - 1) / thread_count);
        int end = begin + (n - k - 1) / thread_count;
        if (end > n)
            end = n;
        for (int i = begin; i < end; i++)
        {
            A[k][i] = A[k][i] / A[k][k];
        }
        for (int i = k + 1; i < n; i++)
        {
            for (int j = begin; j < end; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);
    }
    pthread_exit(NULL);
}
void Gauss_pthread_col()
{
    pthread_t threadID[thread_count];//存线程号
    for (int k = 0; k < n; k++)
    {
        if (k == 0)//若未创建线程则创建线程
        {
            for (int i = 0; i < thread_count; i++)
            {
                pthread_create(&threadID[i], NULL, dealwithbycol, (void*)&id[i]);
            }
        }
        for (int i = 0; i < thread_count; i++)//
        {
            sem_wait(&sem_parent);
        }
        //所有子线程执行完毕
        pthread_barrier_wait(&childbarrier_col);
        A[k][k] = 1;
        for (int i = k + 1; i < n; i++)
            A[i][k] = 0;

    }
    for (int i = 0; i < thread_count; i++)
    {
        pthread_join(threadID[i], NULL);
    }
    return;
}
void omp_SSE_row_gauss()
{
    __m128 t1, t2, t3;//四位单精度构成的向量
# pragma omp parallel num_threads(thread_count)\
    shared(A)
    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int begin = k + 1 + preprocessnumber;
        float head[4] = { A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm_loadu_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
#pragma omp for
        for (int j = begin; j < n; j += 4)
        {
            t1 = _mm_loadu_ps(A[k] + j);
            t1 = _mm_div_ps(t1, t2);
            _mm_store_ss(A[k] + j, t1);
        }
        A[k][k] = 1;
        t1 = _mm_setzero_ps();//清零
        t2 = _mm_setzero_ps();
        //先去头，为了四个四个的处理
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
#pragma omp for
        for (int i = k + 1; i < n; i++)
        {
            float head1[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm_loadu_ps(head1);
            for (int j = begin; j < n; j += 4)
            {
                t1 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_loadu_ps(A[i] + j);
                t1 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t1);
                _mm_store_ss(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
    }
}

void* dealwithbyrow_SSE(void* ID)//处理高斯消去的每一个square线程函数
{
    int* threadid = (int*)ID;
    __m128 t1, t2, t3;//四位单精度构成的向量
    for (int k = 0; k < n; k++)
    {
        int begin = k + 1 + *threadid * ((n - k - 1) / thread_count);
        int end = begin + (n - k - 1) / thread_count;
        if (end > n)
            end = n;
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int begincol = k + 1 + preprocessnumber;
        for (int i = begin; i < end; i++)
        {
            for (int j = k + 1; j < preprocessnumber; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        for (int i = begin; i < end; i++)
        {
            float head1[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm_loadu_ps(head1);
            for (int j = begincol; j < n; j += 4)
            {
                t1 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_loadu_ps(A[i] + j);
                t1 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t1);
                _mm_store_ss(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
        sem_post(&sem_parent);//唤醒主线程，信号量
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}
void Gauss_pthread_row_SSE()
{
    pthread_t threadID[thread_count];//存线程号
    __m128 t1, t2, t3;
    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int begin = k + 1 + preprocessnumber;
        float head[4] = { A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm_loadu_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = begin; j < n; j++)
        {
            //A[k][j]=A[k][j]/A[k][k];
            t1 = _mm_loadu_ps(A[k] + j);
            t1 = _mm_div_ps(t1, t2);
            _mm_store_ss(A[k] + j, t1);
        }
        A[k][k] = 1;
        if (k == 0)//若未创建线程则创建线程
        {
            for (int i = 0; i < thread_count; i++)
            {
                pthread_create(&threadID[i], NULL, dealwithbyrow_SSE, (void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for (int i = 0; i < thread_count; i++)//
        {
            sem_wait(&sem_parent);
        }
    }
    pthread_barrier_wait(&childbarrier_row);
    for (int i = 0; i < thread_count; i++)
    {
        pthread_join(threadID[i], NULL);
    }
    return;
}
void* dealwithbycol_SSE(void* ID)
{
    int* threadid = (int*)ID;
    __m128 t1, t2, t3;
    for (int k = 0; k < n; k++)
    {
        int begin = k + 1 + *threadid * ((n - k - 1) / thread_count);
        int end = begin + (n - k - 1) / thread_count;
        if (end > n)
            end = n;
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int beginrow = k + 1 + preprocessnumber;
        float head[4] = { A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm_loadu_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = beginrow; j < end; j += 4)
        {
            t1 = _mm_loadu_ps(A[k] + j);
            t1 = _mm_div_ps(t1, t2);
            _mm_store_ss(A[k] + j, t1);
        }
        t1 = _mm_setzero_ps();//清零
        t2 = _mm_setzero_ps();

        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        for (int i = k + 1; i < n; i++)
        {
            float head1[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm_loadu_ps(head1);
            for (int j = beginrow; j < end; j += 4)
            {
                t1 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_loadu_ps(A[i] + j);
                t1 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t1);
                _mm_store_ss(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);

    }
    pthread_exit(NULL);
}
void Gauss_pthread_col_SSE()
{
    pthread_t threadID[thread_count];//存线程号
    for (int k = 0; k < n; k++)
    {
        if (k == 0)//若未创建线程则创建线程
        {
            for (int i = 0; i < thread_count; i++)
            {
                pthread_create(&threadID[i], NULL, dealwithbycol_SSE, (void*)&id[i]);
            }
        }
        for (int i = 0; i < thread_count; i++)//
        {
            sem_wait(&sem_parent);
        }
        //所有子线程执行完毕
        pthread_barrier_wait(&childbarrier_col);
        A[k][k] = 1;
        for (int i = k + 1; i < n; i++)
            A[i][k] = 0;


    }
    for (int i = 0; i < thread_count; i++)
    {
        pthread_join(threadID[i], NULL);
    }
    return;
}

int main()
{
    pthread_barrier_init(&childbarrier_row, NULL, thread_count + 1);
    pthread_barrier_init(&childbarrier_col, NULL, thread_count + 1);
    sem_init(&sem_parent, 0, 0);//信号量初始化
    for (int i = 0; i < thread_count; i++)
        id[i] = i;


    init();

    omp_gauss();
    cout << "omp" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init();
    Gauss_pthread_col();
    cout << "pthread_col" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init();
    Gauss_pthread_row();
    cout << "pthread_row" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init();
    Gauss_pthread_row_SSE();
    cout << "pthread_SSE_row" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init();
    Gauss_pthread_col_SSE();
    cout << "pthread_SSE_col" << (tail - head) * 1000.0 / freq << "ms" << endl;


    init();
    omp_SSE_row_gauss();
    cout << "omp_SSE" << (tail - head) * 1000.0 / freq << "ms" << endl;




}
