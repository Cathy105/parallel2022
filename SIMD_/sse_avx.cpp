#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <malloc.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#define n 1024
using namespace std;
float A[n][n];
__attribute__((aligned(16)))float A1[n][n];
__attribute__((aligned(32)))float A2[n][n];
long long head, tail, freq;
void init(float A[n][n])
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = i + j;
}

void initA1()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A1[i][j] = i + j;
}
void initA2()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A2[i][j] = i + j;
}
void normal_gausseliminate(float A[n][n])
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
void SIMD_SSE_gausseliminate(float A[n][n])//对齐
{
    __m128 t1, t2, t3;//四位单精度构成的向量
    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int begin = k + 1 + preprocessnumber;
        __attribute__((aligned(16)))float head[4] = { A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm_load_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = begin; j < n; j += 4)
        {
            t1 = _mm_load_ps(A[k] + j);
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
        for (int i = k + 1; i < n; i++)
        {
            __attribute__((aligned(16)))float head1[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm_load_ps(head1);
            for (int j = begin; j < n; j += 4)
            {
                t1 = _mm_load_ps(A[k] + j);
                t2 = _mm_load_ps(A[i] + j);
                t1 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t1);
                _mm_store_ss(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
    }
}
void SIMD_notaligned_SSE_gausseliminate(float A[n][n])
{
    __m128 t1, t2, t3;//四位单精度构成的向量
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
void SIMD_notaligned_partialimprove_SSE_gausseliminate(float A[n][n])
{
    __m128 t1, t2, t3;//四位单精度构成的向量
    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 4;//预处理的数量,能被四整除
        int begin = k + 1 + preprocessnumber;
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
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
void SIMD_AVX_gausseliminate(float A[n][n])
{
    __m256 t1, t2, t3;//八位单精度构成的向量

    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 8;//预处理的数量,能被八整除
        int begin = k + 1 + preprocessnumber;
        __attribute__((aligned(32)))float head[8] = { A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm256_load_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = begin; j < n; j += 8)
        {
            t1 = _mm256_load_ps(A[k] + j);
            t1 = _mm256_div_ps(t1, t2);
            _mm256_store_ps(A[k] + j, t1);
        }
        A[k][k] = 0;
        t1 = _mm256_setzero_ps();//清零
        t2 = _mm256_setzero_ps();
        //先去头，为了四个四个的处理

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
            __attribute__((aligned(32))) float head1[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm256_load_ps(head1);
            for (int j = begin; j < n; j += 8)
            {
                t1 = _mm256_load_ps(A[k] + j);
                t2 = _mm256_load_ps(A[i] + j);
                t1 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t1);
                _mm256_store_ps(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
    }
}
void SIMD_AVX_notaligned_gausseliminate(float A[n][n])
{
    __m256 t1, t2, t3;//八位单精度构成的向量

    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 8;//预处理的数量,能被四整除
        int begin = k + 1 + preprocessnumber;
        float head[8] = { A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k] };
        t2 = _mm256_loadu_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = begin; j < n; j += 8)
        {
            t1 = _mm256_loadu_ps(A[k] + j);
            t1 = _mm256_div_ps(t1, t2);
            _mm256_storeu_ps(A[k] + j, t1);
        }
        A[k][k] = 0;
        t1 = _mm256_setzero_ps();//清零
        t2 = _mm256_setzero_ps();
        //先去头，为了四个四个的处理

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
            float head1[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            t3 = _mm256_loadu_ps(head1);
            for (int j = begin; j < n; j += 8)
            {
                t1 = _mm256_loadu_ps(A[k] + j);
                t2 = _mm256_loadu_ps(A[i] + j);
                t1 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t1);
                _mm256_storeu_ps(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
    }
}

int main()
{

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    init(A);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    normal_gausseliminate(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "normal" << (tail - head) * 1000.0 / freq << "ms" << endl;

    initA1();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SIMD_SSE_gausseliminate(A1);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE_aligned" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SIMD_notaligned_partialimprove_SSE_gausseliminate(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE_notaligned_partialimprove" << (tail - head) * 1000.0 / freq << "ms" << endl;


    init(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SIMD_notaligned_SSE_gausseliminate(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE_notaligned" << (tail - head) * 1000.0 / freq << "ms" << endl;

    initA2();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SIMD_AVX_gausseliminate(A2);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX_aligned" << (tail - head) * 1000.0 / freq << "ms" << endl;

    init(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SIMD_AVX_notaligned_gausseliminate(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX_notaligned" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
