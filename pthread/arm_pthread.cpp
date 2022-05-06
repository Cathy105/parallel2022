#include <iostream>
#include<arm_neon.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include<unistd.h>
#include <pthread.h>
#include <semaphore.h>
#define n 2048
#define thread_count 4
using namespace std;

float A[n][n];
int id[thread_count];
sem_t sem_parent;//主线程
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;
void init()
{
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            A[i][j]=i+j;
}
void SIMD_Neon_gausseliminate(float A[n][n])//对齐
{
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;//要分别存储入内存中，不能一次将四个单精度浮点数存入内存
    for(int k=0;k<n;k++)
    {
        int preprocessnumber=(n-k-1)%4;//预处理的数量,能被四整除
        int begin=k+1+preprocessnumber;
        float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=vld1q_f32(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=begin;j<n;j+=4)
        {
            t1=vld1q_f32(A[k]+j);
            t1=vdivq_f32(t1,t2);
            s1 = vget_low_f32(t1);
            s2 = vget_high_f32(t1);
            vst1_lane_f32(A[k]+j,s2,0);
            vst1_lane_f32(A[k]+j+2,s1,0);
        }
        A[k][k]=1;
        t1=vdupq_n_f32(0.0);
        t2=vdupq_n_f32(0.0);

        //先去头，为了四个四个的处理
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
            t3=vld1q_f32(head1);
            for(int j=begin;j<n;j+=4)
            {
                t1=vld1q_f32(A[k]+j);
                t2=vld1q_f32(A[i]+j);
                t1=vmulq_f32(t1,t3);
                t2=vsubq_f32(t2,t3);
                s1 = vget_low_f32(t2);
                s2 = vget_high_f32(t2);
                vst1_lane_f32(A[i]+j,s2,0);
                vst1_lane_f32(A[i]+j+2,s1,0);
            }
            A[i][k]=0;
        }
    }
}

void * dealwithbyrow_Neon(void * ID)//处理高斯消去的每一个square线程函数
{
    int* threadid= (int*)ID;
    //__m128 t1,t2,t3;//四位单精度构成的向量
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;//要分别存储入内存中，不能一次将四个单精度浮点数存入内存
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%4;//预处理的数量,能被四整除
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
            //t3=_mm_loadu_ps(head1);
            t3=vld1q_f32(head1);
            for(int j=begincol;j<n;j+=4)
            {
                //t1=_mm_loadu_ps(A[k]+j);
                //t2=_mm_loadu_ps(A[i]+j);
                //t1=_mm_mul_ps(t1,t3);
                //t2=_mm_sub_ps(t2,t1);
                //_mm_store_ss(A[i]+j,t2);
                t1=vld1q_f32(A[k]+j);
                t2=vld1q_f32(A[i]+j);
                t1=vmulq_f32(t1,t3);
                t2=vsubq_f32(t2,t3);
                s1 = vget_low_f32(t2);
                s2 = vget_high_f32(t2);
                vst1_lane_f32(A[i]+j,s2,0);
                vst1_lane_f32(A[i]+j+2,s1,0);
            }
            A[i][k]=0;
        }
        pthread_barrier_wait(&childbarrier_row);
        sem_post(&sem_parent);//唤醒主线程，信号量
    }
    pthread_exit(NULL);
}
void * dealwithbycol_Neon(void * ID)
{
    int * threadid= (int*)ID;
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;//要分别存储入内存中，不能一次将四个单精度浮点数存入内存
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%4;//预处理的数量,能被四整除
        int beginrow=k+1+preprocessnumber;
        float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=vld1q_f32(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=beginrow;j<end;j+=4)
        {
            t1=vld1q_f32(A[k]+j);
            t1=vdivq_f32(t1,t2);
            s1 = vget_low_f32(t1);
            s2 = vget_high_f32(t1);
            vst1_lane_f32(A[k]+j,s2,0);
            vst1_lane_f32(A[k]+j+2,s1,0);
        }
        t1=vdupq_n_f32(0.0);
        t2=vdupq_n_f32(0.0);
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
            t3=vld1q_f32(head1);
            for(int j=beginrow;j<end;j+=4)
            {
                t1=vld1q_f32(A[k]+j);
                t2=vld1q_f32(A[i]+j);
                t1=vmulq_f32(t1,t3);
                t2=vsubq_f32(t2,t3);
                s1 = vget_low_f32(t2);
                s2 = vget_high_f32(t2);
                vst1_lane_f32(A[i]+j,s2,0);
                vst1_lane_f32(A[i]+j+2,s1,0);
            }
            A[i][k]=0;
        }

        pthread_barrier_wait(&childbarrier_col);
        sem_post(&sem_parent);
    }
    pthread_exit(NULL);
}

void Gausseliminate_pthread_col_Neon()
{
    pthread_t threadID[thread_count];//存线程号
    for(int k=0;k<n;k++)
    {
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,dealwithbycol_Neon,(void*)&id[i]);
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

    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}
void Gausseliminate_pthread_row_Neon()
{
    pthread_t threadID[thread_count];//存线程号
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;//要分别存储入内存中，不能一次将四个单精度浮点数存入内存
    for(int k=0;k<n;k++)
    {
        int preprocessnumber=(n-k-1)%4;//预处理的数量,能被四整除
        int begin=k+1+preprocessnumber;
        float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t2=vld1q_f32(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        for(int j=begin;j<n;j++)
        {
            t1=vld1q_f32(A[k]+j);
            t1=vdivq_f32(t1,t2);
            s1 = vget_low_f32(t1);
            s2 = vget_high_f32(t1);
            vst1_lane_f32(A[k]+j,s2,0);
            vst1_lane_f32(A[k]+j+2,s1,0);
        }
        A[k][k]=1;
        if(k==0)//若未创建线程则创建线程
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,dealwithbyrow_Neon,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)//
        {
            sem_wait(&sem_parent);
        }
    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}
int main()
{
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
    pthread_barrier_init(&childbarrier_row, NULL,thread_count);
    pthread_barrier_init(&childbarrier_col,NULL, thread_count);
    sem_init(&sem_parent, 0, 0);//信号量初始化

    init();
    gettimeofday(&start,NULL);
    Gausseliminate_pthread_row_Neon();
    gettimeofday(&end,NULL);
    diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    printf("thedifference is %ld\n",diff);

    init();
    gettimeofday(&start,NULL);
    Gausseliminate_pthread_col_Neon();
    gettimeofday(&end,NULL);
    diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    printf("thedifference is %ld\n",diff);

    sem_destroy(&sem_parent);
    pthread_barrier_destroy(&childbarrier_col);
    pthread_barrier_destroy(&childbarrier_row);
}
