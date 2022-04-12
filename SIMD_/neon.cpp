#include <iostream>
#include<arm_neon.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include<unistd.h>
#define n 2048
using namespace std;
float A[n][n];

void init(float A[n][n])
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
int main()
{
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
    init(A);
    gettimeofday(&start,NULL);
    SIMD_Neon_gausseliminate(A);
    gettimeofday(&end,NULL);
    diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    printf("thedifference is %ld\n",diff);
}
