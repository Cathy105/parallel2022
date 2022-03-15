#include <iostream>
#include<ctime>
#include <stdlib.h>
#include <windows.h>
using namespace std;

int n=20000;
int *a=new int[n];
int *sum=new int[n];
int **b=new int*[n];

void init()
{
    for(int i=0;i<n;i++)
    {
        sum[i] = 0;
        a[i]=i;
        b[i]=new int[n];
        for(int j=0;j<n;j++)
            b[i][j]=i+j;
    }
}
long long head, tail , freq ;
void normal()//平凡算法
{
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            sum[i] += b[j][i] * a[j];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout <<"平凡算法："<<(tail-head) *1000.0/freq <<"ms" <<endl;
}

void optimize()//优化算法
{
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j=0;j<n;j++)
        {
            for(int i=0;i<n;i++)
            {
                sum[i]+=b[j][i]*a[i];
            }
        }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout <<"优化算法："<<(tail-head) *1000.0/freq <<"ms"<<endl;
}
int main()
{
   init();
   normal();
   optimize();
   return 0;
}
