  #include <iostream>
  #include<ctime>
  #include <stdlib.h>
  #include <windows.h>
  using namespace std;

  int n=200000;
  int sum=0;
  int *a=new int[n];

  void init()
  {
      for(int i=0;i<n;i++)
      {
          a[i]=i;
      }
  }

  void normal()//ƽ���㷨
  {
      for(int i=0;i<n;i++)
      {
          sum+=a[i];
      }
  }

  void multiplelink()//��·��ʽ
  {
      int sum0=0,sum1=0;
      for(int i0=0,i1=n/2;i0<n/2&&i1<n;i0++,i1++)
      {
          sum0+=a[i0];
          sum1+=a[i1];
      }
      sum=sum0+sum1;
      return;
  }

  void recursion(int s)//�ݹ�
  {
      if(s==1)
      {
          return;
      }
      else
      {
          for(int i=0;i<s/2;i++)
          {
              a[i]+=a[s-i-1];
              s=s/2;
              recursion(s);
          }
      }
  }

  void doubleloop(int n)//˫��ѭ��
  {
      for(int i=n;i>=1;i=i/2)
      {
          for(int j=0;j<i/2;j++)
          {
              a[j]+=a[i-j-1];
          }
      }
      return;
  }

    int main()
  {
     long long head, tail , freq ;
      //��ͨ���
      init();
      QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
      QueryPerformanceCounter((LARGE_INTEGER *)&head);
      normal();
      QueryPerformanceCounter((LARGE_INTEGER *)&tail );
      cout <<"��ͨ��ͣ�"<<(tail-head) *1000.0/freq <<"ms " <<endl;
      //�Ż��㷨
      sum=0;
      QueryPerformanceCounter((LARGE_INTEGER *)&head);
      multiplelink();
      QueryPerformanceCounter((LARGE_INTEGER *)&tail );
      cout <<"��·��ʽ��"<<(tail-head) *1000.0/freq <<"ms " <<endl;
      sum=0;
      QueryPerformanceCounter((LARGE_INTEGER *)&head);
      recursion(n);
      QueryPerformanceCounter((LARGE_INTEGER *)&tail );
      cout <<"�ݹ飺"<<(tail-head) *1000.0/freq <<"ms " <<endl;
      sum=0;
      QueryPerformanceCounter((LARGE_INTEGER *)&head);
      doubleloop(n);
      QueryPerformanceCounter((LARGE_INTEGER *)&tail );
      cout <<"˫��ѭ����"<<(tail-head) *1000.0/freq <<"ms " <<endl;
      return 0;
  }
