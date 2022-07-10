#include <iostream>
#include <windows.h>
#include <pthread.h>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <semaphore.h>
#include <mpi.h>
#include <stdint.h>

#if def_OPENMP
#include<omp.h>
#endif

using namespace std;

#define n 20
#define thread_count 4

float A[n][n];
int id[thread_count];
long long head, tail, freq;
sem_t sem_parent;//���߳�
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;



MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);  //进程0广播
MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (rank)    data = array(N / (size - 1), D);  //其他进程分配存储数据集的空间
all_in_cluster = (int*)malloc(N / (size - 1) * size * sizeof(int));  //用于进程0
local_in_cluster = (int*)malloc(N / (size - 1) * sizeof(int));  //用于每个进程
in_cluster = (int*)malloc(N * sizeof(int));  //用于进程0
sum_diff = (float*)malloc(K * sizeof(float));  //进程中每个聚类的数据点与其中心点的距离之和
global_sum_diff = (float*)malloc(K * sizeof(float));
for (i = 0; i < K; i++)    sum_diff[i] = 0.0;  //初始化

if (!rank) {  //进程0向其他进程分配数据集
    for (i = 0; i < N; i += (N / (size - 1)))
        for (j = 0; j < (N / (size - 1)); j++)
            MPI_Send(data[i + j], D, MPI_FLOAT, (i + j) / (N / (size - 1)) + 1, 99, MPI_COMM_WORLD);
    printf("Data sets:\n");
    for (i = 0; i < N; i++)
        for (j = 0; j < D; j++) {
            printf("%-8.2f", data[i][j]);
            if ((j + 1) % D == 0)    putchar('\n');
        }
    printf("-----------------------------\n");
}
else {  //其他进程接收进程0数据
    for (i = 0; i < (N / (size - 1)); i++)
        MPI_Recv(data[i], D, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
}
MPI_Barrier(MPI_COMM_WORLD);  //同步一下
cluster_center = array(K, D);  //中心点
if (!rank) {  //进程0产生随机中心点
    srand((unsigned int)(time(NULL)));  //随机初始化k个中心点
    for (i = 0; i < K; i++)
        for (j = 0; j < D; j++)
            cluster_center[i][j] = data[(int)((double)N * rand() / (RAND_MAX + 1.0))][j];
}
for (i = 0; i < K; i++)    MPI_Bcast(cluster_center[i], D, MPI_FLOAT, 0, MPI_COMM_WORLD);  //进程0向其他进程广播中心点
if (rank) {
    cluster(N / (size - 1), K, D, data, cluster_center, local_in_cluster);  //其他进程进行聚类
    getDifference(K, N / (size - 1), D, local_in_cluster, data, cluster_center, sum_diff);
    for (i = 0; i < N / (size - 1); i++)
        printf("data[%d] in cluster-%d\n", (rank - 1) * (N / (size - 1)) + i, local_in_cluster[i] + 1);
}
MPI_Gather(local_in_cluster, N / (size - 1), MPI_INT, all_in_cluster, N / (size - 1), MPI_INT, 0, MPI_COMM_WORLD);  //全收集于进程0
MPI_Reduce(sum_diff, global_sum_diff, K, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  //归约至进程0,进程中每个聚类的数据点与其中心点的距离之和
if (!rank) {
    for (i = N / (size - 1); i < N + N / (size - 1); i++)
        in_cluster[i - N / (size - 1)] = all_in_cluster[i];  //处理收集的标记数组
    temp1 = 0.0;
    for (i = 0; i < K; i++) temp1 += global_sum_diff[i];
    printf("The difference between data and center is: %.2f\n\n", temp1);
    count++;
}
MPI_Bcast(&temp1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

do {   //比较前后两次迭代，若不相等继续迭代
    temp1 = temp2;
    if (!rank)    getCenter(K, D, N, in_cluster, data, cluster_center);  //更新中心点
    for (i = 0; i < K; i++)    MPI_Bcast(cluster_center[i], D, MPI_FLOAT, 0, MPI_COMM_WORLD);  //广播中心点
    if (rank) {
        cluster(N / (size - 1), K, D, data, cluster_center, local_in_cluster);  //其他进程进行聚类
        for (i = 0; i < K; i++)    sum_diff[i] = 0.0;
        getDifference(K, N / (size - 1), D, local_in_cluster, data, cluster_center, sum_diff);
        for (i = 0; i < N / (size - 1); i++)
            printf("data[%d] in cluster-%d\n", (rank - 1) * (N / (size - 1)) + i, local_in_cluster[i] + 1);
    }
    MPI_Gather(local_in_cluster, N / (size - 1), MPI_INT, all_in_cluster, N / (size - 1), MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank)
        for (i = 0; i < K; i++)    global_sum_diff[i] = 0.0;
    MPI_Reduce(sum_diff, global_sum_diff, K, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
        for (i = N / (size - 1); i < N + N / (size - 1); i++)
            in_cluster[i - N / (size - 1)] = all_in_cluster[i];
        temp2 = 0.0;
        for (i = 0; i < K; i++) temp2 += global_sum_diff[i];
        printf("The difference between data and center is: %.2f\n\n", temp2);
        count++;
    }
    MPI_Bcast(&temp2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
} while (fabs(temp2 - temp1) != 0.0);
if (!rank)    printf("The total number of cluster is: %d\n\n", count);
MPI_Finalize();
}