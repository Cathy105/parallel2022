int seq_kmeans(float** objects,      /* in: [numObjs][numCoords] */
    int     numCoords,    /* no. features */
    int     numObjs,      /* no. objects */
    int     numClusters,  /* no. clusters */
    float   threshold,    /* % objects change membership */
    int* membership,   /* out: [numObjs] */
    float** clusters)     /* out: [numClusters][numCoords] */

{
    int      i, j, index, loop = 0;
    int* newClusterSize; /* [numClusters]: no. objects assigned in each
                         new cluster */
    float    delta;          /* % of objects change their clusters */
    float** newClusters;    /* [numClusters][numCoords] */

    /* ≥ı ºªØ membership[] */
    for (i = 0; i < numObjs; i++) membership[i] = -1;

    newClusterSize = (int*)calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters = (float**)malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i = 1; i < numClusters; i++)
        newClusters[i] = newClusters[i - 1] + numCoords;

    do {
        delta = 0.0;
        for (i = 0; i < numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                clusters);


            if (membership[i] != index) delta += 1.0;

            membership[i] = index;

            newClusterSize[index]++;
            for (j = 0; j < numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }

        for (i = 0; i < numClusters; i++) {
            for (j = 0; j < numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;

    
int omp_kmeans(int     is_perform_atomic, /* in: */
        float** objects,           /* in: [numObjs][numCoords] */
        int     numCoords,         /* no. coordinates */
        int     numObjs,           /* no. objects */
        int     numClusters,       /* no. clusters */
        float   threshold,         /* % objects change membership */
        int* membership,        /* out: [numObjs] */
        float** clusters)          /* out: [numClusters][numCoords] */
    {
        int      i, j, k, index, loop = 0;
        int* newClusterSize; /* [numClusters]: no. objects assigned in each
                             new cluster */
        float    delta;          /* % of objects change their clusters */
        float** newClusters;    /* [numClusters][numCoords] */
        double   timing;

        int      nthreads;             /* no. threads */
        int** local_newClusterSize; /* [nthreads][numClusters] */
        float*** local_newClusters;    /* [nthreads][numClusters][numCoords] */

        nthreads = omp_get_max_threads();

        /* initialize membership[] */
        for (i = 0; i < numObjs; i++) membership[i] = -1;

        /* need to initialize newClusterSize and newClusters[0] to all 0 */
        newClusterSize = (int*)calloc(numClusters, sizeof(int));
        assert(newClusterSize != NULL);

        newClusters = (float**)malloc(numClusters * sizeof(float*));
        assert(newClusters != NULL);
        newClusters[0] = (float*)calloc(numClusters * numCoords, sizeof(float));
        assert(newClusters[0] != NULL);
        for (i = 1; i < numClusters; i++)
            newClusters[i] = newClusters[i - 1] + numCoords;

        if (!is_perform_atomic) {

            local_newClusterSize = (int**)malloc(nthreads * sizeof(int*));
            assert(local_newClusterSize != NULL);
            local_newClusterSize[0] = (int*)calloc(nthreads * numClusters,
                sizeof(int));
            assert(local_newClusterSize[0] != NULL);
            for (i = 1; i < nthreads; i++)
                local_newClusterSize[i] = local_newClusterSize[i - 1] + numClusters;

            /* local_newClusters is a 3D array */
            local_newClusters = (float***)malloc(nthreads * sizeof(float**));
            assert(local_newClusters != NULL);
            local_newClusters[0] = (float**)malloc(nthreads * numClusters *
                sizeof(float*));
            assert(local_newClusters[0] != NULL);
            for (i = 1; i < nthreads; i++)
                local_newClusters[i] = local_newClusters[i - 1] + numClusters;
            for (i = 0; i < nthreads; i++) {
                for (j = 0; j < numClusters; j++) {
                    local_newClusters[i][j] = (float*)calloc(numCoords,
                        sizeof(float));
                    assert(local_newClusters[i][j] != NULL);
                }
            }
        }

        if (_debug) timing = omp_get_wtime();
        do {
            delta = 0.0;

            if (is_perform_atomic) {
#pragma omp parallel for \
         private(i,j,index) \
         firstprivate(numObjs,numClusters,numCoords) \
         shared(objects,clusters,membership,newClusters,newClusterSize) \
         schedule(static) \
         reduction(+:delta)
                for (i = 0; i < numObjs; i++) {
                    index = find_nearest_cluster(numClusters, numCoords, objects[i],
                        clusters);

                    if (membership[i] != index) delta += 1.0;

                    membership[i] = index;

#pragma omp atomic
                    newClusterSize[index]++;
                    for (j = 0; j < numCoords; j++)
#pragma omp atomic
                        newClusters[index][j] += objects[i][j];
                }
            }
            else {
#pragma omp parallel \
         shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();
#pragma omp for \
                 private(i,j,index) \
                 firstprivate(numObjs,numClusters,numCoords) \
                 schedule(static) \
                 reduction(+:delta)
                for (i = 0; i < numObjs; i++) {
                    index = find_nearest_cluster(numClusters, numCoords,
                        objects[i], clusters);

                    if (membership[i] != index) delta += 1.0;

                    membership[i] = index;

                    local_newClusterSize[tid][index]++;
                    for (j = 0; j < numCoords; j++)
                        local_newClusters[tid][index][j] += objects[i][j];
                }
            }
            for (i = 0; i < numClusters; i++) {
                for (j = 0; j < nthreads; j++) {
                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0.0;
                    for (k = 0; k < numCoords; k++) {
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }
            }

            for (i = 0; i < numClusters; i++) {
                for (j = 0; j < numCoords; j++) {
                    if (newClusterSize[i] > 1)
                        clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                    newClusters[i][j] = 0.0;   /* set back to 0 */
                }
                newClusterSize[i] = 0;   /* set back to 0 */
            }

            delta /= numObjs;
        } while (delta > threshold && loop++ < 500);

        if (_debug) {
            timing = omp_get_wtime() - timing;
            printf("nloops = %2d (T = %7.4f)", loop, timing);
        }

        if (!is_perform_atomic) {
            free(local_newClusterSize[0]);
            free(local_newClusterSize);

            for (i = 0; i < nthreads; i++)
                for (j = 0; j < numClusters; j++)
                    free(local_newClusters[i][j]);
            free(local_newClusters[0]);
            free(local_newClusters);
        }
        free(newClusters[0]);
        free(newClusters);
        free(newClusterSize);

        return 1;
    }