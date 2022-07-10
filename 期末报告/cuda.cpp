for (unsigned int iterations = 0; iterations < 30; iterations++) {
    std::cout << "Iteration " << iterations << "...\n";//开始迭代

    thrust::sequence(scatterResults.begin(), scatterResults.end(), 0, numClusters);//计算结果

    //每一次迭代计算距离
    for (int i = 0; i < numClusters; i++) {

        thrust::gather(dTemporary.begin(), dTemporary.end(), clusterMask.begin(), dClusters.begin() + i * dNumAttributes[0]); //聚类中心坐标
        thrust::transform(dFeature.begin(), dFeature.end(), dTemporary.begin(), dTemporary.begin(), sqDiff()); //计算平方

        thrust::experimental::inclusive_segmented_scan(dTemporary.begin(), dTemporary.end(), segScanMask.begin(), dTemporary.begin());

        thrust::gather(dTemporary2.begin(), dTemporary2.end(), gatherDistMask.begin(), dTemporary.begin());
        thrust::scatter(dTemporary2.begin(), dTemporary2.end(), scatterResults.begin(), dResults.begin()); //Scatter result to result table
        thrust::transform(scatterResults.begin(), scatterResults.end(), thrust::make_constant_iterator(1), scatterResults.begin(), thrust::plus<int>());

        //计算下一个聚类结果


    }//next cluster

        //寻找每个点最好的簇
    thrust::experimental::inclusive_segmented_scan
    (thrust::make_zip_iterator(make_tuple(dResults.begin(), clusterPointsMask.begin())),
        thrust::make_zip_iterator(make_tuple(dResults.end(), clusterPointsMask.end())),
        pointMinMask.begin(),
        thrust::make_zip_iterator(make_tuple(dResults.begin(), clusterPointsMask.begin())),
        minPair());
    thrust::device_vector<int> mapObjects(dNumObjects[0]);
    thrust::sequence(mapObjects.begin(), mapObjects.end(), numClusters - 1, numClusters);

    thrust::device_vector<float> pointClusters(dNumObjects[0]);
    thrust::gather(pointClusters.begin(), pointClusters.end(), mapObjects.begin(), clusterPointsMask.begin());
    thrust::sequence(mapObjects.begin(), mapObjects.end(), 0, (int)dNumAttributes[0]);
    thrust::sorting::radix_sort_by_key(pointClusters.begin(), pointClusters.end(), mapObjects.begin());

    thrust::fill(objectCount.begin(), objectCount.end(), 1);
    thrust::experimental::inclusive_segmented_scan(objectCount.begin(), objectCount.end(), pointClusters.begin(), objectCount.begin());

    thrust::device_vector<int> gatherSumsVector(dNumObjects[0]);

    thrust::adjacent_difference(pointClusters.begin(), pointClusters.end(), gatherSumsVector.begin());
    thrust::transform(gatherSumsVector.begin(), gatherSumsVector.end(), gatherSumsVector.begin(), unitify<int>());
    thrust::device_vector<int> activeClusters(1);
    activeClusters[0] = thrust::reduce(gatherSumsVector.begin(), gatherSumsVector.end()) + 1;
    thrust::transform(gatherSumsVector.begin(), gatherSumsVector.end(), thrust::make_counting_iterator(1), gatherSumsVector.begin(), thrust::multiplies<int>());
    thrust::remove(gatherSumsVector.begin(), gatherSumsVector.end(), 0);

    thrust::transform(gatherSumsVector.begin(), gatherSumsVector.end(), thrust::make_constant_iterator(2), gatherSumsVector.begin(), thrust::minus<int>());
    gatherSumsVector[activeClusters[0] - 1] = dNumObjects[0] - 1;

    thrust::device_vector<int> scatterClustersVector(activeClusters[0]);

    thrust::gather(scatterClustersVector.begin(), scatterClustersVector.end(), gatherSumsVector.begin(), pointClusters.begin());

    thrust::device_vector<int> clusterObjectCount(activeClusters[0]);
    thrust::gather(clusterObjectCount.begin(), clusterObjectCount.end(), gatherSumsVector.begin(), objectCount.begin());


    thrust::transform(scatterClustersVector.begin(), scatterClustersVector.end(), thrust::make_constant_iterator(numAttributes), scatterClustersVector.begin(), thrust::multiplies<int>());

    std::cout << "Updating clusters...";
    for (unsigned int attrNo = 0; attrNo < numAttributes; attrNo++) {
        thrust::gather(temporaryStorage.begin(), temporaryStorage.end(), mapObjects.begin(), dFeature.begin());
        thrust::experimental::inclusive_segmented_scan(temporaryStorage.begin(), temporaryStorage.end(), pointClusters.begin(), temporaryStorage.begin());
        thrust::gather(temporaryStorage2.begin(), temporaryStorage2.begin() + activeClusters[0], gatherSumsVector.begin(), temporaryStorage.begin());
        thrust::scatter(temporaryStorage2.begin(), temporaryStorage2.begin() + activeClusters[0], scatterClustersVector.begin(), dClusters.begin()); //Change
        thrust::transform(mapObjects.begin(), mapObjects.end(), thrust::make_constant_iterator(1), mapObjects.begin(), thrust::plus<int>());
        thrust::transform(scatterClustersVector.begin(), scatterClustersVector.end(), thrust::make_constant_iterator(1), scatterClustersVector.begin(), thrust::plus<int>());
    }
    thrust::gather(divisionBy.begin(), divisionBy.end(), clusterCoordinate.begin(), clusterObjectCount.begin());
    thrust::transform(dClusters.begin(), dClusters.end(), divisionBy.begin(), dClusters.begin(), division());
}//下一次迭代