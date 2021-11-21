__kernel void prefixSum(    __global uint* temp,
                            __global uint* iteration){ 
    https://github.com/michead/BroadPhaseCollisionDetection/tree/master/ParallelComputedCollisionDetection
    int i = get_global_id(0);

/*
    uint temp2;
    uint temp3;
    barrier(CLK_GLOBAL_MEM_FENCE);
    uint iter = *iteration;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        temp2 = temp[i - (int)pow(2, (float)(iter))];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        temp3 = temp[i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        temp[i] = temp2 + temp3;
*/
    uint iter = *iteration;
    uint temp2 = 0;
    uint temp3 = 0;
    if(i >= ((int)pow(2, (float)iter)))
        temp2 = temp[i - (int)pow(2, (float)(iter))];
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        temp3 = temp[i];
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        atom_xchg(&temp[i], temp2 + temp3);
}