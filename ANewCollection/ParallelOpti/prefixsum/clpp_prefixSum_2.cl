__kernel void prefixSum(    __global uint* temp,
                            __global uint* iteration){ 
    
    int i = get_global_id(0);

/*https://github.com/michead/BroadPhaseCollisionDetection/tree/master/ParallelComputedCollisionDetection
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
    uint temp2;
    uint temp3;
    uint iter = *iteration;
    //barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        //temp2 = temp[i - (int)pow(2, (float)(iter) - 1)];
        atom_xchg(&temp2, temp[i - (int)pow(2, (float)(iter) - 1)]);
    //barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        //temp3 = temp[i];
        atom_xchg(&temp3, temp[i]);
    //barrier(CLK_GLOBAL_MEM_FENCE);
    if(i >= ((int)pow(2, (float)iter)))
        //temp[i] = temp2 + temp3;
        atomic_xchg(&temp[i], temp2 + temp3);
}