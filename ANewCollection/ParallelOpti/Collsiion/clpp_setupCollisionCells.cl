//#define ITERATIONS 10

__kernel void setupCollisionCells(  __global const ulong* in, //1024
                                    //__global ulong* out, //1024
                                    __global uint* temp, //1024
                                    __global uint* n, // 1
                                    __global uint* offset, // 1093
                                    __global uint* numOfOcc){ // 1093
									//https://github.com/michead/BroadPhaseCollisionDetection/tree/master/ParallelComputedCollisionDetection
    int i = get_global_id(0);
    
    //if((uint)in[i] == (uint)4294967295)
        //return;
    
    if((uint)in[i] != (uint)4294967295){
        atom_inc(&numOfOcc[(uint)in[i]]);
        atom_inc(n);
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    //count of occurrences of each cell ID
    if((i==0 || ((uint)in[i] != (uint)in[i-1])) && (in[i] & ((ulong)1 << 63)) && ((uint)in[i] != (uint)4294967295)){
        //offset[(uint)in[i]] = (uint)i;
        //offset[(uint)in[i]] |= ((uint)(1 << 16));
        temp[i] = 1;
        offset[i] = i;
        //atom_inc(n);
        
        //temp[i] = numOfOcc[(uint)in[i]]; //!!!!!!  why copy doesn't work???? -- barrier works?
        for(uint p = 1; p < numOfOcc[(uint)in[i]]; p++){
           temp[i + p] = 1;
           //barrier(CLK_GLOBAL_MEM_FENCE);
        }
        //atom_xchg(numOfCCells, *numOfCCells + numOfOcc[(int)in[i]]);
    }
}