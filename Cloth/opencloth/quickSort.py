import numpy as np

arr = np.array([6,3,5,8,2,1,5,3,8,6])
idx = np.array([0,1,2,3,4,5,6,7,8,9])

def quickSort(left,right):
    if left >= right:
        return 
    key = arr[left]
    keyidx = idx[left]
    i = left
    j = right
    while i < j:
        while i < j and arr[j] >= key:
            j -= 1
        if i < j:
            arr[i] = arr[j]
            idx[i] = idx[j]
            i += 1
        while i < j and arr[i] < key:
            i += 1
        if i < j:
            arr[j] = arr[i]
            idx[j] = idx[i]
            j -= 1
    arr[i] = key
    idx[i] = keyidx
    quickSort(left, i-1)
    quickSort(i+1, right)
    
quickSort(0,len(arr)-1)