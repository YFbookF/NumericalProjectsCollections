using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Matrix2x2 A = new Matrix2x2(2.0f, 2.0f, -1.0f, 1.0f);

        Matrix2x2 U = new Matrix2x2();
        Matrix2x2 sig = new Matrix2x2();
        Matrix2x2 V = new Matrix2x2();
        Matrix2x2.SVD(A, ref U, ref sig, ref V);
        Matrix2x2.Log(U,"A");
        Matrix2x2.Log(sig, "B");
        Matrix2x2.Log(V, "C");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
