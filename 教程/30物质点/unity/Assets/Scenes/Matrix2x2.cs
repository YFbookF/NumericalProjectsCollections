using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Matrix2x2
{
    public float v00;
    public float v01;
    public float v10;
    public float v11;

    public Matrix2x2()
    {
        this.v00 = 0;
        this.v01 = 0;
        this.v10 = 0;
        this.v11 = 0;
    }

    public Matrix2x2(float v00, float v01, float v10, float v11)
    {
        this.v00 = v00;
        this.v01 = v01;
        this.v10 = v10;
        this.v11 = v11;
    }

    public static Matrix2x2 identity()
    {
        Matrix2x2 res = new Matrix2x2(1.0f,0.0f,0.0f,1.0f);
        return res;
    }

    public static Matrix2x2 Multiple(Matrix2x2 A, Matrix2x2 B)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00 * B.v00 + A.v01 * B.v10;
        res.v01 = A.v00 * B.v01 + A.v01 * B.v11;
        res.v10 = A.v10 * B.v00 + A.v11 * B.v10;
        res.v11 = A.v10 * B.v01 + A.v11 * B.v11;
        return res;
    }

    public static Matrix2x2 Transpose(Matrix2x2 A)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00;
        res.v01 = A.v10;
        res.v10 = A.v01;
        res.v11 = A.v11;
        return res;
    }

    
    public static Matrix2x2 operator * (Matrix2x2 A, float scaler)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00 * scaler;
        res.v01 = A.v01 * scaler;
        res.v10 = A.v10 * scaler;
        res.v11 = A.v11 * scaler;
        return res;
    }

    public static Matrix2x2 operator +(Matrix2x2 A, Matrix2x2 B)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00 + B.v00;
        res.v01 = A.v01 + B.v01;
        res.v10 = A.v10 + B.v10;
        res.v11 = A.v11 + B.v11;
        return res;
    }

    public static Matrix2x2 operator -(Matrix2x2 A, Matrix2x2 B)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00 - B.v00;
        res.v01 = A.v01 - B.v01;
        res.v10 = A.v10 - B.v10;
        res.v11 = A.v11 - B.v11;
        return res;
    }
    public static Matrix2x2 Dot(Matrix2x2 A, Matrix2x2 B)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = A.v00 * B.v00;
        res.v01 = A.v01 * B.v01;
        res.v10 = A.v10 * B.v10;
        res.v11 = A.v11 * B.v11;
        return res;
    }

    //https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
    /*
     * A = |2  2| = |1 0||2.828 0.000||0.707  0.707|
     *     |-1 1|   |0 1||0.000 1.141||-0.707 0.707|
     */
    public static void SVD(Matrix2x2 A,ref Matrix2x2 U,ref Matrix2x2 sig,ref Matrix2x2 V)
    {
        float E = (A.v00 + A.v11) / 2, F = (A.v00 - A.v11) / 2, G = (A.v10 + A.v01) / 2, H = (A.v10 - A.v01) / 2;
        float Q = Mathf.Sqrt(E * E + H * H), R = Mathf.Sqrt(F * F + G * G);
        sig.v00 = Q + R;
        sig.v11 = Q - R;
        float a1 = Mathf.Atan2(G, F), a2 = Mathf.Atan2(H, E);
        float theta = (a2 - a1) / 2, phi = (a2 + a1) / 2;
        U.v00 = Mathf.Cos(phi);
        U.v01 = -Mathf.Sin(phi);
        U.v10 = Mathf.Sin(phi);
        U.v11 = Mathf.Cos(phi);

        V.v00 = Mathf.Cos(theta);
        V.v01 = -Mathf.Sin(theta);
        V.v10 = Mathf.Sin(theta);
        V.v11 = Mathf.Cos(theta);
    }
    public static void Log(Matrix2x2 A,string str)
    {
        Debug.Log(str + "| " + A.v00 + "," + A.v01 + " |");
        Debug.Log("    | " + A.v10 + "," + A.v11 + " |");
    }

}
/*
 
        float y1 = A.v10 + A.v01, x1 = A.v00 + A.v11;
        float y2 = A.v01 - A.v10, x2 = A.v00 - A.v11;

        float theta0 = Mathf.Atan2(y1, x1) / 2;
        float thetad = Mathf.Atan2(y2, x2) / 2;

        float s0 = Mathf.Sqrt(y1 * y1 + x1 * x1) / 2;
        float s1 = Mathf.Sqrt(y2 * y2 + x2 * x2) / 2;

        sig.v00 = s0 + s1;
        sig.v11 = s0 - s1;

        U.v00 = Mathf.Cos(theta0 - thetad);
        U.v01 = -Mathf.Sin(theta0 - thetad);
        U.v10 = Mathf.Sin(theta0 - thetad);
        U.v11 = Mathf.Cos(theta0 - thetad);

        V.v00 = Mathf.Cos(theta0 + thetad);
        V.v01 = -Mathf.Sin(theta0 + thetad);
        V.v10 = Mathf.Sin(theta0 + thetad);
        V.v11 = Mathf.Cos(theta0 + thetad);
 
 */