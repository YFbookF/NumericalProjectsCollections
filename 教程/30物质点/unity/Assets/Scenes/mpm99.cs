using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.IO;

public class mpm99 : MonoBehaviour
{
    
    // taichi mpm99
    // https://github.com/taichi-dev/taichi/blob/master/examples/simulation/mpm99.py
    public GameObject spherePrefab0;
    public GameObject spherePrefab1;
    public GameObject spherePrefab2;
    private GameObject[] Particle;

    // 粒子参数
    private const int ParticleNum = 9000;
    private Vector2[] ParticlePos;
    private Vector2[] ParticleVel;
    
    private Matrix2x2[] ParticleC; // affine velocity field
    private Matrix2x2[] ParticleF; // deformation gradient

    private float[] ParticleJ; // plastic deformation
    private int[] ParticleMaterial; // material id

    // 网格参数
    private const int GridSize = 128; // 网格数量
    private const float dx = 1 / (float)GridSize; // 网格长度
    private const float dx_inv = (float)GridSize;
    private Vector2[] GridVel; // 网格速度
    private float[] GridMass; // 网格质量

    // 物理量
    private static readonly float E = 1000.0f; // 杨氏模量
    private static readonly float nu = 0.2f; // 粘度
    private float mu_0 = E / (2.0f * (1 + nu)); // Lame 常数
    private float lambda_0 = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu)); // Lame 常数
    private int bound = 3; // 网格边界宽度
    private const float gravity = 9.8f;//重力
    private const float p_rho = 1.0f; // 粒子密度
    private const float p_vol = (dx * 0.5f) * (dx * 0.5f); // 粒子体积
    private const float p_mass = p_rho * p_vol;// 粒子重量
    private const float dt = 0.0001f; // 步长
    private float solverDomainLength = 10; // 求解域长度


    private void initArray()
    {
        Particle = new GameObject[ParticleNum];
        ParticlePos = new Vector2[ParticleNum];
        ParticleVel = new Vector2[ParticleNum];
        ParticleC = new Matrix2x2[ParticleNum];
        ParticleF = new Matrix2x2[ParticleNum];
        ParticleJ = new float[ParticleNum];
        ParticleMaterial = new int[ParticleNum];
        GridVel = new Vector2[GridSize * GridSize];
        GridMass = new float[GridSize * GridSize];
    }

    private void initParticle()
    {



        for (int i = 0; i < ParticleNum; i++)
        {
            int group = i % 3;
            ParticleMaterial[i] = group;// 0:fluid 1:jelly 2:snow
            Debug.Log(group);
            ParticlePos[i].x = Random.Range(0.0f, 1.0f) * 0.2f + 0.10f * group + 0.3f;
            ParticlePos[i].y = Random.Range(0.0f, 1.0f) * 0.2f + 0.32f * group + 0.05f;
            if(group == 0)
            {
                Particle[i] = Instantiate(spherePrefab0, new Vector3(ParticlePos[i].x * solverDomainLength, ParticlePos[i].y * solverDomainLength, 0), Quaternion.identity);
            }
            else if(group == 1)
            {
                Particle[i] = Instantiate(spherePrefab1, new Vector3(ParticlePos[i].x * solverDomainLength, ParticlePos[i].y * solverDomainLength, 0), Quaternion.identity);
            }else
            {
                Particle[i] = Instantiate(spherePrefab2, new Vector3(ParticlePos[i].x * solverDomainLength, ParticlePos[i].y * solverDomainLength, 0), Quaternion.identity);
            }
            

            ParticleVel[i].x = 0;
            ParticleVel[i].y = 0;


            ParticleC[i] = new Matrix2x2();
            ParticleF[i] = Matrix2x2.identity();
            ParticleJ[i] = 1;
        }

        /*
        string[] myText = File.ReadAllLines("posx.txt");
        for (int i = 0; i < ParticleNum; i++)
        {
            string[] pos = myText[i].Split(' ');
            ParticlePos[i].x = float.Parse(pos[0]);
            ParticlePos[i].y = float.Parse(pos[1]);
        }*/
    }



    private void ParticleToGrid()
    {
        for (int j = 0; j < GridSize; j++)
        {
            for (int i = 0; i < GridSize; i++)
            {
                int idx = j * GridSize + i;
                GridVel[idx].x = GridVel[idx].y = 0;
                GridMass[idx] = 0;
            }
        }

        for (int p = 0; p < ParticleNum; p++)
        {
            float Xpx = ParticlePos[p].x / dx;
            float Xpy = ParticlePos[p].y / dx;
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };
            
            ParticleF[p] = Matrix2x2.Multiple(Matrix2x2.identity() + ParticleC[p] * dt,ParticleF[p]); // deformation gradient update

            float h = Mathf.Exp(10.0f * (1 - ParticleJ[p])); // Hardening coefficient: snow gets harder when compressed
            if (ParticleMaterial[p] == 1) // jelly, make it softer
                h = 0.3f;
            float mu = mu_0 * h;
            float la = lambda_0 * h;
            if (ParticleMaterial[p] == 0)
                mu = 0;

            Matrix2x2 U = new Matrix2x2();
            Matrix2x2 sig = new Matrix2x2();
            Matrix2x2 Vt = new Matrix2x2();
            Matrix2x2.SVD(ParticleF[p], ref U, ref sig, ref Vt);
            
            float J = 1.0f;

            float[] sigi = { sig.v00, sig.v11 };
            for(int i = 0;i < 2;i++)
            {
                float new_sig = sigi[i];
                if (ParticleMaterial[p] == 2)//Snow
                    new_sig = Mathf.Min(Mathf.Max(sigi[i], 1 - 0.025f), 1.0045f);//Plasticity
                ParticleJ[p] *= sigi[i] / new_sig;
                sigi[i] = new_sig;
                J *= new_sig;
            }
            sig.v00 = sigi[0];
            sig.v11 = sigi[1];
            if (ParticleMaterial[p] == 0)
                ParticleF[p] = Matrix2x2.identity() * Mathf.Sqrt(J);// Reset deformation gradient to avoid numerical instability
            else if (ParticleMaterial[p] == 2)
                ParticleF[p] = Matrix2x2.Multiple(Matrix2x2.Multiple(U, sig),Vt);//Reconstruct elastic deformation gradient after plasticity

            Matrix2x2 term = (ParticleF[p] - Matrix2x2.Multiple(U,Vt)) * 2.0f * mu;
            Matrix2x2 stress = Matrix2x2.Multiple(term, Matrix2x2.Transpose(ParticleF[p])) + Matrix2x2.identity() * (la * J * (J - 1));
            stress = stress * (-dt * p_vol * 4 * dx_inv * dx_inv);
            /*
                def debugst():
                    for i in range(n_particles):
                        print(" %.6f" % st[i][0,0]," %.6f" % st[i][0,1])
                        print(" %.6f" % st[i][1,0]," %.6f" % st[i][1,1])
                        print("hahaha")
             */
            Matrix2x2 affine = ParticleC[p] * p_mass + stress ;

            //Matrix2x2.Log(affine, " stress " + p);
            for (int j = 0;j < 3;j++)
            {
                for(int i = 0;i < 3;i++)
                {
                    if (basex + i < 0 || basex + i >= GridSize)
                        continue;
                    if (basey + j < 0 || basey + j >= GridSize)
                        continue;
                    int idx = (basey + j) * GridSize + basex + i;
                    float weight = wx[i] * wy[j];
                    float dposx = (i - fx) * dx;
                    float dposy = (j - fy) * dx;

                    float aff = affine.v00 * dposx + affine.v01 * dposy;
                    GridVel[idx].x += weight * (p_mass * ParticleVel[p].x + aff);

                    aff = affine.v10 * dposx + affine.v11 * dposy;
                    GridVel[idx].y += weight * (p_mass * ParticleVel[p].y + aff);

                    GridMass[idx] += weight * p_mass;
                }
            }



        }
    }


    private void Boundary()
    {
        for (int j = 0; j < GridSize; j++)
        {
            for (int i = 0; i < GridSize; i++)
            {
                int idx = j * GridSize + i;
                if (GridMass[idx] > 0)
                    GridVel[idx] /= GridMass[idx];
                GridVel[idx].y -= dt * 50.0f;
                if (i < bound && GridVel[idx].x < 0)
                    GridVel[idx].x = 0;
                if (i > GridSize - bound && GridVel[idx].x > 0)
                    GridVel[idx].x = 0;
                if (j < bound && GridVel[idx].y < 0)
                    GridVel[idx].y = 0;
                if (j > GridSize - bound && GridVel[idx].y > 0)
                    GridVel[idx].y = 0;

            }
        }
    }

    private void GridToParticle()
    {
        for (int p = 0; p < ParticleNum; p++)
        {
            float Xpx = ParticlePos[p].x / dx;
            float Xpy = ParticlePos[p].y / dx;
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };

            Vector2 new_vel = new Vector2(0.0f, 0.0f);
            float new_C00 = 0;
            float new_C01 = 0;
            float new_C10 = 0;
            float new_C11 = 0;

            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (basex + i < 0 || basex + i >= GridSize)
                        continue;
                    if (basey + j < 0 || basey + j >= GridSize)
                        continue;
                    int idx = (basey + j) * GridSize + basex + i;
                    float weight = wx[i] * wy[j];
                    float dposx = i - fx;
                    float dposy = j - fy;

                    new_vel += weight * GridVel[idx];

                    new_C00 += 4 * weight / dx * GridVel[idx].x * dposx;
                    new_C01 += 4 * weight / dx * GridVel[idx].x * dposy;
                    new_C10 += 4 * weight / dx * GridVel[idx].y * dposx;
                    new_C11 += 4 * weight / dx * GridVel[idx].y * dposy;
                }
            }
            ParticleVel[p] = new_vel;
            ParticlePos[p] += dt * new_vel;
            ParticleC[p].v00 = new_C00;
            ParticleC[p].v01 = new_C01;
            ParticleC[p].v10 = new_C10;
            ParticleC[p].v11 = new_C11;
            //ParticleJ[p] *= (1 + dt * (new_C00 + new_C11));

            Particle[p].transform.position = new Vector3(ParticlePos[p].x * solverDomainLength, ParticlePos[p].y * solverDomainLength, 0.0f);
            //Debug.Log(ParticlePos[p].ToString("f4"));
            //Matrix2x2.Log(ParticleC[p]," C " + p);

        }
    }

    void Start()
    {
        initArray();
        initParticle();

    }

    // Update is called once per frame
    void Update()
    {

        ParticleToGrid();
        Boundary();
        GridToParticle();
    }
}
