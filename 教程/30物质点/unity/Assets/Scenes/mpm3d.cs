using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class mpm3d : MonoBehaviour
{
    public GameObject spherePrefab;
    public GameObject[] Particle;

    // 粒子参数
    private const int ParticleNum = 8192;
    private Vector3[] ParticlePos;
    private Vector3[] ParticleVel;
    private Matrix3x3[] ParticleC;
    private float[] ParticleJ;

    // 网格参数
    private const int GridSize = 64; // 网格数量
    private const float dx = 1 / (float)GridSize; // 网格长度
    private Vector3[] GridVel; // 网格速度
    private float[] GridMass; // 网格质量

    // 物理量
    private float E = 400; // 杨氏模量
    private int bound = 3; // 网格边界宽度
    private const float gravity = 9.8f;//重力
    private const float p_rho = 1; // 粒子密度
    private const float p_vol = (dx * 0.5f) * (dx * 0.5f); // 粒子体积
    private const float p_mass = p_rho * p_vol;// 粒子重量
    private const float dt = 0.0004f; // 步长
    private float solverDomainLength = 10; // 求解域长度

    private void initArray()
    {
        Particle = new GameObject[ParticleNum];
        ParticlePos = new Vector3[ParticleNum];
        ParticleVel = new Vector3[ParticleNum];
        ParticleC = new Matrix3x3[ParticleNum];
        ParticleJ = new float[ParticleNum];

        GridVel = new Vector3[GridSize * GridSize * GridSize];
        GridMass = new float[GridSize * GridSize * GridSize];
    }

    private void initParticle()
    {
        for (int i = 0; i < ParticleNum; i++)
        {
            ParticlePos[i].x = Random.Range(0.0f, 1.0f) * 0.4f + 0.15f;
            ParticlePos[i].y = Random.Range(0.0f, 1.0f) * 0.4f + 0.15f;
            ParticlePos[i].z = Random.Range(0.0f, 1.0f) * 0.4f + 0.15f;
            Particle[i] = Instantiate(spherePrefab, new Vector3(ParticlePos[i].x * solverDomainLength, ParticlePos[i].y * solverDomainLength, ParticlePos[i].z * solverDomainLength), Quaternion.identity);

            ParticleVel[i].x = 0;
            ParticleVel[i].y = -1;
            ParticleVel[i].z = 0;
            ParticleC[i] = new Matrix3x3();

            ParticleJ[i] = 1;

        }
    }

    private void ParticleToGrid()
    {
        for(int k = 0; k < GridSize;k++)
        {
            for(int j = 0;j < GridSize;j++)
            {
                for(int i = 0;i < GridSize;i++)
                {
                    int idx = k * GridSize * GridSize + j * GridSize + i;
                    GridVel[idx].x = GridVel[idx].y = GridVel[idx].z = 0;
                    GridMass[idx] = 0;
                }
            }
        }
        for (int p = 0; p < ParticleNum; p++)
        {
            float Xpx = ParticlePos[p].x / dx;
            float Xpy = ParticlePos[p].y / dx;
            float Xpz = ParticlePos[p].z / dx;
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            int basez = (int)Mathf.Floor(Xpz - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;
            float fz = Xpz - basez;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };
            float[] wz = { 0.5f * (1.5f - fz) * (1.5f - fz), 0.75f - (fz - 1.0f) * (fz - 1.0f), 0.5f * (fz - 0.5f) * (fz - 0.5f) };

            float stress = -dt * 4 * E * p_vol * (ParticleJ[p] - 1) / dx / dx;
            Matrix3x3 affine = Matrix3x3.identity() * stress + ParticleC[p] * p_mass;

            for(int k = 0; k < 3;k++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        if (basex + i < 0 || basex + i > GridSize)
                            continue;
                        if (basey + j < 0 || basey + j > GridSize)
                            continue;
                        if (basez + k < 0 || basez + k > GridSize)
                            continue;
                        int idx = (basez + k) * GridSize * GridSize + (basey + j) * GridSize + basex + i;
                        float weight = wx[i] * wy[j] * wz[k];
                        float dposx = (i - fx) * dx;
                        float dposy = (j - fy) * dx;
                        float dposz = (k - fz) * dx;

                        float aff = affine.v00 * dposx + affine.v01 * dposy + affine.v02 * dposz;
                        GridVel[idx].x += weight * (p_mass * ParticleVel[p].x + aff);

                        aff = affine.v10 * dposx + affine.v11 * dposy + affine.v12 * dposz;
                        GridVel[idx].y += weight * (p_mass * ParticleVel[p].y + aff);

                        aff = affine.v20 * dposx + affine.v21 * dposy + affine.v22 * dposz;
                        GridVel[idx].z += weight * (p_mass * ParticleVel[p].z + aff);

                        GridMass[idx] += weight * p_mass;
                    }
                }
            }



        }
    }


    private void Boundary()
    {
        for (int k = 0; k < GridSize; k++)
        {
            for (int j = 0; j < GridSize; j++)
            {
                for (int i = 0; i < GridSize; i++)
                {
                    int idx = k * GridSize * GridSize + j * GridSize + i;
                    if (GridMass[idx] > 0)
                        GridVel[idx] /= GridMass[idx];
                    GridVel[idx].y -= dt * gravity;
                    if (i < bound && GridVel[idx].x < 0)
                        GridVel[idx].x = 0;
                    if (i > GridSize - bound && GridVel[idx].x > 0)
                        GridVel[idx].x = 0;
                    if (j < bound && GridVel[idx].y < 0)
                        GridVel[idx].y = 0;
                    if (j > GridSize - bound && GridVel[idx].y > 0)
                        GridVel[idx].y = 0;
                    if (k < bound && GridVel[idx].z < 0)
                        GridVel[idx].z = 0;
                    if (k > GridSize - bound && GridVel[idx].z > 0)
                        GridVel[idx].z = 0;
                }

            }
        }
    }

    private void GridToParticle()
    {
        for (int p = 0; p < ParticleNum; p++)
        {
            float Xpx = ParticlePos[p].x / dx;
            float Xpy = ParticlePos[p].y / dx;
            float Xpz = ParticlePos[p].z / dx;
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            int basez = (int)Mathf.Floor(Xpz - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;
            float fz = Xpz - basez;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };
            float[] wz = { 0.5f * (1.5f - fz) * (1.5f - fz), 0.75f - (fz - 1.0f) * (fz - 1.0f), 0.5f * (fz - 0.5f) * (fz - 0.5f) };

            Vector3 new_vel = new Vector3(0.0f, 0.0f,0.0f);
            Matrix3x3 new_C = new Matrix3x3();

            for (int k = 0; k < 3; k++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        if (basex + i < 0 || basex + i > GridSize)
                            continue;
                        if (basey + j < 0 || basey + j > GridSize)
                            continue;
                        if (basez + k < 0 || basez + k > GridSize)
                            continue;
                        int idx = (basez + k) * GridSize * GridSize + (basey + j) * GridSize + basex + i;
                        float weight = wx[i] * wy[j] * wz[k];
                        float dposx = (i - fx) * dx;
                        float dposy = (j - fy) * dx;
                        float dposz = (k - fz) * dx;

                        new_vel += weight * GridVel[idx];

                        new_C.v00 += 4 * weight / dx / dx * GridVel[idx].x * dposx;
                        new_C.v01 += 4 * weight / dx / dx * GridVel[idx].x * dposy;
                        new_C.v02 += 4 * weight / dx / dx * GridVel[idx].x * dposz;

                        new_C.v10 += 4 * weight / dx / dx * GridVel[idx].y * dposx;
                        new_C.v11 += 4 * weight / dx / dx * GridVel[idx].y * dposy;
                        new_C.v12 += 4 * weight / dx / dx * GridVel[idx].y * dposz;

                        new_C.v20 += 4 * weight / dx / dx * GridVel[idx].z * dposx;
                        new_C.v21 += 4 * weight / dx / dx * GridVel[idx].z * dposy;
                        new_C.v22 += 4 * weight / dx / dx * GridVel[idx].z * dposz;
                    }
                }
            }
            ParticleVel[p] = new_vel;
            ParticlePos[p] += dt * new_vel;
            ParticleC[p] = new_C;
            ParticleJ[p] *= (1 + dt * (new_C.v00 + new_C.v11 + new_C.v22));

            Particle[p].transform.position = new Vector3(ParticlePos[p].x * solverDomainLength, ParticlePos[p].y * solverDomainLength, ParticlePos[p].z * solverDomainLength);

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
