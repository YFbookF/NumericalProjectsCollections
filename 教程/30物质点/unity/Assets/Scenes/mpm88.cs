using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class mpm88 : MonoBehaviour
{
    // taichi mpm88
    // https://github.com/clatterrr/NumericalComputation/blob/master/ParticleInCell/flipcopy.py
    public GameObject spherePrefab;
    public GameObject[] Particle;

    // 粒子参数
    private const int ParticleNum = 3096;
    private Vector2[] ParticlePos;
    private Vector2[] ParticleVel;
    private float[] ParticleC00;
    private float[] ParticleC01;
    private float[] ParticleC10;
    private float[] ParticleC11;
    private float[] ParticleJ;

    // 网格参数
    private const int GridSize = 128; // 网格数量
    private const float dx = 1 / (float)GridSize; // 网格长度
    private Vector2[] GridVel; // 网格速度
    private float[] GridMass; // 网格质量

    // 物理量
    private float E = 400; // 杨氏模量
    private int bound = 3; // 网格边界宽度
    private const float gravity = 9.8f;//重力
    private const float p_rho = 1; // 粒子密度
    private const float p_vol = (dx * 0.5f) * (dx * 0.5f); // 粒子体积
    private const float p_mass = p_rho * p_vol;// 粒子重量
    private const float dt = 0.0002f; // 步长
    private float solverDomainLength = 10; // 求解域长度

    private void initArray()
    {
        Particle = new GameObject[ParticleNum];
        ParticlePos = new Vector2[ParticleNum];
        ParticleVel = new Vector2[ParticleNum];
        ParticleC00 = new float[ParticleNum];
        ParticleC01 = new float[ParticleNum];
        ParticleC10 = new float[ParticleNum];
        ParticleC11 = new float[ParticleNum];
        ParticleJ = new float[ParticleNum];

        GridVel = new Vector2[GridSize * GridSize];
        GridMass = new float[GridSize * GridSize];
    }

    private void initParticle()
    {
        for(int i = 0;i < ParticleNum;i++)
        {
            ParticlePos[i].x = Random.Range(0.0f,1.0f) * 0.3f + 0.01f;
            ParticlePos[i].y = Random.Range(0.0f,1.0f) * 0.3f + 0.01f;
            Particle[i] = Instantiate(spherePrefab, new Vector3(ParticlePos[i].x * solverDomainLength, ParticlePos[i].y * solverDomainLength, 0), Quaternion.identity);

            ParticleVel[i].x = 0;
            ParticleVel[i].y = -1;

            ParticleJ[i] = 1;

        }
    }

    private void ParticleToGrid()
    {
        for(int j = 0;j < GridSize;j++)
        {
            for(int i = 0;i < GridSize;i++)
            {
                int idx = j * GridSize + i;
                GridVel[idx].x = GridVel[idx].y = 0;
                GridMass[idx] = 0;
            }
        }

        for(int p = 0;p < ParticleNum;p++)
        {
            float Xpx = ParticlePos[p].x / dx;
            float Xpy = ParticlePos[p].y / dx;
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };

            float stress = -dt * 4 * E * p_vol * (ParticleJ[p] - 1) / dx / dx;

            float affine00 = stress + p_mass * ParticleC00[p];
            float affine01 = p_mass * ParticleC01[p];
            float affine10 = p_mass * ParticleC10[p];
            float affine11 = stress + p_mass * ParticleC11[p];

            for(int j = 0;j < 3;j++)
            {
                for(int i = 0;i < 3;i++)
                {
                    if (basex + i < 0 || basex + i > GridSize)
                        continue;
                    if (basey + j < 0 || basey + j > GridSize)
                        continue;
                    int idx = (basey + j) * GridSize + basex + i;
                    float weight = wx[i] * wy[j];
                    float dposx = (i - fx) * dx;
                    float dposy = (j - fy) * dx;

                    float aff = affine00 * dposx + affine01 * dposy;
                    GridVel[idx].x += weight * (p_mass * ParticleVel[p].x + aff);

                    aff = affine10 * dposx + affine11 * dposy;
                    GridVel[idx].y += weight * (p_mass * ParticleVel[p].y + aff);

                    GridMass[idx] += weight * p_mass;
                }
            }


        }
    }


    private void Boundary()
    {
        for(int j = 0; j < GridSize;j++)
        {
            for(int i = 0;i < GridSize;i++)
            {
                int idx = j * GridSize + i;
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
                int obstacleSize = GridSize / 8;
                if(i > GridSize / 2 && i < GridSize / 2 + obstacleSize && j < obstacleSize)
                {
                    GridVel[idx].x = 0;
                    GridVel[idx].y = 0;
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
            int basex = (int)Mathf.Floor(Xpx - 0.5f);
            int basey = (int)Mathf.Floor(Xpy - 0.5f);
            float fx = Xpx - basex;
            float fy = Xpy - basey;

            float[] wx = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
            float[] wy = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };

            Vector2 new_vel = new Vector2(0.0f,0.0f);
            float new_C00 = 0;
            float new_C01 = 0;
            float new_C10 = 0;
            float new_C11 = 0;

            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (basex + i < 0 || basex + i > GridSize)
                        continue;
                    if (basey + j < 0 || basey + j > GridSize)
                        continue;
                    int idx = (basey + j) * GridSize + basex + i;
                    float weight = wx[i] * wy[j];
                    float dposx = (i - fx) * dx;
                    float dposy = (j - fy) * dx;

                    new_vel += weight * GridVel[idx];

                    new_C00 += 4 * weight / dx / dx * GridVel[idx].x * dposx;
                    new_C01 += 4 * weight / dx / dx * GridVel[idx].x * dposy;
                    new_C10 += 4 * weight / dx / dx * GridVel[idx].y * dposx;
                    new_C11 += 4 * weight / dx / dx * GridVel[idx].y * dposy;
                }
            }
            ParticleVel[p] = new_vel;
            ParticlePos[p] += dt * new_vel;
            ParticleC00[p] = new_C00;
            ParticleC01[p] = new_C01;
            ParticleC10[p] = new_C10;
            ParticleC11[p] = new_C11;
            ParticleJ[p] *= (1 + dt * (new_C00 + new_C11));

            Particle[p].transform.position = new Vector3(ParticlePos[p].x * solverDomainLength, ParticlePos[p].y * solverDomainLength, 0.0f);

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
