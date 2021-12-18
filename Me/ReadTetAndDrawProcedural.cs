using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Runtime.InteropServices;

public class Implicit3D : MonoBehaviour
{
    // Start is called before the first frame update

    int node_num;
    Vector3[] node_pos;

    int elem_num;
    int[] elem_idx;

    int face_num;
    int[] face_idx;

    public Material displayMaterial;

    private ComputeBuffer node_pos_buf;
    private ComputeBuffer face_idx_buf;
    void Start()
    {
        //ReadFile();
        node_num = 4;
        node_pos = new Vector3[node_num];
        node_pos[0] = new Vector3(0, 0, 0);
        node_pos[1] = new Vector3(0, 1, 0);
        node_pos[2] = new Vector3(0, 0, 1);
        node_pos[3] = new Vector3(0, 1, 1);
        face_num = 2;
        face_idx = new int[6];
        face_idx[0] = 0;
        face_idx[1] = 1;
        face_idx[2] = 2;
        face_idx[3] = 1;
        face_idx[4] = 3;
        face_idx[5] = 2;

        node_pos_buf = new ComputeBuffer(node_num, Marshal.SizeOf(typeof(Vector3)));
        node_pos_buf.SetData(node_pos);
        face_idx_buf = new ComputeBuffer(face_num * 3, Marshal.SizeOf(typeof(int)));
        face_idx_buf.SetData(face_idx);
        displayMaterial.SetBuffer("_vertices", node_pos_buf);
        displayMaterial.SetBuffer("_idx", face_idx_buf);
        /*
        node_pos_buf = new ComputeBuffer(node_num, Marshal.SizeOf(typeof(Vector3)));
        node_pos_buf.SetData(node_pos);
        face_idx_buf = new ComputeBuffer(face_num * 3, Marshal.SizeOf(typeof(int)));
        face_idx_buf.SetData(face_idx);
        displayMaterial.SetBuffer("_vertices", node_pos_buf);
        displayMaterial.SetBuffer("_idx", face_idx_buf);
        */
    }

    void ReadFile()
    {
        string[] textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/creeper.node");
        node_num = int.Parse(textTxt[0].Split(' ')[0]);
        node_pos = new Vector3[node_num];
        for (int i = 1;i < textTxt.Length-1;i++)
        {
            string[] splitTxt = textTxt[i].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            //for (int j = 0; j < splitTxt.Length; j++) Debug.Log(splitTxt[j]);
            node_pos[i - 1] = new Vector3(float.Parse(splitTxt[1]), float.Parse(splitTxt[2]), float.Parse(splitTxt[3]));
            Debug.Log(node_pos[i - 1]);
        }

        textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/creeper.ele");
        elem_num = int.Parse(textTxt[0].Split(' ')[0]);
        elem_idx = new int[elem_num * 4];
        for (int i = 1; i < textTxt.Length - 1; i++)
        {
            string[] splitTxt = textTxt[i].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            elem_idx[i * 4 + 0] = int.Parse(splitTxt[1]);
            elem_idx[i * 4 + 1] = int.Parse(splitTxt[2]);
            elem_idx[i * 4 + 2] = int.Parse(splitTxt[3]);
            elem_idx[i * 4 + 3] = int.Parse(splitTxt[4]);
        }

        textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/creeper.face");
        face_num = int.Parse(textTxt[0].Split(' ')[0]);
        face_idx = new int[face_num * 4];
        for (int i = 1; i < textTxt.Length - 1; i++)
        {
            string[] splitTxt = textTxt[i].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            face_idx[i * 3 + 0] = int.Parse(splitTxt[1]);
            face_idx[i * 3 + 1] = int.Parse(splitTxt[2]);
            face_idx[i * 3 + 2] = int.Parse(splitTxt[3]);
        }
    }

    private void OnRenderObject()
    {
        displayMaterial.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Triangles, face_num * 3, 1);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
