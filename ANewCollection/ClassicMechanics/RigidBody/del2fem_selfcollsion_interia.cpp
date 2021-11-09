void ApplyRigidImpactZone
//del2fem
(std::vector<double>& aUVWm, // (in,out)RIZで更新された中間速度
 ////
 const std::vector< std::set<int> >& aRIZ,  // (in)各RIZに属する節点の集合(set)の配列
 const std::vector<double>& aXYZ, // (in) 前ステップの節点の位置の配列
 const std::vector<double>& aUVWm0) // (in) RIZを使う前の中間速度
{
  for(const auto & iriz : aRIZ){
    std::vector<int> aInd; // index of points belong to this RIZ
    for(auto jtr=iriz.begin();jtr!=iriz.end();jtr++){
      aInd.push_back(*jtr);
    }
    dfm2::CVec3d gc(0,0,0); // 重心位置
    dfm2::CVec3d av(0,0,0); // 平均速度
    for(int ino : aInd){
      gc += dfm2::CVec3d(aXYZ[  ino*3+0],aXYZ[  ino*3+1],aXYZ[  ino*3+2]);
      av += dfm2::CVec3d(aUVWm0[ino*3+0],aUVWm0[ino*3+1],aUVWm0[ino*3+2]);
    }
    gc /= (double)aInd.size();
    av /= (double)aInd.size();
    dfm2::CVec3d L(0,0,0); // 角運動量
    double I[9] = {0,0,0, 0,0,0, 0,0,0}; // 慣性テンソル
    for(int ino : aInd){
      dfm2::CVec3d p(aXYZ[  ino*3+0],aXYZ[  ino*3+1],aXYZ[  ino*3+2]);
      dfm2::CVec3d v(aUVWm0[ino*3+0],aUVWm0[ino*3+1],aUVWm0[ino*3+2]);
      L += Cross(p-gc,v-av);
      dfm2::CVec3d q = p-gc;
      I[0] += v.dot(v) - q[0]*q[0];  I[1] +=          - q[0]*q[1];  I[2] +=          - q[0]*q[2];
      I[3] +=          - q[1]*q[0];  I[4] += v.dot(v) - q[1]*q[1];  I[5] +=          - q[1]*q[2];
      I[6] +=          - q[2]*q[0];  I[7] +=          - q[2]*q[1];  I[8] += v.dot(v) - q[2]*q[2];
    }
    // 角速度を求める
    double Iinv[9];
    CalcInvMat3(Iinv,I);
    dfm2::CVec3d omg;
    omg.p[0] = Iinv[0]*L.x + Iinv[1]*L.y + Iinv[2]*L.z;
    omg.p[1] = Iinv[3]*L.x + Iinv[4]*L.y + Iinv[5]*L.z;
    omg.p[2] = Iinv[6]*L.x + Iinv[7]*L.y + Iinv[8]*L.z;
    // 中間速度の更新
    for(int ino : aInd){
      dfm2::CVec3d p(aXYZ[  ino*3+0],aXYZ[  ino*3+1],aXYZ[  ino*3+2]);
      dfm2::CVec3d rot = -Cross(p-gc,omg);
      aUVWm[ino*3+0] = av.x + rot.x;
      aUVWm[ino*3+1] = av.y + rot.y;
      aUVWm[ino*3+2] = av.z + rot.z;
    }
  }
}