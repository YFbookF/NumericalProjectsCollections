q1.从角动量计算角速度
$$
\bold w = \bold R \bold I^{-1} \bold R^T \bold L
$$
R是旋转矩阵，I是inertia monentum

```
void RigidBody::ComputeAngularVelocity()
{
  double temp0, temp1, temp2;
 
  // temp = R^T * L 
  temp0 = R[0] * angularMomentumX + R[3] * angularMomentumY + R[6] * angularMomentumZ;
  temp1 = R[1] * angularMomentumX + R[4] * angularMomentumY + R[7] * angularMomentumZ;
  temp2 = R[2] * angularMomentumX + R[5] * angularMomentumY + R[8] * angularMomentumZ;

  // temp = I^{body}^{-1} * temp = diag(invIBodyX, invIBodyY, invIBodyZ) * temp;
  temp0 = inverseInertiaTensorAtRestX * temp0;
  temp1 = inverseInertiaTensorAtRestY * temp1;
  temp2 = inverseInertiaTensorAtRestZ * temp2;

  // angularVelocity = R * temp
  angularVelocityX = R[0] * temp0 + R[1] * temp1 + R[2] * temp2;
  angularVelocityY = R[3] * temp0 + R[4] * temp1 + R[5] * temp2;
  angularVelocityZ = R[6] * temp0 + R[7] * temp1 + R[8] * temp2;
}
```

q2.计算Inertia tensor
$$
inertia =  \bold R diag(intertia\_rest) \bold R^T
$$

```
void RigidBody::ComputeInertiaTensor(double inertiaTensor[9])
{
  //inertiaTensor = R * diag(inertiaTensorAtRestX, inertiaTensorAtRestY, inertiaTensorAtRestZ) * R^T
  
  inertiaTensor[0] = R[0] * inertiaTensorAtRestX * R[0] + R[1] * inertiaTensorAtRestY * R[1] + R[2] * inertiaTensorAtRestZ * R[2];
  inertiaTensor[1] = R[3] * inertiaTensorAtRestX * R[0] + R[4] * inertiaTensorAtRestY * R[1] + R[5] * inertiaTensorAtRestZ * R[2];
  inertiaTensor[2] = R[6] * inertiaTensorAtRestX * R[0] + R[7] * inertiaTensorAtRestY * R[1] + R[8] * inertiaTensorAtRestZ * R[2];

  inertiaTensor[4] = R[3] * inertiaTensorAtRestX * R[3] + R[4] * inertiaTensorAtRestY * R[4] + R[5] * inertiaTensorAtRestZ * R[5];
  inertiaTensor[5] = R[6] * inertiaTensorAtRestX * R[3] + R[7] * inertiaTensorAtRestY * R[4] + R[8] * inertiaTensorAtRestZ * R[5];

  inertiaTensor[8] = R[6] * inertiaTensorAtRestX * R[6] + R[7] * inertiaTensorAtRestY * R[7] + R[8] * inertiaTensorAtRestZ * R[8];

  // symmetric
  inertiaTensor[3] = inertiaTensor[1];
  inertiaTensor[6] = inertiaTensor[2];
  inertiaTensor[7] = inertiaTensor[5];
}
```

q3.euler intergration step

