/*********************************************************************
 * deformobj.h
 * Authored by Kris Hauser 2002-2003
 *
 * Headers for the deformable object runtime classes.
 *
 * Copyright 2003, Regents of the University of California 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *********************************************************************/



#ifndef DEFORM_OBJECTS_H
#define DEFORM_OBJECTS_H

#include "deform.h"
#include "deformdata.h"
#include "math/geometry.h"

struct RigidBodyState
{
  matrix4_t mRotation;
  vec3_t vDisp, vVel, vAccel;
  vec3_t vAngDisp, vAngVel, vAngAccel;
};



struct DeformObject
{
public:
  DeformObject();
  virtual ~DeformObject();

  void Init(DeformObjectData*, ConstraintState*, float_t* position_buffer = NULL); 
  void DrawTetrahedra();

  inline bool IsCarton() const { return object_data->carton_embedded; }
  inline int NumMeshVertices() const { return object_data->NumVertices(); }
  inline int NumVertices() const { return state->NumVertices(); }
  inline int NumModes() const { return state->NumModes(); }
  inline int NumPolygons() const { return mesh_buffer.NumPolygons(); }
  inline int NumElements() const { return object_data->num_elements; }

  //position output
  inline float_t* GetPositionBuffer() const { return position_buffer; }
  inline void UpdatePositions() {
    state->GetPositions(position_buffer);
  }
  inline void UpdatePartialPositions(int* vertices, int n) {
    state->GetPositionsPartial(position_buffer, vertices, n);
  }
  inline void UpdateSelectedPositions(int* vertices, float_t* positions, int n) {
    state->GetPositionsSelected(positions, vertices, n);
  }

  inline void ApplyVertexImpulses(int* vertices, const vec3_t* impulses, int n) {
    state->applyImpulses((float_t*)impulses, vertices, n);
  }
  inline void ApplyVertexImpulse(int vertex, const vec3_t impulse) {
    state->applyImpulses(impulse, &vertex, 1);
  }
  inline void ApplyInternalImpulse(const vec3_t point, const vec3_t impulse) {
    int indices[4];
    bary_t bary;
    if(GetBarycentricCoords(point, bary, indices) != -1)
      ApplyBarycentricImpulse(bary, indices, impulse);
  }
  inline void ApplyBarycentricImpulse(bary_t bary, int indices [4], const vec3_t impulse) {
    vec3_t forces [4];
    vec3_multiply(forces[0], impulse, bary[0]);
    vec3_multiply(forces[1], impulse, bary[1]);
    vec3_multiply(forces[2], impulse, bary[2]);
    vec3_multiply(forces[3], impulse, bary[3]);
    state->applyImpulses(forces[0], indices, 4);
  }

  Constraint* AddVertexConstraint(int vertex);
  Constraint* AddBarycentricConstraint(const bary_t bary, const int ind [4]);
  Constraint* AddInternalConstraint(const vec3_t point);
  inline void SetConstraintPosition(Constraint* c, const vec3_t pos) {
    vec3_equal(c->pos, pos);
  }
  inline void SetConstraintVelocity(Constraint* c, const vec3_t vel) {
    vec3_equal(c->vel, vel);
  }
  inline void RemoveConstraint(Constraint* c) { state->RemoveConstraint(c); }

  void SetRigidBodyState(const RigidBodyState&);
  void AddRigidBodyForces(const matrix4_t mRotation,
			  const vec3_t dVel,
			  const vec3_t angVel,
			  const vec3_t dAngVel);
  bool use_centrifugal_effects;
  bool use_coriolis_effects;

  int GetBarycentricCoords(const vec3_t pos, bary_t bary, int ind[4]) const;


  //Helper functions for carton meshes
  float_t* CartonCoords(int v) const;
  int CartonTet(int v) const;
  int* CartonTetIndices(int v) const;

  //THESE USED TO BE IN DeformObjectInterface
  void Draw();

  //these five functions ought to be overwritten if the mesh has a different
  //structure than the normal one
  virtual void Update();
  virtual void UpdateMesh();   //added at UCB
  virtual void ApplyMeshVertexImpulse(int v, const vec3_t f);
  virtual void ApplyMeshBarycentricImpulse(const vec3_t bary, int ind [3], const vec3_t f);
  virtual Constraint* AddMeshVertexConstraint(int v);
  virtual Constraint* AddMeshBarycentricConstraint(const vec3_t bary, int ind [3]);

  Constraint* ConstrainMeshVertex(int v, vec3_t pos);
  Constraint* ConstrainMeshBarycentric(const vec3_t bary, int ind [3], vec3_t pos);

  void RayForce(const Ray& r, float_t force);
  Constraint* RayVertexConstrain(const Ray& r);
  Constraint* RayConstrain(const Ray& r);


  bool buffered;
  float_t* position_buffer;

  ConstraintState* state;
  DeformObjectData* object_data;
  DeformMeshData mesh_buffer;

  DeformTetPointers tets;
};





#endif
