/*********************************************************************
 * smesh.h
 * Authored by Kris Hauser 2002-2003
 *
 * A simple mesh structure.
 *
 * Copyright (c) 2003, Regents of the University of California
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

#ifndef SURFACE_MESH_H
#define SURFACE_MESH_H

#include <primitives.h>
#include <fstream.h>

struct surface_mesh
{
  surface_mesh();
  ~surface_mesh();
  bool load(istream&);
  bool save(ostream&);
  bool load(const char* fn);
  bool save(const char* fn);

  //creates this as an instance of the given mesh (shares the same topology
  //as the other surface mesh, but the coordinates are different)
  void instantiate(surface_mesh*);
  //calculates the vertex normals by averaging face normals
  void calculate_normals();
  //initializes the rest of the structure from just the vertices and faces
  void init_verts_faces();

  void draw();

  bool is_valid() const;
  void get_aabb(float_t bb [6]) const;
  int ray_cast(const vec3_t src, const vec3_t dir, vec3_t out) const;
  bool face_ray_intersect(const vec3_t src, const vec3_t dir, int f, vec3_t out, float_t& dist) const;
  bool face_ray_intersect_bf(const vec3_t src, const vec3_t dir, int f, vec3_t out, float_t& dist) const;

  //Data types.
  typedef float_t texcoord_t [2];
  typedef float_t rgba_t [4];
  struct vertex_ref_t {
    int vertex;
    int normal;
    int texcoords;
  };
  typedef vertex_ref_t face_t [3];
  struct material_t {
    rgba_t ambient, diffuse, specular, emissive;
    float_t specular_strength;
  };
  struct face_group_t {
    int face_start, face_end;
    int material;
  };

  //Mesh data.  These are the data that are saved and loaded.
  vec3_t* vertices;
  vec3_t* normals;
  texcoord_t* texcoords;
  face_t* faces;
  material_t* materials;
  face_group_t* face_groups;
  int num_vertices;
  int num_normals;
  int num_texcoords;
  int num_faces;
  int num_materials;
  int num_face_groups;

  //this is a flag set by instantiate().
  //linked meshes share faces, materials, and face groups
  //so don't delete them on destruction.
  bool linked;
};


#endif  //SURFACE_MESH_H
