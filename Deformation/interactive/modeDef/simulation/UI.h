/*********************************************************************
 * UI.h
 * Authored by Kris Hauser 2002-2003
 *
 * Defines a generic interface from GLUT into a framework program.
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

#include "viewport.h"


struct BaseMenu;

struct DeformMain
{
  DeformMain();
  ~DeformMain();

  void InitViewports();
  void Update();
  void Draw();
  void Reset();
  
  void NextMenu();
  void PrevMenu();

  void NextTool();
  void PrevTool();
  void Click(int button, int state);
  void Nav(bool up, bool down, bool left, bool right);
  void Drag(int button, float x, float y);
  void MoveTo(float x, float y);
  void Rotate(float x, float y);
  void Zoom(float z);
  void Key(unsigned char c);
  
  //HLIGHTSYSTEM light_system;
  editor_viewport viewport;

  BaseMenu* current_menu;
  int i_current_menu;
 
  int mouse_x, mouse_y;
  int mouse_dx, mouse_dy;
  int mouse_ddx, mouse_ddy; 

  float rotation_dx, rotation_dy;
  vec3_t rotation_d;
  vec3_t rotation_dd;
};
