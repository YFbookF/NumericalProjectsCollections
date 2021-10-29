/*********************************************************************
 * viewport.h
 * Authored by Kris Hauser 2002-2003
 *
 * Defines a viewport class.
 *
 * Copyright (c) 2002 SquireSoft, used with permission.
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



#ifndef EDITOR_VIEWPORT_H
#define EDITOR_VIEWPORT_H

enum { VIEW_FRONT = 0, VIEW_BACK, VIEW_LEFT, VIEW_RIGHT, VIEW_TOP, VIEW_BOTTOM, VIEW_PERSPECTIVE };

#include <primitives.h>

struct editor_viewport
{
	editor_viewport();
	void set_view(int);
	void set_mode();

	int view;
	bool show_grid;
	bool smooth;
	bool wire;

	vec3_t position;
	float scale;
	float pitch,yaw;

	matrix4_t mat;

	int x,y,w,h;
	float n, f;
	float dist;

	bool clicked(int mx, int my) const;
	void zoom(float s);
	void scroll(float x, float y);
	void truck(float z);
	void push( float z);
	void refresh_matrix();

	void get_view_vector(vec3_t) const;
	void get_movement_vector(float x, float y, vec3_t) const;
	float get_depth(int x, int y) const;
	
	void get_click_vector(int mx, int my, vec3_t) const;
	void get_click_source(int mx, int my, vec3_t) const;
};


#endif

