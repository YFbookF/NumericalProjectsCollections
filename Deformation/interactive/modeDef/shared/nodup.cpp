/*********************************************************************
 * nodup.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * An efficient class defining a set of (non duplicate) numbers.
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

#include "nodup.h"


duplicate_remover::duplicate_remover()
:used(NULL), list(NULL), cursize(0), maxsize(0), iter(0)
{}

void duplicate_remover::init(int size)
{
	SafeArrayDelete(used);
	SafeArrayDelete(list);
	maxsize = size;
	used = new int[size];
	list = new int[size];
	for(int i=0; i<size; i++)
		used[i] = -1;
	cursize = 0;
}

void duplicate_remover::insert(int i)
{
	if(used[i] != iter)
	{
		used[i] = iter;
		list[cursize++] = i;
	}
}

void duplicate_remover::clear()
{
	cursize = 0;
	iter++;
}


