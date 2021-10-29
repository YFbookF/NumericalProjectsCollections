/*********************************************************************
 * myarray.h
 * Authored by Kris Hauser 2002-2003
 *
 * A templated array class with variable growth size.
 *
 * Copyright (c) 2002,2003 SquireSoft, used with permission.
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


#ifndef MY_ARRAY_H
#define MY_ARRAY_H

#include "stdincludes.h"

template <class type>
class myarray
{
public:
	myarray();
	myarray(int size);
	~myarray() { clear(); }

	inline int size() const { return cursize; }		//this is the "official" size
	inline int capacity() const { return maxsize; }	//this is the amount of memory allocated
	void resize(int size);				//this will resize only if size > maxsize, and copies items
	void resize_strict(int size);		//this forces a resize, no matter what size is given, and doesnt copies items
	void resize_strict_copy(int size);	//this forces a resize, and does copy items
	void clear();

	void set_grow_size(int gs) { growsize = gs; }

	void operator = (const myarray<type>&);
	inline operator type* () { return list; }
	inline operator const type* () const { return list; }
	inline type& operator [] (int i) { return list[i]; }
	inline const type& operator [] (int i) const { return list[i]; }

private:
	type* list;
	int cursize;
	int maxsize;
	int growsize;
};

template <class type>
myarray<type>::myarray()
:list(NULL), cursize(0), maxsize(0), growsize(1)
{}

template <class type>
myarray<type>::myarray(int size)
:list(NULL), cursize(0), maxsize(0), growsize(1)
{
	resize(size);
}

template <class type>
void myarray<type>::resize(int size)
{
	if(size > maxsize)
	{
		int newsize = max(size, maxsize+growsize);
		resize_strict_copy(newsize);
	}
	cursize = size;
}

template <class type>
void myarray<type>::resize_strict(int size)
{
	if(size != maxsize)
	{
		type* newlist = new type[size];
		SafeArrayDelete(list);
		list = newlist;
	}
	maxsize = size;
	cursize = size;
}

template <class type>
void myarray<type>::resize_strict_copy(int size)
{
	if(size != maxsize)
	{
		type* newlist = new type[size];
		int ncopy = min(size, cursize);
		for(int i=0; i<ncopy; i++)
			newlist[i] = list[i];
		SafeArrayDelete(list);
		list = newlist;
	}
	maxsize = size;
	cursize = size;
}

template <class type>
void myarray<type>::clear()
{
	SafeArrayDelete(list);
	maxsize = 0;
	cursize = 0;
}

template <class type>
void myarray<type>::operator = (const myarray<type>& l)
{
	if(maxsize < l.cursize)
		resize_strict(l.cursize);
	cursize = l.cursize;
	for(int i=0; i<cursize; i++)
		list[i] = l.list[i];
}

#endif

