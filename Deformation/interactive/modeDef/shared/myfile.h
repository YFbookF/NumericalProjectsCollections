/*********************************************************************
 * myfile.h
 * Authored by Kris Hauser 2002-2003
 *
 * A convenient platform-independent file interface.
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

#ifndef DUNC_MYFILE
#define DUNC_MYFILE

#include "stdincludes.h"

#if (defined(PS2) && !defined(PS2LINUX))

#include <sifdev.h>
typedef int HANDLE;

#define FILE_BEGIN SCE_SEEK_SET
#define FILE_CURRENT SCE_SEEK_CUR
#define FILE_END SCE_SEEK_END

#else

#include <stdio.h>
typedef FILE* HANDLE;

#define FILE_BEGIN SEEK_SET
#define FILE_CURRENT SEEK_CUR
#define FILE_END SEEK_END

#endif //(defined(PS2) && !defined(PS2LINUX))

#define DATASIZE_UNKNOWN -1
#define FILEREAD 0x1
#define FILEWRITE 0x2

class myFile
{
public:
	myFile();
	~myFile();
	bool Open(const char*, int openmode = FILEREAD | FILEWRITE);
	bool Open(HANDLE, int openmode = FILEREAD | FILEWRITE);
	bool Open(void*, int = DATASIZE_UNKNOWN, int openmode = FILEREAD | FILEWRITE);
	void Close();

	bool Seek(int, int from = FILE_CURRENT);
	int Position() const;
	int Length();

	bool ReadData(void*, int size);
	bool WriteData(const void*, int size);

	bool ReadString(char*, int bufsize);
	bool WriteString(const char*);

private:
	int _mode;
	int _srctype;

	HANDLE _file;
	unsigned char* _datafile;
	unsigned char* _pos;
	unsigned char* _dataend;
	int _datasize;
};



template <class type>
bool ReadFile(myFile& f, type& t)
{
	return f.ReadData(&t, sizeof(t));
}

template <class type>
bool WriteFile(myFile& f, const type& t)
{
	return f.WriteData(&t, sizeof(t));
}



template <class type>
bool ReadArrayFile(myFile& f, type* t, int n)
{
	return f.ReadData(t, sizeof(type)*n);
}

template <class type>
bool WriteArrayFile(myFile& f, const type* t, int n)
{
	return f.WriteData(t, sizeof(type)*n);
}


#endif
