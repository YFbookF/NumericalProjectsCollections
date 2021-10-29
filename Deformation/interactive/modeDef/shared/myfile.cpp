/*********************************************************************
 * myfile.cpp
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

#include "myfile.h"
#include <stdlib.h>
#include <string.h>

#if (defined(PS2) && !defined(PS2LINUX))

#define INVALID_HANDLE_VALUE -1
#define IS_INVALID_HANDLE(handle) (handle < 0)
#define FILEOPEN_R(fn) sceOpen(fn,SCE_RDONLY)
#define FILEOPEN_W(fn) sceOpen(fn,SCE_WRONLY)
#define FILEOPEN_RW(fn) sceOpen(fn,SCE_RDWR)
#define FILECLOSE(f) sceClose(f)
#define FILESEEK(f,pos,whence) sceLseek(f,pos,whence)
#define FILEPOSITION(f) sceLseek(f,0,SCE_SEEK_CUR)
#define FILEREADDATA(f,data,size) sceRead(f, data, size)
#define FILEWRITEDATA(f,data,size) sceWrite(f, data, size)

#else

#define INVALID_HANDLE_VALUE NULL
#define IS_INVALID_HANDLE(handle) (handle == NULL)
#define FILEOPEN_R(fn) fopen(fn,"rb")
#define FILEOPEN_W(fn) fopen(fn,"wb")
#define FILEOPEN_RW(fn) fopen(fn,"w+")
#define FILECLOSE(f) fclose(f)
#define FILESEEK(f,pos,whence) fseek(f,pos,whence)
#define FILEPOSITION(f) ftell(f)
#define FILEREADDATA(f,data,size) fread(data, 1, size, f)
#define FILEWRITEDATA(f,data,size) fwrite(data, 1, size, f)

#endif //(defined(PS2) && !defined(PS2LINUX))

enum { MODE_NONE,
	MODE_MYFILE, 
	MODE_EXTFILE,
	MODE_DATA
};

myFile::myFile()
:_file(INVALID_HANDLE_VALUE)
{
}

myFile::~myFile()
{
	Close();
}

void myFile::Close()
{
	if(_srctype == MODE_MYFILE && _file != INVALID_HANDLE_VALUE)
	{
	  FILECLOSE(_file);
	}
	_srctype = MODE_NONE;
	_mode = 0;
	_file = INVALID_HANDLE_VALUE;
	_datafile = NULL;
	_pos = NULL;
	_dataend = NULL;
}

bool myFile::Open(void* data, int size, int openmode)
{
	Close();

	if(!data)
		return false;
	if(size < 0)
		return false;

	_srctype = MODE_DATA;
	if(openmode == 0)
		return false;

	_datafile = (unsigned char*)data;
	_pos = _datafile;
	_dataend = _datafile + size;
	_datasize = size;
	_mode = openmode;
	return true;
}

bool myFile::Open(const char* fn, int openmode)
{
	Close();

	if(openmode == 0)
		return false;

	if(openmode & FILEREAD)
	{
		if(openmode & FILEWRITE)
		  _file = FILEOPEN_RW(fn);
		else
		  _file = FILEOPEN_R(fn);
	}
	else if(openmode & FILEWRITE)
	{
	  _file = FILEOPEN_W(fn);
	}

	if(IS_INVALID_HANDLE(_file))
		return false;
	_srctype = MODE_MYFILE;
	_mode = openmode;
	return true;
}


bool myFile::Open(HANDLE f, int openmode)
{
	Close();

	_srctype = MODE_EXTFILE;
	if(openmode == 0)
		return false;

	_file = f;
	_mode = openmode;
	return true;
}

int myFile::Position() const
{
	switch(_srctype)
	{
	case MODE_MYFILE:
	case MODE_EXTFILE:
	  return FILEPOSITION(_file);
	case MODE_DATA:
		return (int)_pos - (int)_datafile;
	}
	return -1;
}

bool myFile::Seek(int p, int from)
{
	switch(_srctype)
	{
	case MODE_MYFILE:
	case MODE_EXTFILE:
	  return FILESEEK(_file, p, from) != 0;
	  break;
	case MODE_DATA:
		switch (from)
		{
		case FILE_CURRENT:
			if(_pos + p >= _dataend || _pos + p < _datafile)
				return false;
			_pos += p;
			break;
		case FILE_BEGIN:
			if(_datafile + p >= _dataend || p < 0)
				return false;
			_pos = _datafile + p;
			break;
		case FILE_END:
			if(_dataend + p < _datafile || p > 0)
				return false;
			_pos = _dataend + p;
			break;
		}
	}
	return true;
}

int myFile::Length()
{
  switch(_srctype) {
  case MODE_MYFILE:
  case MODE_EXTFILE:
    {
    if(!_file)
      return 0;
    int p = Position();
    Seek(0, FILE_END);
    int l = Position();
    Seek(p, FILE_BEGIN);
    return l;
    }
  case MODE_DATA:
    return _datasize;
  }
  return 0;
}

bool myFile::ReadData(void* d, int size)
{
	if(_mode & FILEREAD)
	{
		switch(_srctype)
		{
		case MODE_MYFILE:
		case MODE_EXTFILE:
		  int s;
		  s = FILEREADDATA(_file, d, size);
		  //ReportError("Read %d bytes\n", s);
		  if(s != size)
		    return false;
		  return true;
		case MODE_DATA:
			if(_pos + size >= _dataend)
				return false;
			memcpy(d, _pos, size);
			_pos += size;
			return true;
		}
	}
	return false;
}

bool myFile::WriteData(const void* d, int size)
{
	if(_mode & FILEWRITE)
	{
		switch(_srctype)
		{
		case MODE_MYFILE:
		case MODE_EXTFILE:
		  if(FILEWRITEDATA(_file, d, size) != size)
		    return false;
		  return true;
		case MODE_DATA:
			if(_pos + size >= _dataend)
				return false;
			memcpy(_pos, d, size);
			_pos += size;
			return true;
		}
	}
	return false;
}



int ReadChar(HANDLE file)
{
	char c;
	if(FILEWRITEDATA(file, &c, 1) != 1)
	  return 0;
	return c;
}

bool myFile::ReadString(char* str, int bufsize)
{
	if(_mode & FILEREAD)
	{
		int i,c;
		switch(_srctype)
		{
		case MODE_MYFILE:
		case MODE_EXTFILE:
			for(i=0; i<bufsize; i++)
			{
				c = ReadChar(_file);
				if(c==EOF)
					return false;
				str[i]=c;
				if(c==0)
					return true;
			}
			break;
		case MODE_DATA:
			for(i=0; i<bufsize; i++)
			{
				if(_pos >= _dataend)
					return false;
				str[i]=*_pos;
				_pos++;
				if(str[i]==0)
					return true;
			}
			break;
		}
	}
	return false;
}

bool myFile::WriteString(const char* str)
{
	return WriteData(str, strlen(str)+1);
}
