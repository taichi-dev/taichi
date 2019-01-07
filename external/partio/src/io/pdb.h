/*
PARTIO SOFTWARE
Copyright 2010 Disney Enterprises, Inc. All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.

* The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
Studios" or the names of its contributors may NOT be used to
endorse or promote products derived from this software without
specific prior written permission from Walt Disney Pictures.

Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

This code is based on the Gifts/readpdb directory of Autodesk Maya

NOTE: some modifications were made and "32-bit" versions of the
structures were created to allow us to assume the pointers were
32-bit integers instead of system dependent pointers. As originally
defined, this format would be architecture dependent (behave
differently on 32-bit and 64-bit architectures). Maya's writers
may or may not also be doing this.  If you have any files that
don't work, please let me know. But in general, it would be
difficult/error prone to detect what type the PDB file is. 

*/


#define PDB_VECTOR          1
#define PDB_REAL            2
#define PDB_LONG            3
#define PDB_CHAR            4
#define PDB_POINTERT        5

typedef float Real;

typedef struct { Real x,y,z; } Vector;


typedef struct Channel_Data32 {
    int                 type;
    unsigned int        datasize;
    unsigned int        blocksize;
    int                 num_blocks;
    unsigned int        block; //was void **block;
} Channel_Data32;

typedef struct {
    int                 type;
    unsigned int        datasize;
    unsigned int        blocksize;
    int                 num_blocks;
    void                **block;
} Channel_Data;

typedef struct Channel {

    char                    *name;
    int                     type;
    unsigned int            size;
    unsigned int            active_start;
    unsigned int            active_end;

    char                    hide;
    char                    disconnect;
    Channel_Data            *data;

    struct Channel          *link;
    struct Channel          *next;              } Channel;

typedef struct Channel32 {
    unsigned int  name;
    int						type;
    unsigned int            size;
    unsigned int            active_start;
    unsigned int            active_end;

    char                    hide;
    char                    disconnect;
    unsigned int  data;

    unsigned int  link;
    unsigned int  next;
} Channel32;



typedef struct {

    char                magic;
    unsigned short      swap;
    char                encoding;
    char                type;           }       Channel_io_Header;

typedef struct {

    int         numAttributes;
    int         numParticles;
    float       time;
    short      *types;
    char      **names;
    void      **data;     }  PDBdata;

typedef struct {

    int         numAttributes;
    int         numParticles;
    float       time;
    short      *types;
    char      **names;
    unsigned int  data;     }  PDBdata32;
#define	PDB_MAGIC 670


typedef	struct {

    int			    magic;
    unsigned short	    swap;
    float		    version;
    float		    time;
    unsigned		    data_size;
    unsigned		    num_data;

    char		    padding[32];

    Channel		    **data;	    
	}	    PDB_Header;


typedef struct {
    int                magic;
    unsigned short        swap;
    float            version;
    float            time;
    unsigned            data_size;
    unsigned            num_data;

    char            padding[32];

    unsigned int  data;   
}		PDB_Header32;

