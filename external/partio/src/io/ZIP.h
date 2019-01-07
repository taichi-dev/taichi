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
*/

#ifndef __ZIP__
#define __ZIP__

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace Partio{
struct ZipFileHeader;
//#####################################################################
// Functions Gzip_Out/Gzip_In - Create streams that read/write .gz
//#####################################################################
std::istream* Gzip_In(const std::string& filename,std::ios::openmode mode);
std::ostream* Gzip_Out(const std::string& filename,std::ios::openmode mode);
//#####################################################################
// Class ZipFileWriter
//#####################################################################
class ZipFileWriter
{
    std::ofstream ostream;
    std::vector<ZipFileHeader*> files;
public:

//#####################################################################
    ZipFileWriter(const std::string& filename);
    virtual ~ZipFileWriter();
    std::ostream* Add_File(const std::string& filename,const bool binary=true);
//#####################################################################
};

//#####################################################################
// Class ZipFileReader
//#####################################################################
class ZipFileReader
{
    std::ifstream istream;
public:
    std::map<std::string,ZipFileHeader*> filename_to_header;
    
//#####################################################################
    ZipFileReader(const std::string& filename);
    virtual ~ZipFileReader();
    std::istream* Get_File(const std::string& filename,const bool binary=true);
    void Get_File_List(std::vector<std::string>& filenames) const;
private:
    bool Find_And_Read_Central_Header();
//#####################################################################
};
}
#endif
