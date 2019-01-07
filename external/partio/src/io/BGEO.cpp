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

#include "../Partio.h"
#include "PartioEndian.h"
#include "../core/ParticleHeaders.h"
#include "ZIP.h"

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

namespace Partio {

using namespace std;

void writeHoudiniStr(ostream &ostream, const string &s) {
  write<BIGEND>(ostream, (short)s.size());
  ostream.write(s.c_str(), s.size());
}

ParticlesDataMutable *readBGEO(const char *filename, const bool headersOnly) {
}

bool writeBGEO(const char *filename,
               const ParticlesData &p,
               const bool compressed) {
  auto_ptr<ostream> output(
      compressed ? Gzip_Out(filename, ios::out | ios::binary)
                 : new ofstream(filename, ios::out | ios::binary));

  if (!*output) {
    cerr << "Partio Unable to open file " << filename << endl;
    return false;
  }

  int magic = ((((('B' << 8) | 'g') << 8) | 'e') << 8) | 'o';
  char versionChar = 'V';
  int version = 5;
  int nPoints = p.numParticles();
  int nPrims = 1;
  int nPointGroups = 0;
  int nPrimGroups = 0;
  int nPointAttrib = p.numAttributes() - 1;
  int nVertexAttrib = 0;
  int nPrimAttrib = 1;
  int nAttrib = 0;

  write<BIGEND>(*output, magic, versionChar, version, nPoints, nPrims,
                nPointGroups);
  write<BIGEND>(*output, nPrimGroups, nPointAttrib, nVertexAttrib, nPrimAttrib,
                nAttrib);

  vector<ParticleAttribute> handles;
  vector<ParticleAccessor> accessors;
  vector<int> attrOffsets;
  bool foundPosition = false;
  int particleSize = 4;
  for (int i = 0; i < p.numAttributes(); i++) {
    ParticleAttribute attr;
    p.attributeInfo(i, attr);
    if (attr.name == "position") {
      attrOffsets.push_back(0);
      foundPosition = true;
    } else {
      writeHoudiniStr(*output, attr.name);
      if (attr.type == INDEXEDSTR) {
        int houdiniType = 4;
        unsigned short size = attr.count;
        const std::vector<std::string> &indexTable = p.indexedStrs(attr);
        int numIndexes = indexTable.size();
        write<BIGEND>(*output, size, houdiniType, numIndexes);
        for (int i = 0; i < numIndexes; i++)
          writeHoudiniStr(*output, indexTable[i]);
      } else {
        int houdiniType = 0;
        switch (attr.type) {
          case FLOAT:
            houdiniType = 0;
            break;
          case INT:
            houdiniType = 1;
            break;
          case VECTOR:
            houdiniType = 5;
            break;
          case INDEXEDSTR:
          case NONE:
            assert(false);
            houdiniType = 0;
            break;
        }
        unsigned short size = attr.count;
        write<BIGEND>(*output, size, houdiniType);
        for (int i = 0; i < attr.count; i++) {
          int defaultValue = 0;
          write<BIGEND>(*output, defaultValue);
        }
      }
      attrOffsets.push_back(particleSize);
      particleSize += attr.count;
    }
    handles.push_back(attr);
    accessors.push_back(ParticleAccessor(handles.back()));
  }
  if (!foundPosition) {
    cerr << "Partio: didn't find attr 'position' while trying to write GEO"
         << endl;
    return false;
  }

  ParticlesData::const_iterator iterator = p.begin();
  for (size_t i = 0; i < accessors.size(); i++)
    iterator.addAccessor(accessors[i]);

  int *buffer = new int[particleSize];
  for (ParticlesData::const_iterator end = p.end(); iterator != end;
       ++iterator) {
    for (unsigned int attrIndex = 0; attrIndex < handles.size(); attrIndex++) {
      ParticleAttribute &handle = handles[attrIndex];
      ParticleAccessor &accessor = accessors[attrIndex];
      // TODO: this violates strict aliasing, we could just go to char* and make
      // a different endian swapper
      const int *data = accessor.raw<int>(iterator);
      for (int k = 0; k < handle.count; k++) {
        buffer[attrOffsets[attrIndex] + k] = data[k];
        BIGEND::swap(buffer[attrOffsets[attrIndex] + k]);
      }
    }
    // set homogeneous coordinate
    float *w = (float *)&buffer[3];
    *w = 1.;
    BIGEND::swap(*w);
    output->write((char *)buffer, particleSize * sizeof(int));
  }
  delete[] buffer;

  // Write primitive attribs
  writeHoudiniStr(*output, "generator");
  write<BIGEND>(*output, (short)0x1);  // count 1
  write<BIGEND>(*output, (int)0x4);    // type 4 index
  write<BIGEND>(*output, (int)0x1);    // type 4 index
  writeHoudiniStr(*output, "papi");

  // Write the primitive
  write<BIGEND>(*output, (int)0x8000);
  write<BIGEND>(*output, (int)nPoints);
  if (nPoints > (int)1 << 16)
    for (int i = 0; i < nPoints; i++)
      write<BIGEND>(*output, (int)i);
  else
    for (int i = 0; i < nPoints; i++)
      write<BIGEND>(*output, (unsigned short)i);
  write<BIGEND>(*output, (int)0);

  // Write extra
  write<BIGEND>(*output, (char)0x00);
  write<BIGEND>(*output, (char)0xff);

  // success
  return true;
}

}  // namespace Partio
