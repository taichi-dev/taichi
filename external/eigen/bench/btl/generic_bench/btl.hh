//=====================================================
// File   :  btl.hh
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//=====================================================
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
#ifndef BTL_HH
#define BTL_HH

#include "bench_parameter.hh"
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "utilities.h"

#if (defined __GNUC__)
#define BTL_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define BTL_ALWAYS_INLINE inline
#endif

#if (defined __GNUC__)
#define BTL_DONT_INLINE __attribute__((noinline))
#else
#define BTL_DONT_INLINE
#endif

#if (defined __GNUC__)
#define BTL_ASM_COMMENT(X)  asm("#" X)
#else
#define BTL_ASM_COMMENT(X)
#endif

#ifdef __SSE__
#include "xmmintrin.h"
// This enables flush to zero (FTZ) and denormals are zero (DAZ) modes:
#define BTL_DISABLE_SSE_EXCEPTIONS()  { _mm_setcsr(_mm_getcsr() | 0x8040); }
#else
#define BTL_DISABLE_SSE_EXCEPTIONS()
#endif

/** Enhanced std::string
*/
class BtlString : public std::string
{
public:
    BtlString() : std::string() {}
    BtlString(const BtlString& str) : std::string(static_cast<const std::string&>(str)) {}
    BtlString(const std::string& str) : std::string(str) {}
    BtlString(const char* str) : std::string(str) {}

    operator const char* () const { return c_str(); }

    void trim( bool left = true, bool right = true )
    {
        int lspaces, rspaces, len = length(), i;
        lspaces = rspaces = 0;

        if ( left )
            for (i=0; i<len && (at(i)==' '||at(i)=='\t'||at(i)=='\r'||at(i)=='\n'); ++lspaces,++i);

        if ( right && lspaces < len )
            for(i=len-1; i>=0 && (at(i)==' '||at(i)=='\t'||at(i)=='\r'||at(i)=='\n'); rspaces++,i--);

        *this = substr(lspaces, len-lspaces-rspaces);
    }

    std::vector<BtlString> split( const BtlString& delims = "\t\n ") const
    {
        std::vector<BtlString> ret;
        unsigned int numSplits = 0;
        size_t start, pos;
        start = 0;
        do
        {
            pos = find_first_of(delims, start);
            if (pos == start)
            {
                ret.push_back("");
                start = pos + 1;
            }
            else if (pos == npos)
                ret.push_back( substr(start) );
            else
            {
                ret.push_back( substr(start, pos - start) );
                start = pos + 1;
            }
            //start = find_first_not_of(delims, start);
            ++numSplits;
        } while (pos != npos);
        return ret;
    }

    bool endsWith(const BtlString& str) const
    {
        if(str.size()>this->size())
            return false;
        return this->substr(this->size()-str.size(),str.size()) == str;
    }
    bool contains(const BtlString& str) const
    {
        return this->find(str)<this->size();
    }
    bool beginsWith(const BtlString& str) const
    {
        if(str.size()>this->size())
            return false;
        return this->substr(0,str.size()) == str;
    }

    BtlString toLowerCase( void )
    {
        std::transform(begin(), end(), begin(), static_cast<int(*)(int)>(::tolower) );
        return *this;
    }
    BtlString toUpperCase( void )
    {
        std::transform(begin(), end(), begin(), static_cast<int(*)(int)>(::toupper) );
        return *this;
    }

    /** Case insensitive comparison.
    */
    bool isEquiv(const BtlString& str) const
    {
        BtlString str0 = *this;
        str0.toLowerCase();
        BtlString str1 = str;
        str1.toLowerCase();
        return str0 == str1;
    }

    /** Decompose the current string as a path and a file.
        For instance: "dir1/dir2/file.ext" leads to path="dir1/dir2/" and filename="file.ext"
    */
    void decomposePathAndFile(BtlString& path, BtlString& filename) const
    {
        std::vector<BtlString> elements = this->split("/\\");
        path = "";
        filename = elements.back();
        elements.pop_back();
        if (this->at(0)=='/')
            path = "/";
        for (unsigned int i=0 ; i<elements.size() ; ++i)
            path += elements[i] + "/";
    }
};

class BtlConfig
{
public:
  BtlConfig()
    : overwriteResults(false), checkResults(true), realclock(false), tries(DEFAULT_NB_TRIES)
  {
    char * _config;
    _config = getenv ("BTL_CONFIG");
    if (_config!=NULL)
    {
      std::vector<BtlString> config = BtlString(_config).split(" \t\n");
      for (unsigned int i = 0; i<config.size(); i++)
      {
        if (config[i].beginsWith("-a"))
        {
          if (i+1==config.size())
          {
            std::cerr << "error processing option: " << config[i] << "\n";
            exit(2);
          }
          Instance.m_selectedActionNames = config[i+1].split(":");

          i += 1;
        }
        else if (config[i].beginsWith("-t"))
        {
          if (i+1==config.size())
          {
            std::cerr << "error processing option: " << config[i] << "\n";
            exit(2);
          }
          Instance.tries = atoi(config[i+1].c_str());

          i += 1;
        }
        else if (config[i].beginsWith("--overwrite"))
        {
          Instance.overwriteResults = true;
        }
        else if (config[i].beginsWith("--nocheck"))
        {
          Instance.checkResults = false;
        }
        else if (config[i].beginsWith("--real"))
        {
          Instance.realclock = true;
        }
      }
    }

    BTL_DISABLE_SSE_EXCEPTIONS();
  }

  BTL_DONT_INLINE static bool skipAction(const std::string& _name)
  {
    if (Instance.m_selectedActionNames.empty())
      return false;

    BtlString name(_name);
    for (unsigned int i=0; i<Instance.m_selectedActionNames.size(); ++i)
      if (name.contains(Instance.m_selectedActionNames[i]))
        return false;

    return true;
  }

  static BtlConfig Instance;
  bool overwriteResults;
  bool checkResults;
  bool realclock;
  int tries;

protected:
  std::vector<BtlString> m_selectedActionNames;
};

#define BTL_MAIN \
  BtlConfig BtlConfig::Instance

#endif // BTL_HH
