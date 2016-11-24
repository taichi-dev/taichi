/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2012, assimp team
All rights reserved.

Redistribution and use of this software in source and binary forms, 
with or without modification, are permitted provided that the 
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file ProgressHandler.h
 *  @brief Abstract base class 'ProgressHandler'.
 */
#ifndef INCLUDED_AI_PROGRESSHANDLER_H
#define INCLUDED_AI_PROGRESSHANDLER_H
#include "types.h"
namespace Assimp	{

// ------------------------------------------------------------------------------------
/** @brief CPP-API: Abstract interface for custom progress report receivers.
 *
 *  Each #Importer instance maintains its own #ProgressHandler. The default 
 *  implementation provided by Assimp doesn't do anything at all. */
class ASSIMP_API ProgressHandler
#ifndef SWIG
	: public Intern::AllocateFromAssimpHeap
#endif
{
protected:
	/** @brief	Default constructor	*/
	ProgressHandler () {
	}
public:
	/** @brief	Virtual destructor	*/
	virtual ~ProgressHandler () {
	}

	// -------------------------------------------------------------------
	/** @brief Progress callback.
	 *  @param percentage An estimate of the current loading progress,
	 *    in percent. Or -1.f if such an estimate is not available.
	 *
	 *  There are restriction on what you may do from within your 
	 *  implementation of this method: no exceptions may be thrown and no
	 *  non-const #Importer methods may be called. It is 
	 *  not generally possible to predict the number of callbacks 
	 *  fired during a single import.
	 *
	 *  @return Return false to abort loading at the next possible
	 *   occasion (loaders and Assimp are generally allowed to perform
	 *   all needed cleanup tasks prior to returning control to the
	 *   caller). If the loading is aborted, #Importer::ReadFile()
	 *   returns always NULL.
	 *
	 *  @note Currently, percentage is always -1.f because there is 
	 *   no reliable way to compute it.
	 *   */
	virtual bool Update(float percentage = -1.f) = 0;



}; // !class ProgressHandler 
// ------------------------------------------------------------------------------------
} // Namespace Assimp

#endif
