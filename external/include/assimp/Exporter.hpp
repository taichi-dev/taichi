/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2011, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms, 
with or without modification, are permitted provided that the following 
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file  export.hpp
*  @brief Defines the CPP-API for the Assimp export interface
*/
#ifndef AI_EXPORT_HPP_INC
#define AI_EXPORT_HPP_INC

#ifndef ASSIMP_BUILD_NO_EXPORT

#include "cexport.h"

namespace Assimp	{
	class ExporterPimpl;
	class IOSystem;


// ----------------------------------------------------------------------------------
/** CPP-API: The Exporter class forms an C++ interface to the export functionality 
 * of the Open Asset Import Library. Note that the export interface is available
 * only if Assimp has been built with ASSIMP_BUILD_NO_EXPORT not defined.
 * 
 * The interface is modelled after the importer interface and mostly
 * symmetric. The same rules for threading etc. apply.
 *
 * In a nutshell, there are two export interfaces: #Export, which writes the
 * output file(s) either to the regular file system or to a user-supplied 
 * #IOSystem, and #ExportToBlob which returns a linked list of memory
 * buffers (blob), each referring to one output file (in most cases
 * there will be only one output file of course, but this extra complexity is
 * needed since Assimp aims at supporting a wide range of file formats). 
 * 
 * #ExportToBlob is especially useful if you intend to work 
 * with the data in-memory. 
*/
class ASSIMP_API Exporter
	// TODO: causes good ol' base class has no dll interface warning
//#ifdef __cplusplus
//	: public boost::noncopyable
//#endif // __cplusplus
{
public:

	/** Function pointer type of a Export worker function */
	typedef void (*fpExportFunc)(const char*,IOSystem*,const aiScene*);

	/** Internal description of an Assimp export format option */
	struct ExportFormatEntry
	{
		/// Public description structure to be returned by aiGetExportFormatDescription()
		aiExportFormatDesc mDescription;

		// Worker function to do the actual exporting
		fpExportFunc mExportFunction;

		// Postprocessing steps to be executed PRIOR to invoking mExportFunction
		unsigned int mEnforcePP;

		// Constructor to fill all entries
		ExportFormatEntry( const char* pId, const char* pDesc, const char* pExtension, fpExportFunc pFunction, unsigned int pEnforcePP = 0u)
		{
			mDescription.id = pId;
			mDescription.description = pDesc;
			mDescription.fileExtension = pExtension;
			mExportFunction = pFunction;
			mEnforcePP = pEnforcePP;
		}

		ExportFormatEntry() : mExportFunction(), mEnforcePP() {}
	};


public:

	
	Exporter();
	~Exporter();

public:


	// -------------------------------------------------------------------
	/** Supplies a custom IO handler to the exporter to use to open and
	 * access files. 
	 * 
	 * If you need #Export to use custom IO logic to access the files, 
	 * you need to supply a custom implementation of IOSystem and 
	 * IOFile to the exporter. 
	 *
	 * #Exporter takes ownership of the object and will destroy it 
	 * afterwards. The previously assigned handler will be deleted.
	 * Pass NULL to take again ownership of your IOSystem and reset Assimp
	 * to use its default implementation, which uses plain file IO.
	 *
	 * @param pIOHandler The IO handler to be used in all file accesses 
	 *   of the Importer. */
	void SetIOHandler( IOSystem* pIOHandler);

	// -------------------------------------------------------------------
	/** Retrieves the IO handler that is currently set.
	 * You can use #IsDefaultIOHandler() to check whether the returned
	 * interface is the default IO handler provided by ASSIMP. The default
	 * handler is active as long the application doesn't supply its own
	 * custom IO handler via #SetIOHandler().
	 * @return A valid IOSystem interface, never NULL. */
	IOSystem* GetIOHandler() const;

	// -------------------------------------------------------------------
	/** Checks whether a default IO handler is active 
	 * A default handler is active as long the application doesn't 
	 * supply its own custom IO handler via #SetIOHandler().
	 * @return true by default */
	bool IsDefaultIOHandler() const;



	// -------------------------------------------------------------------
	/** Exports the given scene to a chosen file format. Returns the exported 
	* data as a binary blob which you can write into a file or something.
	* When you're done with the data, simply let the #Exporter instance go 
	* out of scope to have it released automatically.
	* @param pScene The scene to export. Stays in possession of the caller,
	*   is not changed by the function.
	* @param pFormatId ID string to specify to which format you want to 
	*   export to. Use 
	* #GetExportFormatCount / #GetExportFormatDescription to learn which 
	*   export formats are available.
	* @param pPreprocessing See the documentation for #Export
	* @return the exported data or NULL in case of error.
	* @note If the Exporter instance did already hold a blob from
	*   a previous call to #ExportToBlob, it will be disposed. 
	*   Any IO handlers set via #SetIOHandler are ignored here.
	* @note Use aiCopyScene() to get a modifiable copy of a previously
	*   imported scene. */
	const aiExportDataBlob* ExportToBlob(  const aiScene* pScene, const char* pFormatId, unsigned int pPreprocessing = 0u );
	inline const aiExportDataBlob* ExportToBlob(  const aiScene* pScene, const std::string& pFormatId, unsigned int pPreprocessing = 0u );


	// -------------------------------------------------------------------
	/** Convenience function to export directly to a file. Use
	 *  #SetIOSystem to supply a custom IOSystem to gain fine-grained control
	 *  about the output data flow of the export process.
	 * @param pBlob A data blob obtained from a previous call to #aiExportScene. Must not be NULL.
	 * @param pPath Full target file name. Target must be accessible.
	 * @param pPreprocessing Accepts any choice of the #aiPostProcessing enumerated
	 *   flags, but in reality only a subset of them makes sense here. Specifying
	 *   'preprocessing' flags is useful if the input scene does not conform to 
	 *   Assimp's default conventions as specified in the @link data Data Structures Page @endlink. 
	 *   In short, this means the geometry data should use a right-handed coordinate systems, face 
	 *   winding should be counter-clockwise and the UV coordinate origin is assumed to be in
	 *   the upper left. The #aiProcess_MakeLeftHanded, #aiProcess_FlipUVs and 
	 *   #aiProcess_FlipWindingOrder flags are used in the import side to allow users
	 *   to have those defaults automatically adapted to their conventions. Specifying those flags
	 *   for exporting has the opposite effect, respectively. Some other of the
	 *   #aiPostProcessSteps enumerated values may be useful as well, but you'll need
	 *   to try out what their effect on the exported file is. Many formats impose
	 *   their own restrictions on the structure of the geometry stored therein,
	 *   so some preprocessing may have little or no effect at all, or may be
	 *   redundant as exporters would apply them anyhow. A good example 
	 *   is triangulation - whilst you can enforce it by specifying
	 *   the #aiProcess_Triangulate flag, most export formats support only
	 *   triangulate data so they would run the step even if it wasn't requested.
	 *
	 *   If assimp detects that the input scene was directly taken from the importer side of 
     *   the library (i.e. not copied using aiCopyScene and potetially modified afterwards), 
     *   any postprocessing steps already applied to the scene will not be applied again, unless
     *   they show non-idempotent behaviour (#aiProcess_MakeLeftHanded, #aiProcess_FlipUVs and 
     *   #aiProcess_FlipWindingOrder).
	 * @return AI_SUCCESS if everything was fine. 
	 * @note Use aiCopyScene() to get a modifiable copy of a previously
	 *   imported scene.*/
	aiReturn Export( const aiScene* pScene, const char* pFormatId, const char* pPath, unsigned int pPreprocessing = 0u);
	inline aiReturn Export( const aiScene* pScene, const std::string& pFormatId, const std::string& pPath,  unsigned int pPreprocessing = 0u);


	// -------------------------------------------------------------------
	/** Returns an error description of an error that occurred in #Export
	 *    or #ExportToBlob
	 *
	 * Returns an empty string if no error occurred.
	 * @return A description of the last error, an empty string if no 
	 *   error occurred. The string is never NULL.
	 *
	 * @note The returned function remains valid until one of the 
	 * following methods is called: #Export, #ExportToBlob, #FreeBlob */
	const char* GetErrorString() const;


	// -------------------------------------------------------------------
	/** Return the blob obtained from the last call to #ExportToBlob */
	const aiExportDataBlob* GetBlob() const;


	// -------------------------------------------------------------------
	/** Orphan the blob from the last call to #ExportToBlob. This means
	 *  the caller takes ownership and is thus responsible for calling
	 *  the C API function #aiReleaseExportBlob to release it. */
	const aiExportDataBlob* GetOrphanedBlob() const;


	// -------------------------------------------------------------------
	/** Frees the current blob.
	 *
	 *  The function does nothing if no blob has previously been 
	 *  previously produced via #ExportToBlob. #FreeBlob is called
	 *  automatically by the destructor. The only reason to call
	 *  it manually would be to reclain as much storage as possible
	 *  without giving up the #Exporter instance yet. */
	void FreeBlob( );


	// -------------------------------------------------------------------
	/** Returns the number of export file formats available in the current
	 *  Assimp build. Use #Exporter::GetExportFormatDescription to
	 *  retrieve infos of a specific export format */
	size_t GetExportFormatCount() const;


	// -------------------------------------------------------------------
	/** Returns a description of the nth export file format. Use #
	 *  #Exporter::GetExportFormatCount to learn how many export 
	 *  formats are supported. 
	 * @param pIndex Index of the export format to retrieve information 
	 *  for. Valid range is 0 to #Exporter::GetExportFormatCount
	 * @return A description of that specific export format. 
	 *  NULL if pIndex is out of range. */
	const aiExportFormatDesc* GetExportFormatDescription( size_t pIndex ) const;


	// -------------------------------------------------------------------
	/** Register a custom exporter. Custom export formats are limited to
	 *    to the current #Exporter instance and do not affect the
	 *    library globally.
	 *  @param desc Exporter description.
	 *  @return aiReturn_SUCCESS if the export format was successfully
	 *    registered. A common cause that would prevent an exporter
	 *    from being registered is that its format id is already
	 *    occupied by another format. */
	aiReturn RegisterExporter(const ExportFormatEntry& desc);


	// -------------------------------------------------------------------
	/** Remove an export format previously registered with #RegisterExporter
	 *  from the #Exporter instance (this can also be used to drop
	 *  builtin exporters because those are implicitly registered
	 *  using #RegisterExporter).
	 *  @param id Format id to be unregistered, this refers to the
	 *    'id' field of #aiExportFormatDesc. 
	 *  @note Calling this method on a format description not yet registered
	 *    has no effect.*/
	void UnregisterExporter(const char* id);


protected:

	// Just because we don't want you to know how we're hacking around.
	ExporterPimpl* pimpl;
};


// ----------------------------------------------------------------------------------
inline const aiExportDataBlob* Exporter :: ExportToBlob(  const aiScene* pScene, const std::string& pFormatId,unsigned int pPreprocessing ) 
{
	return ExportToBlob(pScene,pFormatId.c_str(),pPreprocessing);
}

// ----------------------------------------------------------------------------------
inline aiReturn Exporter :: Export( const aiScene* pScene, const std::string& pFormatId, const std::string& pPath, unsigned int pPreprocessing )
{
	return Export(pScene,pFormatId.c_str(),pPath.c_str(),pPreprocessing);
}

} // namespace Assimp
#endif // ASSIMP_BUILD_NO_EXPORT
#endif // AI_EXPORT_HPP_INC

