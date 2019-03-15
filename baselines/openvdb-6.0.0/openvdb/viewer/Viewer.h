///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#ifndef OPENVDB_VIEWER_VIEWER_HAS_BEEN_INCLUDED
#define OPENVDB_VIEWER_VIEWER_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <string>


namespace openvdb_viewer {

class Viewer;

enum { DEFAULT_WIDTH = 900, DEFAULT_HEIGHT = 800 };


/// @brief Initialize and return a viewer.
/// @param progName      the name of the calling program (for use in info displays)
/// @param background    if true, run the viewer in a separate thread
/// @note Currently, the viewer window is a singleton (but that might change
/// in the future), so although this function returns a new Viewer instance
/// on each call, all instances are associated with the same window.
Viewer init(const std::string& progName, bool background);

/// @brief Destroy all viewer windows and release resources.
/// @details This should be called from the main thread before your program exits.
void exit();


/// Manager for a window that displays OpenVDB grids
class Viewer
{
public:
    /// Set the size of and open the window associated with this viewer.
    void open(int width = DEFAULT_WIDTH, int height = DEFAULT_HEIGHT);

    /// Display the given grids.
    void view(const openvdb::GridCPtrVec&);

    /// @brief Process any pending user input (keyboard, mouse, etc.)
    /// in the window associated with this viewer.
    void handleEvents();

    /// @brief Close the window associated with this viewer.
    /// @warning The window associated with this viewer might be shared with other viewers.
    void close();

    /// Resize the window associated with this viewer.
    void resize(int width, int height);

    /// Return a string with version number information.
    std::string getVersionString() const;

private:
    friend Viewer init(const std::string&, bool);
    Viewer();
};

} // namespace openvdb_viewer

#endif // OPENVDB_VIEWER_VIEWER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
