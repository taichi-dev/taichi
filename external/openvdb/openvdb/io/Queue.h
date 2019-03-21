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

/// @file Queue.h
/// @author Peter Cucka

#ifndef OPENVDB_IO_QUEUE_HAS_BEEN_INCLUDED
#define OPENVDB_IO_QUEUE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <algorithm> // for std::copy
#include <functional>
#include <iterator> // for std::back_inserter
#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

class Archive;

/// @brief Queue for asynchronous output of grids to files or streams
///
/// @warning The queue holds shared pointers to grids.  It is not safe
/// to modify a grid that has been placed in the queue.  Instead,
/// make a deep copy of the grid (Grid::deepCopy()).
///
/// @par Example:
/// @code
/// #include <openvdb/openvdb.h>
/// #include <openvdb/io/Queue.h>
/// #include <tbb/concurrent_hash_map.h>
/// #include <functional>
///
/// using openvdb::io::Queue;
///
/// struct MyNotifier
/// {
///     // Use a concurrent container, because queue callback functions
///     // must be thread-safe.
///     using FilenameMap = tbb::concurrent_hash_map<Queue::Id, std::string>;
///     FilenameMap filenames;
///
///     // Callback function that prints the status of a completed task.
///     void callback(Queue::Id id, Queue::Status status)
///     {
///         const bool ok = (status == Queue::SUCCEEDED);
///         FilenameMap::accessor acc;
///         if (filenames.find(acc, id)) {
///             std::cout << (ok ? "wrote " : "failed to write ")
///                 << acc->second << std::endl;
///             filenames.erase(acc);
///         }
///     }
/// };
///
/// int main()
/// {
///     // Construct an object to receive notifications from the queue.
///     // The object's lifetime must exceed the queue's.
///     MyNotifier notifier;
///
///     Queue queue;
///
///     // Register the callback() method of the MyNotifier object
///     // to receive notifications of completed tasks.
///     queue.addNotifier(std::bind(&MyNotifier::callback, &notifier,
///         std::placeholders::_1, std::placeholders::_2));
///
///     // Queue grids for output (e.g., for each step of a simulation).
///     for (int step = 1; step <= 10; ++step) {
///         openvdb::FloatGrid::Ptr grid = ...;
///
///         std::ostringstream os;
///         os << "mygrid." << step << ".vdb";
///         const std::string filename = os.str();
///
///         Queue::Id id = queue.writeGrid(grid, openvdb::io::File(filename));
///
///         // Associate the filename with the ID of the queued task.
///         MyNotifier::FilenameMap::accessor acc;
///         notifier.filenames.insert(acc, id);
///         acc->second = filename;
///     }
/// }
/// @endcode
/// Output:
/// @code
/// wrote mygrid.1.vdb
/// wrote mygrid.2.vdb
/// wrote mygrid.4.vdb
/// wrote mygrid.3.vdb
/// ...
/// wrote mygrid.10.vdb
/// @endcode
/// Note that tasks do not necessarily complete in the order in which they were queued.
class OPENVDB_API Queue
{
public:
    /// Default maximum queue length (see setCapacity())
    static const Index32 DEFAULT_CAPACITY = 100;
    /// @brief Default maximum time in seconds to wait to queue a task
    /// when the queue is full (see setTimeout())
    static const Index32 DEFAULT_TIMEOUT = 120; // seconds

    /// ID number of a queued task or of a registered notification callback
    using Id = Index32;

    /// Status of a queued task
    enum Status { UNKNOWN, PENDING, SUCCEEDED, FAILED };


    /// Construct a queue with the given capacity.
    explicit Queue(Index32 capacity = DEFAULT_CAPACITY);
    /// Block until all queued tasks complete (successfully or unsuccessfully).
    ~Queue();

    /// @brief Return @c true if the queue is empty.
    bool empty() const;
    /// @brief Return the number of tasks currently in the queue.
    Index32 size() const;

    /// @brief Return the maximum number of tasks allowed in the queue.
    /// @details Once the queue has reached its maximum size, adding
    /// a new task will block until an existing task has executed.
    Index32 capacity() const;
    /// Set the maximum number of tasks allowed in the queue.
    void setCapacity(Index32);

    /// Return the maximum number of seconds to wait to queue a task when the queue is full.
    Index32 timeout() const;
    /// Set the maximum number of seconds to wait to queue a task when the queue is full.
    void setTimeout(Index32 seconds = DEFAULT_TIMEOUT);

    /// @brief Return the status of the task with the given ID.
    /// @note Querying the status of a task that has already completed
    /// (whether successfully or not) removes the task from the status registry.
    /// Subsequent queries of its status will return UNKNOWN.
    Status status(Id) const;

    using Notifier = std::function<void (Id, Status)>;
    /// @brief Register a function that will be called with a task's ID
    /// and status when that task completes, whether successfully or not.
    /// @return an ID that can be passed to removeNotifier() to deregister the function
    /// @details When multiple notifiers are registered, they are called
    /// in the order in which they were registered.
    /// @warning Notifiers are called from worker threads, so they must be thread-safe
    /// and their lifetimes must exceed that of the queue.  They must also not call,
    /// directly or indirectly, addNotifier(), removeNotifier() or clearNotifiers(),
    /// as that can result in a deadlock.
    Id addNotifier(Notifier);
    /// Deregister the notifier with the given ID.
    void removeNotifier(Id);
    /// Deregister all notifiers.
    void clearNotifiers();

    /// @brief Queue a single grid for output to a file or stream.
    /// @param grid  the grid to be serialized
    /// @param archive  the io::File or io::Stream to which to output the grid
    /// @param fileMetadata  optional file-level metadata
    /// @return an ID with which the status of the queued task can be queried
    /// @throw RuntimeError if the task cannot be queued within the time limit
    /// (see setTimeout()) because the queue is full
    /// @par Example:
    /// @code
    /// openvdb::FloatGrid::Ptr grid = ...;
    ///
    /// openvdb::io::Queue queue;
    ///
    /// // Write the grid to the file mygrid.vdb.
    /// queue.writeGrid(grid, openvdb::io::File("mygrid.vdb"));
    ///
    /// // Stream the grid to a binary string.
    /// std::ostringstream ostr(std::ios_base::binary);
    /// queue.writeGrid(grid, openvdb::io::Stream(ostr));
    /// @endcode
    Id writeGrid(GridBase::ConstPtr grid, const Archive& archive,
        const MetaMap& fileMetadata = MetaMap());

    /// @brief Queue a container of grids for output to a file.
    /// @param grids  any iterable container of grid pointers
    ///     (e.g., a GridPtrVec or GridPtrSet)
    /// @param archive  the io::File or io::Stream to which to output the grids
    /// @param fileMetadata  optional file-level metadata
    /// @return an ID with which the status of the queued task can be queried
    /// @throw RuntimeError if the task cannot be queued within the time limit
    /// (see setTimeout()) because the queue is full
    /// @par Example:
    /// @code
    /// openvdb::FloatGrid::Ptr floatGrid = ...;
    /// openvdb::BoolGrid::Ptr boolGrid = ...;
    /// openvdb::GridPtrVec grids;
    /// grids.push_back(floatGrid);
    /// grids.push_back(boolGrid);
    ///
    /// openvdb::io::Queue queue;
    ///
    /// // Write the grids to the file mygrid.vdb.
    /// queue.write(grids, openvdb::io::File("mygrid.vdb"));
    ///
    /// // Stream the grids to a (binary) string.
    /// std::ostringstream ostr(std::ios_base::binary);
    /// queue.write(grids, openvdb::io::Stream(ostr));
    /// @endcode
    template<typename GridPtrContainer>
    Id write(const GridPtrContainer& grids, const Archive& archive,
        const MetaMap& fileMetadata = MetaMap());

private:
    // Disallow copying of instances of this class.
    Queue(const Queue&);
    Queue& operator=(const Queue&);

    Id writeGridVec(const GridCPtrVec&, const Archive&, const MetaMap&);

    struct Impl;
    std::unique_ptr<Impl> mImpl;
}; // class Queue


template<typename GridPtrContainer>
inline Queue::Id
Queue::write(const GridPtrContainer& container,
    const Archive& archive, const MetaMap& metadata)
{
    GridCPtrVec grids;
    std::copy(container.begin(), container.end(), std::back_inserter(grids));
    return this->writeGridVec(grids, archive, metadata);
}

// Specialization for vectors of const Grid pointers; no copying necessary
template<>
inline Queue::Id
Queue::write<GridCPtrVec>(const GridCPtrVec& grids,
    const Archive& archive, const MetaMap& metadata)
{
    return this->writeGridVec(grids, archive, metadata);
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_QUEUE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
