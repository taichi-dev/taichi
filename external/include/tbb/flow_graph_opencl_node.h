/*
    Copyright 2005-2015 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks. Threading Building Blocks is free software;
    you can redistribute it and/or modify it under the terms of the GNU General Public License
    version 2  as  published  by  the  Free Software Foundation.  Threading Building Blocks is
    distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See  the GNU General Public License for more details.   You should have received a copy of
    the  GNU General Public License along with Threading Building Blocks; if not, write to the
    Free Software Foundation, Inc.,  51 Franklin St,  Fifth Floor,  Boston,  MA 02110-1301 USA

    As a special exception,  you may use this file  as part of a free software library without
    restriction.  Specifically,  if other files instantiate templates  or use macros or inline
    functions from this file, or you compile this file and link it with other files to produce
    an executable,  this file does not by itself cause the resulting executable to be covered
    by the GNU General Public License. This exception does not however invalidate any other
    reasons why the executable file might be covered by the GNU General Public License.
*/

#ifndef __TBB_flow_graph_opencl_node_H
#define __TBB_flow_graph_opencl_node_H

#include "tbb/tbb_config.h"
#if __TBB_PREVIEW_OPENCL_NODE

#include "flow_graph.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <array>
#include <mutex>
#include <unordered_map>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace tbb {
namespace flow {

namespace interface8 {

class opencl_foundation;
class opencl_device_list;

template <typename Factory>
class opencl_buffer_impl;

class default_opencl_factory;

class opencl_graph : public graph {
public:
    //! Constructs a graph with isolated task_group_context
    opencl_graph() : my_opencl_foundation( NULL ) {}
    //! Constructs a graph with an user context
    explicit opencl_graph( task_group_context& context ) : graph( context ), my_opencl_foundation( NULL ) {}
    //! Destroys a graph
    ~opencl_graph();
    //! Available devices
    const opencl_device_list& available_devices();
    default_opencl_factory& opencl_factory();
protected:
    opencl_foundation *my_opencl_foundation;
    opencl_foundation &get_opencl_foundation();

    template <typename T, typename Factory>
    friend class opencl_buffer;
    template <cl_channel_order channel_order, cl_channel_type channel_type, typename Factory>
    friend class opencl_image2d;
    template<typename... Args>
    friend class opencl_node;
    template <typename DeviceFilter>
    friend class opencl_factory;
};

template <typename T, typename Factory>
class dependency_msg;

template< typename T, typename Factory >
class proxy_dependency_receiver;

template< typename T, typename Factory >
class receiver<dependency_msg<T, Factory>> {
public:
    //! The predecessor type for this node
    typedef sender<dependency_msg<T, Factory>> dependency_predecessor_type;
    typedef proxy_dependency_receiver<T, Factory> proxy;

    receiver() : my_ordinary_receiver( *this ) {}

    //! Put an item to the receiver
    bool try_put( const T& t ) {
        return my_ordinary_receiver.try_put(t);
    }

    //! Put an item to the receiver
    virtual task *try_put_task( const dependency_msg<T, Factory>& ) = 0;

    //! Add a predecessor to the node
    virtual bool register_predecessor( dependency_predecessor_type & ) { return false; }

    //! Remove a predecessor from the node
    virtual bool remove_predecessor( dependency_predecessor_type & ) { return false; }

protected:
    //! put receiver back in initial state
    virtual void reset_receiver( reset_flags f = rf_reset_protocol ) = 0;
    virtual bool is_continue_receiver() { return false; }
private:
    class ordinary_receiver : public receiver < T >, tbb::internal::no_assign {
        //! The predecessor type for this node
        typedef sender<T> predecessor_type;
    public:
        ordinary_receiver(receiver<dependency_msg<T, Factory>>& owner) : my_owner(owner) {}

        //! Put an item to the receiver
        /* override */ task *try_put_task( const T& t ) {
            return my_owner.try_put_task( dependency_msg<T, Factory>( t ) );
        }

        //! Add a predecessor to the node
        /* override */ bool register_predecessor( predecessor_type &p ) {
            tbb::spin_mutex::scoped_lock lock( my_predecessor_map_mutex );
            typename predecessor_map_type::iterator it = my_predecessor_map.emplace( std::piecewise_construct_t(), std::make_tuple( &p ), std::tie( p ) );
            if ( !my_owner.register_predecessor( it->second ) ) {
                my_predecessor_map.erase( it );
                return false;
            }
            return true;
        }

        //! Remove a predecessor from the node
        /* override */ bool remove_predecessor( predecessor_type &p ) {
            tbb::spin_mutex::scoped_lock lock( my_predecessor_map_mutex );
            typename predecessor_map_type::iterator it = my_predecessor_map.find( &p );
            __TBB_ASSERT( it != my_predecessor_map.end(), "Failed to find the predecessor" );
            if ( !my_owner.remove_predecessor( it->second ) )
                return false;
            my_predecessor_map.erase( it );
            return true;
        }

    protected:
        //! put receiver back in initial state
        /* override */ void reset_receiver( reset_flags f = rf_reset_protocol ) {
            my_owner.reset_receiver( f );
        };
        /* override */ bool is_continue_receiver() {
            return my_owner.is_continue_receiver();
        }

    private:
        receiver<dependency_msg<T, Factory>>& my_owner;

        typedef std::multimap<predecessor_type*, typename dependency_predecessor_type::proxy> predecessor_map_type;
        predecessor_map_type my_predecessor_map;
        tbb::spin_mutex my_predecessor_map_mutex;
    };
    ordinary_receiver my_ordinary_receiver;
public:
    ordinary_receiver& ordinary_receiver() { return my_ordinary_receiver; }
};

template< typename T, typename Factory >
class proxy_dependency_sender;

template< typename T, typename Factory >
class proxy_dependency_receiver : public receiver < dependency_msg<T, Factory> >, tbb::internal::no_assign {
public:
    typedef sender<dependency_msg<T, Factory>> dependency_predecessor_type;

    proxy_dependency_receiver( receiver<T>& r ) : my_r( r ) {}

    //! Put an item to the receiver
    /* override */ task *try_put_task( const dependency_msg<T, Factory> &d ) {
        receive_if_memory_object( d );
        receiver<T> *r = &my_r;
        d.register_callback( [r]( const T& t ) {
            r->try_put( t );
        } );
        d.clear_event();
        return SUCCESSFULLY_ENQUEUED;
    }

    //! Add a predecessor to the node
    /* override */ bool register_predecessor( dependency_predecessor_type &s ) {
        return my_r.register_predecessor( s.ordinary_sender() );
    }
    //! Remove a predecessor from the node
    /* override */ bool remove_predecessor( dependency_predecessor_type &s ) {
        return my_r.remove_predecessor( s.ordinary_sender() );
    }
protected:
    //! put receiver back in initial state
    /* override */ void reset_receiver( reset_flags f = rf_reset_protocol ) {
        my_r.reset_receiver( f );
    };

    /* override */ bool is_continue_receiver() {
        return my_r.is_continue_receiver();
    }
private:
    receiver<T> &my_r;
};

template< typename T, typename Factory >
class sender<dependency_msg<T, Factory>> {
public:
    sender() : my_ordinary_sender( *this ) {}

    //! The successor type for this sender
    typedef receiver<dependency_msg<T, Factory>> dependency_successor_type;
    typedef proxy_dependency_sender<T, Factory> proxy;

    //! Add a new successor to this node
    virtual bool register_successor( dependency_successor_type &r ) = 0;

    //! Removes a successor from this node
    virtual bool remove_successor( dependency_successor_type &r ) = 0;

    //! Request an item from the sender
    virtual bool try_get( dependency_msg<T, Factory> & ) { return false; }

    //! Reserves an item in the sender
    virtual bool try_reserve( dependency_msg<T, Factory> & ) { return false; }
private:
    class ordinary_sender : public sender < T >, tbb::internal::no_assign {
        //! The successor type for this sender
        typedef receiver<T> successor_type;
    public:
        ordinary_sender(sender<dependency_msg<T, Factory>>& owner) : my_owner(owner) {}

        //! Add a new successor to this node
        /* override */ bool register_successor( successor_type &r ) {
            tbb::spin_mutex::scoped_lock lock( my_successor_map_mutex );
            typename successor_map_type::iterator it = my_successor_map.emplace( std::piecewise_construct_t(), std::make_tuple( &r ), std::tie( r ) );
            if ( !my_owner.register_successor( it->second ) ) {
                my_successor_map.erase( it );
                return false;
            }
            return true;
        }

        //! Removes a successor from this node
        /* override */ bool remove_successor( successor_type &r ) {
            tbb::spin_mutex::scoped_lock lock( my_successor_map_mutex );
            typename successor_map_type::iterator it = my_successor_map.find( &r );
            __TBB_ASSERT( it != my_successor_map.end(), "The predecessor has already been registered" );
            if ( !my_owner.remove_successor( it->second ) )
                return false;
            my_successor_map.erase( it );
            return true;
        }

        //! Request an item from the sender
        /* override */ bool try_get( T &t ) {
            dependency_msg<T, Factory> d;
            if ( my_owner.try_get( d ) ) {
                t = d.data();
                return true;
            }
            return false;
        }

        /* override */ bool try_reserve( T &t ) {
            dependency_msg<T, Factory> d;
            if ( my_owner.try_reserve( d ) ) {
                t = d.data();
                return true;
            }
            return false;
        }

        bool has_host_successors() {
            tbb::spin_mutex::scoped_lock lock( my_successor_map_mutex );
            return !my_successor_map.empty();
        }
    private:
        sender<dependency_msg<T, Factory>>& my_owner;

        typedef std::multimap<successor_type*, typename dependency_successor_type::proxy> successor_map_type;
        successor_map_type my_successor_map;
        tbb::spin_mutex my_successor_map_mutex;
    };
    ordinary_sender my_ordinary_sender;
public:
    ordinary_sender& ordinary_sender() { return my_ordinary_sender; }

    bool has_host_successors() {
        return my_ordinary_sender.has_host_successors();
    }
};

template< typename T, typename Factory >
class proxy_dependency_sender : public sender < dependency_msg<T, Factory> >, tbb::internal::no_assign {
public:
    typedef receiver<dependency_msg<T, Factory>> dependency_successor_type;

    proxy_dependency_sender( sender<T>& s ) : my_s( s ) {}

    //! Add a new successor to this node
    /* override */ bool register_successor( dependency_successor_type &r ) {
        return my_s.register_successor( r.ordinary_receiver() );
    }

    //! Removes a successor from this node
    /* override */ bool remove_successor( dependency_successor_type &r ) {
        return my_s.remove_successor( r.ordinary_receiver() );
    }

    //! Request an item from the sender
    /* override */ bool try_get( dependency_msg<T, Factory> &d ) {
        return my_s.try_get( d.data() );
    }

    //! Reserves an item in the sender
    /* override */ bool try_reserve( dependency_msg<T, Factory> &d ) {
        return my_s.try_reserve( d.data() );
    }

    //! Releases the reserved item
    /* override */ bool try_release() {
        return my_s.try_release();
    }

    //! Consumes the reserved item
    /* override */ bool try_consume() {
        return my_s.try_consume();
    }
private:
    sender<T> &my_s;
};

template<typename T, typename Factory>
inline void make_edge( sender<T> &s, receiver<dependency_msg<T, Factory>> &r ) {
    make_edge( s, r.ordinary_receiver() );
}

template<typename T, typename Factory>
inline void make_edge( sender<dependency_msg<T, Factory>> &s, receiver<T> &r ) {
    make_edge( s.ordinary_sender(), r );
}

template<typename T, typename Factory>
inline void remove_edge( sender<T> &s, receiver<dependency_msg<T, Factory>> &r ) {
    remove_edge( s, r.ordinary_receiver() );
}

template<typename T, typename Factory>
inline void remove_edge( sender<dependency_msg<T, Factory>> &s, receiver<T> &r ) {
    remove_edge( s.ordinary_sender(), r );
}

inline void enforce_cl_retcode( cl_int err, std::string msg ) {
    if ( err != CL_SUCCESS ) {
        std::cerr << msg << std::endl;
        throw msg;
    }
}

template <typename T>
T event_info( cl_event e, cl_event_info i ) {
    T res;
    enforce_cl_retcode( clGetEventInfo( e, i, sizeof( res ), &res, NULL ), "Failed to get OpenCL event information" );
    return res;
}

template <typename T>
T device_info( cl_device_id d, cl_device_info i ) {
    T res;
    enforce_cl_retcode( clGetDeviceInfo( d, i, sizeof( res ), &res, NULL ), "Failed to get OpenCL device information" );
    return res;
}
template <>
std::string device_info<std::string>( cl_device_id d, cl_device_info i ) {
    size_t required;
    enforce_cl_retcode( clGetDeviceInfo( d, i, 0, NULL, &required ), "Failed to get OpenCL device information" );

    char *buff = (char*)alloca( required );
    enforce_cl_retcode( clGetDeviceInfo( d, i, required, buff, NULL ), "Failed to get OpenCL device information" );

    return buff;
}
template <typename T>
T platform_info( cl_platform_id p, cl_platform_info i ) {
    T res;
    enforce_cl_retcode( clGetPlatformInfo( p, i, sizeof( res ), &res, NULL ), "Failed to get OpenCL platform information" );
    return res;
}
template <>
std::string platform_info<std::string>( cl_platform_id p, cl_platform_info  i ) {
    size_t required;
    enforce_cl_retcode( clGetPlatformInfo( p, i, 0, NULL, &required ), "Failed to get OpenCL platform information" );

    char *buff = (char*)alloca( required );
    enforce_cl_retcode( clGetPlatformInfo( p, i, required, buff, NULL ), "Failed to get OpenCL platform information" );

    return buff;
}


class opencl_device {
    typedef size_t device_id_type;
    enum : device_id_type {
        unknown = device_id_type( -2 ),
        host = device_id_type( -1 )
    };
public:
    opencl_device() : my_device_id( unknown ) {}

    std::string platform_profile() const {
        return platform_info<std::string>( platform(), CL_PLATFORM_PROFILE );
    }
    std::string platform_version() const {
        return platform_info<std::string>( platform(), CL_PLATFORM_VERSION );
    }
    std::string platform_name() const {
        return platform_info<std::string>( platform(), CL_PLATFORM_NAME );
    }
    std::string platform_vendor() const {
        return platform_info<std::string>( platform(), CL_PLATFORM_VENDOR );
    }
    std::string platform_extensions() const {
        return platform_info<std::string>( platform(), CL_PLATFORM_EXTENSIONS );
    }

    template <typename T>
    void info( cl_device_info i, T &t ) const {
        t = device_info<T>( my_cl_device_id, i );
    }
    std::string version() const {
        // The version string format: OpenCL<space><major_version.minor_version><space><vendor-specific information>
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_VERSION );
    }
    int major_version() const {
        int major;
        std::sscanf( version().c_str(), "OpenCL %d", &major );
        return major;
    }
    int minor_version() const {
        int major, minor;
        std::sscanf( version().c_str(), "OpenCL %d.%d", &major, &minor );
        return minor;
    }
    bool out_of_order_exec_mode_on_host_present() const {
#if CL_VERSION_2_0
        if ( major_version() >= 2 )
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        else
#endif /* CL_VERSION_2_0 */
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
    }
    bool out_of_order_exec_mode_on_device_present() const {
#if CL_VERSION_2_0
        if ( major_version() >= 2 )
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        else
#endif /* CL_VERSION_2_0 */
            return false;
    }
    std::array<size_t, 3> max_work_item_sizes() const {
        return device_info<std::array<size_t, 3>>( my_cl_device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES );
    }
    size_t max_work_group_size() const {
        return device_info<size_t>( my_cl_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE );
    }
    bool built_in_kernel_available( const std::string& k ) const {
        const std::string semi = ";";
        // Added semicolumns to force an exact match (to avoid a partial match, e.g. "add" is partly matched with "madd").
        return (semi + built_in_kernels() + semi).find( semi + k + semi ) != std::string::npos;
    }
    std::string built_in_kernels() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_BUILT_IN_KERNELS );
    }
    std::string name() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_NAME );
    }
    cl_bool available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_AVAILABLE );
    }
    cl_bool compiler_available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_COMPILER_AVAILABLE );
    }
    cl_bool linker_available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_LINKER_AVAILABLE );
    }
    bool extension_available( const std::string &ext ) const {
        const std::string space = " ";
        // Added space to force an exact match (to avoid a partial match, e.g. "ext" is partly matched with "ext2").
        return (space + extensions() + space).find( space + ext + space ) != std::string::npos;
    }
    std::string extensions() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_EXTENSIONS );
    }

    cl_device_type type() const {
        return device_info<cl_device_type>( my_cl_device_id, CL_DEVICE_TYPE );
    }

    std::string vendor() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_VENDOR );
    }

    cl_uint address_bits() const {
        return device_info<cl_uint>( my_cl_device_id, CL_DEVICE_ADDRESS_BITS );
    }

private:
    opencl_device( cl_device_id d_id ) : my_device_id( unknown ), my_cl_device_id( d_id ) {}

    cl_platform_id platform() const {
        return device_info<cl_platform_id>( my_cl_device_id, CL_DEVICE_PLATFORM );
    }

    device_id_type my_device_id;
    cl_device_id my_cl_device_id;
    cl_command_queue my_cl_command_queue;

    friend bool operator==(opencl_device d1, opencl_device d2) { return d1.my_cl_device_id == d2.my_cl_device_id; }

    template <typename DeviceFilter>
    friend class opencl_factory;
    template <typename Factory>
    friend class opencl_memory;
    template <typename Factory>
    friend class opencl_program;
    friend class opencl_foundation;

#if TBB_USE_ASSERT
    template <typename T, typename Factory>
    friend class opencl_buffer;
#endif
};

class opencl_device_list {
    typedef std::vector<opencl_device> container_type;
public:
    typedef container_type::iterator iterator;
    typedef container_type::const_iterator const_iterator;
    typedef container_type::size_type size_type;

    opencl_device_list() {}
    opencl_device_list( std::initializer_list<opencl_device> il ) : my_container( il ) {}

    void add( opencl_device d ) { my_container.push_back( d ); }
    size_type size() const { return my_container.size(); }
    iterator begin() { return my_container.begin(); }
    iterator end() { return my_container.end(); }
    const_iterator begin() const { return my_container.begin(); }
    const_iterator end() const { return my_container.end(); }
    const_iterator cbegin() const { return my_container.cbegin(); }
    const_iterator cend() const { return my_container.cend(); }
private:
    container_type my_container;
};

class callback_base : tbb::internal::no_copy {
public:
    virtual void call() const = 0;
    virtual ~callback_base() {}
};

template <typename Callback, typename T>
class callback : public callback_base {
    graph &my_graph;
    Callback my_callback;
    T my_data;
public:
    callback( graph &g, Callback c, const T& t ) : my_graph( g ), my_callback( c ), my_data( t ) {
        // Extend the graph lifetime until the callback completion.
        my_graph.increment_wait_count();
    }
    ~callback() {
        // Release the reference to the graph.
        my_graph.decrement_wait_count();
    }
    /* override */ void call() const {
        my_callback( my_data );
    }
};

template <typename T, typename Factory = default_opencl_factory>
class dependency_msg {
public:
    typedef T value_type;

    dependency_msg() = default;
    explicit dependency_msg( const T& data ) : my_data( data ) {}
    dependency_msg( opencl_graph &g, const T& data ) : my_data( data ), my_graph( &g ) {}
    dependency_msg( const T& data, cl_event event ) : my_data( data ), my_event( event ), my_is_event( true ) {
        enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );
    }
    T& data( bool wait = true ) {
        if ( my_is_event && wait ) {
            enforce_cl_retcode( clWaitForEvents( 1, &my_event ), "Failed to wait for an event" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
            my_is_event = false;
        }
        return my_data;
    }

    const T& data( bool wait = true ) const {
        if ( my_is_event && wait ) {
            enforce_cl_retcode( clWaitForEvents( 1, &my_event ), "Failed to wait for an event" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
            my_is_event = false;
        }
        return my_data;
    }

    dependency_msg( const dependency_msg &dmsg ) : my_data( dmsg.my_data ), my_event( dmsg.my_event ), my_is_event( dmsg.my_is_event ), my_graph( dmsg.my_graph ) {
        if ( my_is_event )
            enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );
    }

    dependency_msg( dependency_msg &&dmsg ) : my_data( std::move(dmsg.my_data) ), my_event( dmsg.my_event ), my_is_event( dmsg.my_is_event ), my_graph( dmsg.my_graph ) {
        dmsg.my_is_event = false;
    }

    dependency_msg& operator=(const dependency_msg &dmsg) {
        my_data = dmsg.my_data;
        my_event = dmsg.my_event;
        my_is_event = dmsg.my_is_event;
        my_graph = dmsg.my_graph;
        if ( my_is_event )
            enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );
        return *this;
    }

    ~dependency_msg() {
        if ( my_is_event )
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
    }

    cl_event const * get_event() const { return my_is_event ? &my_event : NULL; }
    void set_event( cl_event e ) const {
        if ( my_is_event ) {
            cl_command_queue cq = event_info<cl_command_queue>( my_event, CL_EVENT_COMMAND_QUEUE );
            if ( cq != event_info<cl_command_queue>( e, CL_EVENT_COMMAND_QUEUE ) )
                enforce_cl_retcode( clFlush( cq ), "Failed to flush an OpenCL command queue" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
        }
        my_is_event = true;
        my_event = e;
        clRetainEvent( my_event );
    }

    void set_graph( graph &g ) {
        my_graph = &g;
    }

    void clear_event() const {
        if ( my_is_event ) {
            enforce_cl_retcode( clFlush( event_info<cl_command_queue>( my_event, CL_EVENT_COMMAND_QUEUE ) ), "Failed to flush an OpenCL command queue" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
        }
        my_is_event = false;
    }

    template <typename Callback>
    void register_callback( Callback c ) const {
        __TBB_ASSERT( my_is_event, "The OpenCL event is not set" );
        __TBB_ASSERT( my_graph, "The graph is not set" );
        enforce_cl_retcode( clSetEventCallback( my_event, CL_COMPLETE, register_callback_func, new callback<Callback, T>( *my_graph, c, my_data ) ), "Failed to set an OpenCL callback" );
    }

    operator T&() { return data(); }
    operator const T&() const { return data(); }

private:
    static void CL_CALLBACK register_callback_func( cl_event, cl_int event_command_exec_status, void *data ) {
        tbb::internal::suppress_unused_warning( event_command_exec_status );
        __TBB_ASSERT( event_command_exec_status == CL_COMPLETE, NULL );
        __TBB_ASSERT( data, NULL );
        callback_base *c = static_cast<callback_base*>(data);
        c->call();
        delete c;
    }

    T my_data;
    mutable cl_event my_event;
    mutable bool my_is_event = false;
    graph *my_graph = NULL;
};

template <typename K, typename T, typename Factory>
K key_from_message( const dependency_msg<T, Factory> &dmsg ) {
    using tbb::flow::key_from_message;
    const T &t = dmsg.data( false );
    __TBB_STATIC_ASSERT( true, "" );
    return key_from_message<K, T>( t );
}

template <typename Factory>
class opencl_memory {
public:
    opencl_memory() {}
    opencl_memory( Factory &f ) : my_host_ptr( NULL ), my_factory( &f ), my_sending_event_present( false ) {
        my_curr_device_id = my_factory->devices().begin()->my_device_id;
    }

    ~opencl_memory() {
        if ( my_sending_event_present ) enforce_cl_retcode( clReleaseEvent( my_sending_event ), "Failed to release an event for the OpenCL buffer" );
        enforce_cl_retcode( clReleaseMemObject( my_cl_mem ), "Failed to release an memory object" );
    }

    cl_mem get_cl_mem() const {
        return my_cl_mem;
    }

    void* get_host_ptr() {
        if ( !my_host_ptr ) {
            dependency_msg<void*, Factory> d = receive( NULL );
            d.data();
            __TBB_ASSERT( d.data() == my_host_ptr, NULL );
        }
        return my_host_ptr;
    }

    dependency_msg<void*, Factory> send( opencl_device d, const cl_event *e );
    dependency_msg<void*, Factory> receive( const cl_event *e );
    virtual void map_memory( opencl_device, dependency_msg<void*, Factory> & ) = 0;
protected:
    cl_mem my_cl_mem;
    tbb::atomic<opencl_device::device_id_type> my_curr_device_id;
    void* my_host_ptr;
    Factory *my_factory;

    tbb::spin_mutex my_sending_lock;
    bool my_sending_event_present;
    cl_event my_sending_event;
};

template <typename Factory>
class opencl_buffer_impl : public opencl_memory<Factory> {
    size_t my_size;
public:
    opencl_buffer_impl( size_t size, Factory& f ) : opencl_memory<Factory>( f ), my_size( size ) {
        cl_int err;
        this->my_cl_mem = clCreateBuffer( this->my_factory->context(), CL_MEM_ALLOC_HOST_PTR, size, NULL, &err );
        enforce_cl_retcode( err, "Failed to create an OpenCL buffer" );
    }

    size_t size() const {
        return my_size;
    }

    /* override */ void map_memory( opencl_device device, dependency_msg<void*, Factory> &dmsg ) {
        this->my_factory->enque_map_buffer( device, *this, dmsg );
    }

#if TBB_USE_ASSERT
    template <typename, typename>
    friend class opencl_buffer;
#endif
};

enum access_type {
    read_write,
    write_only,
    read_only
};

template <typename T, typename Factory = default_opencl_factory>
class opencl_buffer {
public:
    typedef cl_mem native_object_type;
    typedef opencl_buffer memory_object_type;
    typedef Factory opencl_factory_type;

    template<access_type a> using iterator = T*;

    template <access_type a>
    iterator<a> access() const {
        T* ptr = (T*)my_impl->get_host_ptr();
        __TBB_ASSERT( ptr, NULL );
        return iterator<a>( ptr );
    }

    T* data() const { return &access<read_write>()[0]; }

    template <access_type a = read_write>
    iterator<a> begin() const { return access<a>(); }

    template <access_type a = read_write>
    iterator<a> end() const { return access<a>()+my_impl->size()/sizeof(T); }

    size_t size() const { return my_impl->size()/sizeof(T); }

    T& operator[] ( ptrdiff_t k ) { return begin()[k]; }

    opencl_buffer() {}
    opencl_buffer( opencl_graph &g, size_t size );
    opencl_buffer( Factory &f, size_t size ) : my_impl( std::make_shared<impl_type>( size*sizeof(T), f ) ) {}

    cl_mem native_object() const {
        return my_impl->get_cl_mem();
    }

    const opencl_buffer& memory_object() const {
        return *this;
    }

    void send( opencl_device device, dependency_msg<opencl_buffer, Factory> &dependency ) const {
        __TBB_ASSERT( dependency.data( /*wait = */false ) == *this, NULL );
        dependency_msg<void*, Factory> d = my_impl->send( device, dependency.get_event() );
        const cl_event *e = d.get_event();
        if ( e ) dependency.set_event( *e );
        else dependency.clear_event();
    }
    void receive( const dependency_msg<opencl_buffer, Factory> &dependency ) const {
        __TBB_ASSERT( dependency.data( /*wait = */false ) == *this, NULL );
        dependency_msg<void*, Factory> d = my_impl->receive( dependency.get_event() );
        const cl_event *e = d.get_event();
        if ( e ) dependency.set_event( *e );
        else dependency.clear_event();
    }

private:
    typedef opencl_buffer_impl<Factory> impl_type;

    std::shared_ptr<impl_type> my_impl;

    friend bool operator==(const opencl_buffer<T, Factory> &lhs, const opencl_buffer<T, Factory> &rhs) {
        return lhs.my_impl == rhs.my_impl;
    }

    template <typename>
    friend class opencl_factory;
};


template <typename DeviceFilter>
class opencl_factory {
public:
    opencl_factory( opencl_graph &g ) : my_graph( g ) {}
    ~opencl_factory() {
        if ( my_devices.size() ) {
            for ( opencl_device d : my_devices ) {
                enforce_cl_retcode( clReleaseCommandQueue( d.my_cl_command_queue ), "Failed to release a command queue" );
            }
            enforce_cl_retcode( clReleaseContext( my_cl_context ), "Failed to release a context" );
        }
    }

    bool init( const opencl_device_list &device_list ) {
        tbb::spin_mutex::scoped_lock lock( my_devices_mutex );
        if ( !my_devices.size() ) {
            my_devices = device_list;
            return true;
        }
        return false;
    }


private:
    template <typename Factory>
    void enque_map_buffer( opencl_device device, opencl_buffer_impl<Factory> &buffer, dependency_msg<void*, Factory>& dmsg ) {
        cl_event const* e1 = dmsg.get_event();
        cl_event e2;
        cl_int err;
        void *ptr = clEnqueueMapBuffer( device.my_cl_command_queue, buffer.get_cl_mem(), false, CL_MAP_READ | CL_MAP_WRITE, 0, buffer.size(),
            e1 == NULL ? 0 : 1, e1, &e2, &err );
        enforce_cl_retcode( err, "Failed to map a buffer" );
        dmsg.data( false ) = ptr;
        dmsg.set_event( e2 );
        enforce_cl_retcode( clReleaseEvent( e2 ), "Failed to release an event" );
    }


    template <typename Factory>
    void enque_unmap_buffer( opencl_device device, opencl_memory<Factory> &memory, dependency_msg<void*, Factory>& dmsg ) {
        cl_event const* e1 = dmsg.get_event();
        cl_event e2;
        enforce_cl_retcode(
            clEnqueueUnmapMemObject( device.my_cl_command_queue, memory.get_cl_mem(), memory.get_host_ptr(), e1 == NULL ? 0 : 1, e1, &e2 ),
           "Failed to unmap a buffer" );
        dmsg.set_event( e2 );
        enforce_cl_retcode( clReleaseEvent( e2 ), "Failed to release an event" );
    }

    template <typename GlbNDRange, typename LclNDRange>
    cl_event enqueue_kernel( opencl_device device, cl_kernel kernel,
        GlbNDRange&& global_work_size, LclNDRange&& local_work_size, cl_uint num_events, cl_event* event_list ) {
        auto g_it = global_work_size.begin();
        auto l_it = local_work_size.begin();
        __TBB_ASSERT( g_it != global_work_size.end() , "Empty global work size" );
        __TBB_ASSERT( l_it != local_work_size.end() , "Empty local work size" );
        std::array<size_t, 3> g_size, l_size, g_offset = { { 0, 0, 0 } };
        cl_uint s;
        for ( s = 0; s < 3 && g_it != global_work_size.end() && l_it != local_work_size.end(); ++s ) {
            g_size[s] = *g_it++;
            l_size[s] = *l_it++;
        }
        cl_event event;
        enforce_cl_retcode(
            clEnqueueNDRangeKernel( device.my_cl_command_queue, kernel, s,
                g_offset.data(), g_size.data(), l_size[0] ? l_size.data() : NULL, num_events, num_events ? event_list : NULL, &event ),
            "Failed to enqueue a kernel" );
        return event;
    }

    void flush( opencl_device device ) {
        enforce_cl_retcode( clFlush( device.my_cl_command_queue ), "Failed to flush an OpenCL command queue" );
    }

    const opencl_device_list& devices() {
        std::call_once( my_once_flag, &opencl_factory::init_once, this );
        return my_devices;
    }

    bool is_same_context( opencl_device::device_id_type d1, opencl_device::device_id_type d2 ) {
        __TBB_ASSERT( d1 != opencl_device::unknown && d2 != opencl_device::unknown, NULL );
        // Currently, factory supports only one context so if the both devices are not host it means the are in the same context. 
        if ( d1 != opencl_device::host && d2 != opencl_device::host )
            return true;
        return d1 == d2;
    }

    opencl_factory( const opencl_factory& );
    opencl_factory& operator=(const opencl_factory&);

    cl_context context() {
        std::call_once( my_once_flag, &opencl_factory::init_once, this );
        return my_cl_context;
    }

    void init_once();

    std::once_flag my_once_flag;
    opencl_device_list my_devices;
    cl_context my_cl_context;
    opencl_graph &my_graph;

    tbb::spin_mutex my_devices_mutex;

    template <typename Factory>
    friend class opencl_program;
    template <typename Factory>
    friend class opencl_buffer_impl;
    template <typename Factory>
    friend class opencl_memory;
    template <typename... Args>
    friend class opencl_node;
};

template <typename Factory>
dependency_msg<void*, Factory> opencl_memory<Factory>::receive( const cl_event *e ) {
    dependency_msg<void*, Factory> d = e ? dependency_msg<void*, Factory>( my_host_ptr, *e ) : dependency_msg<void*, Factory>( my_host_ptr );
    // Concurrent receives are prohibited so we do not worry about synchronization.
    if ( my_curr_device_id.load<tbb::relaxed>() != opencl_device::host ) {
        map_memory( *my_factory->devices().begin(), d );
        my_curr_device_id.store<tbb::relaxed>( opencl_device::host );
        my_host_ptr = d.data( false );
    }
    // Release the sending event
    if ( my_sending_event_present ) {
        enforce_cl_retcode( clReleaseEvent( my_sending_event ), "Failed to release an event" );
        my_sending_event_present = false;
    }
    return d;
}

template <typename Factory>
dependency_msg<void*, Factory> opencl_memory<Factory>::send( opencl_device device, const cl_event *e ) {
    opencl_device::device_id_type device_id = device.my_device_id;
    if ( !my_factory->is_same_context( my_curr_device_id.load<tbb::acquire>(), device_id ) ) {
        __TBB_ASSERT( !e, "The buffer has come from another opencl_node but it is not on a device" );
        {
            tbb::spin_mutex::scoped_lock lock( my_sending_lock );
            if ( !my_factory->is_same_context( my_curr_device_id.load<tbb::relaxed>(), device_id ) ) {
                __TBB_ASSERT( my_host_ptr, "The buffer has not been mapped" );
                dependency_msg<void*, Factory> d( my_host_ptr );
                my_factory->enque_unmap_buffer( device, *this, d );
                my_sending_event = *d.get_event();
                my_sending_event_present = true;
                enforce_cl_retcode( clRetainEvent( my_sending_event ), "Failed to retain an event" );
                my_host_ptr = NULL;
                my_curr_device_id.store<tbb::release>(device_id);
            }
        }
        __TBB_ASSERT( my_sending_event_present, NULL );
    }

    // !e means that buffer has come from the host
    if ( !e && my_sending_event_present ) e = &my_sending_event;

    __TBB_ASSERT( !my_host_ptr, "The buffer has not been unmapped" );
    return e ? dependency_msg<void*, Factory>( NULL, *e ) : dependency_msg<void*, Factory>( NULL );
}

struct default_opencl_factory_device_filter {
    opencl_device_list operator()( const opencl_device_list &devices ) {
        opencl_device_list dl;
        dl.add( *devices.begin() );
        return dl;
    }
};

class default_opencl_factory : public opencl_factory < default_opencl_factory_device_filter > {
public:
    default_opencl_factory( opencl_graph &g ) : opencl_factory( g ) {}
private:
    default_opencl_factory( const default_opencl_factory& );
    default_opencl_factory& operator=(const default_opencl_factory&);
};

class opencl_foundation : tbb::internal::no_assign {
    struct default_device_selector_type {
        opencl_device operator()( const opencl_device_list& devices ) { 
            return *devices.begin();
        }
    };
public:
    opencl_foundation( opencl_graph &g ) : my_default_opencl_factory( g ), my_default_device_selector() {
        cl_uint num_platforms;
        enforce_cl_retcode( clGetPlatformIDs( 0, NULL, &num_platforms ), "clGetPlatformIDs failed" );

        std::vector<cl_platform_id> platforms( num_platforms );
        enforce_cl_retcode( clGetPlatformIDs( num_platforms, platforms.data(), NULL ), "clGetPlatformIDs failed" );

        cl_uint num_all_devices = 0;
        for ( cl_platform_id p : platforms ) {
            cl_uint num_devices;
            enforce_cl_retcode( clGetDeviceIDs( p, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices ), "clGetDeviceIDs failed" );
            num_all_devices += num_devices;
        }

        std::vector<cl_device_id> devices( num_all_devices );
        std::vector<cl_device_id>::iterator it = devices.begin();
        for ( cl_platform_id p : platforms ) {
            cl_uint num_devices;
            enforce_cl_retcode( clGetDeviceIDs( p, CL_DEVICE_TYPE_ALL, (cl_uint)std::distance( it, devices.end() ), &*it, &num_devices ), "clGetDeviceIDs failed" );
            it += num_devices;
        }

        for ( cl_device_id d : devices ) my_devices.add( opencl_device( d ) );
    }

    default_opencl_factory &get_default_opencl_factory() {
        return my_default_opencl_factory;
    }

    const opencl_device_list &get_all_devices() {
        return my_devices;
    }

    default_device_selector_type get_default_device_selector() { return my_default_device_selector; }

private:
    default_opencl_factory my_default_opencl_factory;
    opencl_device_list my_devices;

    const default_device_selector_type my_default_device_selector;
};

opencl_foundation &opencl_graph::get_opencl_foundation() {
    opencl_foundation* INITIALIZATION = (opencl_foundation*)1;
    if ( my_opencl_foundation <= INITIALIZATION ) {
        if ( tbb::internal::as_atomic( my_opencl_foundation ).compare_and_swap( INITIALIZATION, NULL ) == 0 ) {
            my_opencl_foundation = new opencl_foundation( *this );
        }
        else {
            tbb::internal::spin_wait_while_eq( my_opencl_foundation, INITIALIZATION );
        }
    }

    __TBB_ASSERT( my_opencl_foundation > INITIALIZATION, "opencl_foundation is not initialized");
    return *my_opencl_foundation;
}

opencl_graph::~opencl_graph() {
    if ( my_opencl_foundation )
        delete my_opencl_foundation;
}

template <typename DeviceFilter>
void opencl_factory<DeviceFilter>::init_once() {
        {
            tbb::spin_mutex::scoped_lock lock( my_devices_mutex );
            if ( !my_devices.size() )
                my_devices = DeviceFilter()(my_graph.get_opencl_foundation().get_all_devices());
        }

    enforce_cl_retcode( my_devices.size() ? CL_SUCCESS : CL_INVALID_DEVICE, "No devices in the device list" );
    cl_platform_id platform_id = my_devices.begin()->platform();
    for ( opencl_device_list::iterator it = ++my_devices.begin(); it != my_devices.end(); ++it )
        enforce_cl_retcode( it->platform() == platform_id ? CL_SUCCESS : CL_INVALID_PLATFORM, "All devices should be in the same platform" );

    std::vector<cl_device_id> cl_device_ids;
    for ( opencl_device d : my_devices ) cl_device_ids.push_back( d.my_cl_device_id );

    cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, (cl_context_properties)NULL };
    cl_int err;
    cl_context ctx = clCreateContext( context_properties,
        (cl_uint)cl_device_ids.size(),
        cl_device_ids.data(),
        NULL, NULL, &err );
    enforce_cl_retcode( err, "Failed to create context" );
    my_cl_context = ctx;

    size_t device_counter = 0;
    for ( opencl_device &d : my_devices ) {
        d.my_device_id = device_counter++;
        cl_int err2;
        cl_command_queue cq;
#if CL_VERSION_2_0
        if ( d.major_version() >= 2 ) {
            if ( d.out_of_order_exec_mode_on_host_present() ) {
                cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 };
                cq = clCreateCommandQueueWithProperties( ctx, d.my_cl_device_id, props, &err2 );
            } else {
                cl_queue_properties props[] = { 0 };
                cq = clCreateCommandQueueWithProperties( ctx, d.my_cl_device_id, props, &err2 );
            }
        } else
#endif
        {
            cl_command_queue_properties props = d.out_of_order_exec_mode_on_host_present() ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
            // Suppress "declared deprecated" warning for the next line.
#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if _MSC_VER || __INTEL_COMPILER
#pragma warning( push )
#if __INTEL_COMPILER
#pragma warning (disable: 1478)
#else
#pragma warning (disable: 4996)
#endif
#endif
            cq = clCreateCommandQueue( ctx, d.my_cl_device_id, props, &err2 );
#if _MSC_VER || __INTEL_COMPILER
#pragma warning( pop )
#endif
#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic pop
#endif
        }
        enforce_cl_retcode( err2, "Failed to create command queue" );
        d.my_cl_command_queue = cq;
    }
}

const opencl_device_list &opencl_graph::available_devices() {
    return get_opencl_foundation().get_all_devices();
}

default_opencl_factory &opencl_graph::opencl_factory() {
    return get_opencl_foundation().get_default_opencl_factory();
}

template <typename T, typename Factory>
opencl_buffer<T, Factory>::opencl_buffer( opencl_graph &g, size_t size ) : my_impl( std::make_shared<impl_type>( size*sizeof(T), g.get_opencl_foundation().get_default_opencl_factory() ) ) {}

    
enum class opencl_program_type {
    SOURCE,
    PRECOMPILED,
    SPIR
};

template <typename Factory = default_opencl_factory>
class opencl_program : tbb::internal::no_assign {
public:
    opencl_program( opencl_program_type type, const std::string& program_name ) : my_type(type) , my_arg_str( program_name) {}
    opencl_program( const char* program_name ) : opencl_program( std::string( program_name ) ) {}
    opencl_program( const std::string& program_name ) : opencl_program( opencl_program_type::SOURCE, program_name ) {}

    opencl_program( const opencl_program &src ) : my_type( src.type ), my_arg_str( src.my_arg_str ), my_cl_program( src.my_cl_program ) {
        // Set my_do_once_flag to the called state.
        std::call_once( my_do_once_flag, [](){} );
    }
private:
    opencl_program( cl_program program ) : my_cl_program( program ) {
        // Set my_do_once_flag to the called state.
        std::call_once( my_do_once_flag, [](){} );
    }

    cl_kernel get_kernel( const std::string& k, Factory &f ) const {
        std::call_once( my_do_once_flag, [this, &k, &f](){ this->init( f, k ); } );
        cl_int err;
        cl_kernel kernel = clCreateKernel( my_cl_program, k.c_str(), &err );
        enforce_cl_retcode( err, std::string( "Failed to create kernel: " ) + k );
        return kernel;
    }

    class file_reader {
    public:
        file_reader( const std::string& filepath ) {
            std::ifstream file_descriptor( filepath, std::ifstream::binary );
            if ( !file_descriptor.is_open() ) {
                std::string str = std::string( "Could not open file: " ) + filepath;
                std::cerr << str << std::endl;
                throw str;
            }
            file_descriptor.seekg( 0, file_descriptor.end );
            size_t length = size_t( file_descriptor.tellg() );
            file_descriptor.seekg( 0, file_descriptor.beg );
            my_content.resize( length );
            char* begin = &*my_content.begin();
            file_descriptor.read( begin, length );
            file_descriptor.close();
        }
        const char* content() { return &*my_content.cbegin(); }
        size_t length() { return my_content.length(); }
    private:
        std::string my_content;
    };

    class opencl_program_builder {
    public:
        typedef void (CL_CALLBACK *cl_callback_type)(cl_program, void*);
        opencl_program_builder( Factory& f, const std::string& name, cl_program program,
                                cl_uint num_devices, cl_device_id* device_list,
                                const char* options, cl_callback_type callback,
                                void* user_data ) {
            cl_int err = clBuildProgram( program, num_devices, device_list, options,
                                         callback, user_data );
            if( err == CL_SUCCESS )
                return;
            std::string str = std::string( "Failed to build program: " ) + name;
            if ( err == CL_BUILD_PROGRAM_FAILURE ) {
                const opencl_device_list &devices = f.devices();
                for ( opencl_device d : devices ) {
                    std::cerr << "Build log for device: " << d.name() << std::endl;
                    size_t log_size;
                    cl_int query_err = clGetProgramBuildInfo(
                        program, d.my_cl_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &log_size );
                    enforce_cl_retcode( query_err, "Failed to get build log size" );
                    if( log_size ) {
                        std::vector<char> output;
                        output.resize( log_size );
                        query_err = clGetProgramBuildInfo(
                            program, d.my_cl_device_id, CL_PROGRAM_BUILD_LOG,
                            output.size(), output.data(), NULL );
                        enforce_cl_retcode( query_err, "Failed to get build output" );
                        std::cerr << output.data() << std::endl;
                    } else {
                        std::cerr << "No build log available" << std::endl;
                    }
                }
            }
            enforce_cl_retcode( err, str );
        }
    };

    class opencl_device_filter {
    public:
        template<typename Filter>
        opencl_device_filter( cl_uint& num_devices, cl_device_id* device_list,
                              Filter filter, const char* message ) {
            for ( cl_uint i = 0; i < num_devices; ++i )
                if ( filter(device_list[i]) ) {
                    device_list[i--] = device_list[--num_devices];
                }
            if ( !num_devices )
                enforce_cl_retcode( CL_DEVICE_NOT_AVAILABLE, message );
        }
    };

    void init( Factory &f, const std::string& ) const {
        cl_uint num_devices;
        enforce_cl_retcode( clGetContextInfo( f.context(), CL_CONTEXT_NUM_DEVICES, sizeof( num_devices ), &num_devices, NULL ),
            "Failed to get OpenCL context info" );
        if ( !num_devices )
            enforce_cl_retcode( CL_DEVICE_NOT_FOUND, "No supported devices found" );
        cl_device_id *device_list = (cl_device_id *)alloca( num_devices*sizeof( cl_device_id ) );
        enforce_cl_retcode( clGetContextInfo( f.context(), CL_CONTEXT_DEVICES, num_devices*sizeof( cl_device_id ), device_list, NULL ),
            "Failed to get OpenCL context info" );
        const char *options = NULL;
        switch ( my_type ) {
        case opencl_program_type::SOURCE: {
            file_reader fr( my_arg_str );
            const char *s[] = { fr.content() };
            const size_t l[] = { fr.length() };
            cl_int err;
            my_cl_program = clCreateProgramWithSource( f.context(), 1, s, l, &err );
            enforce_cl_retcode( err, std::string( "Failed to create program: " ) + my_arg_str );
            opencl_device_filter(
                num_devices, device_list,
                []( const opencl_device& d ) -> bool {
                    return !d.compiler_available() || !d.linker_available();
                }, "No one device supports building program from sources" );
            opencl_program_builder(
                f, my_arg_str, my_cl_program, num_devices, device_list,
                options, /*callback*/ NULL, /*user data*/NULL );
            break;
        }
        case opencl_program_type::SPIR:
            options = "-x spir";
        case opencl_program_type::PRECOMPILED: {
            file_reader fr( my_arg_str );
            std::vector<const unsigned char*> s(
                num_devices, reinterpret_cast<const unsigned char*>(fr.content()) );
            std::vector<size_t> l( num_devices, fr.length() );
            std::vector<cl_int> bin_statuses( num_devices, -1 );
            cl_int err;
            my_cl_program = clCreateProgramWithBinary( f.context(), num_devices,
                                                       device_list, l.data(), s.data(),
                                                       bin_statuses.data(), &err );
            if( err != CL_SUCCESS ) {
                std::string statuses_str;
                for( cl_int st : bin_statuses )
                    statuses_str += std::to_string( st );
                enforce_cl_retcode( err, std::string( "Failed to create program, error " + std::to_string( err ) + " : " ) + my_arg_str +
                                    std::string( ", binary_statuses = " ) + statuses_str );
            }
            opencl_program_builder(
                f, my_arg_str, my_cl_program, num_devices, device_list,
                options, /*callback*/ NULL, /*user data*/NULL );
            break;
        }
        default:
            __TBB_ASSERT( false, "Unsupported program type" );
        }
    }

    opencl_program_type my_type;
    std::string my_arg_str;
    mutable cl_program my_cl_program;
    mutable std::once_flag my_do_once_flag;

    template<typename... Args>
    friend class opencl_node;
};

template <int N1,int N2>
struct port_ref_impl {
    // "+1" since the port_ref range is a closed interval (includes its endpoints).
    static const int size = N2-N1+1;

};

// The purpose of the port_ref_impl is the pretty syntax: the deduction of a compile-time constant is processed from the return type.
// So it is possible to use this helper without parentheses, e.g. "port_ref<0>".
template <int N1, int N2 = N1>
port_ref_impl<N1,N2> port_ref() {
    return port_ref_impl<N1,N2>();
};

template <typename T>
struct num_arguments {
    static const int value = 1;
};

template <int N1, int N2>
struct num_arguments<port_ref_impl<N1,N2>(*)()> {
    static const int value = port_ref_impl<N1,N2>::size;
};

template <int N1, int N2>
struct num_arguments<port_ref_impl<N1,N2>> {
    static const int value = port_ref_impl<N1,N2>::size;
};

template<typename... Args>
class opencl_node;

template <typename... Args>
void ignore_return_values( Args&&... ) {}

template <typename T>
T or_return_values( T&& t ) { return t; }
template <typename T, typename... Rest>
T or_return_values( T&& t, Rest&&... rest ) {
    return t | or_return_values( std::forward<Rest>(rest)... );
}


#define is_typedef(type)                                                    \
    template <typename T>                                                   \
    struct is_##type {                                                      \
        template <typename C>                                               \
        static std::true_type check( typename C::type* );                   \
        template <typename C>                                               \
        static std::false_type check( ... );                                \
                                                                            \
        static const bool value = decltype(check<T>(0))::value;             \
    }

is_typedef( native_object_type );
is_typedef( memory_object_type );

template <typename T>
typename std::enable_if<is_native_object_type<T>::value, typename T::native_object_type>::type get_native_object( const T &t ) {
    return t.native_object();
}

template <typename T>
typename std::enable_if<!is_native_object_type<T>::value, T>::type get_native_object( T t ) {
    return t;
}

// send_if_memory_object checks if the T type has memory_object_type and call the send method for the object.
template <typename T, typename Factory>
typename std::enable_if<is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device device, dependency_msg<T, Factory> &dmsg ) {
    const T &t = dmsg.data( false );
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    dependency_msg<mem_obj_t, Factory> d( mem_obj );
    if ( dmsg.get_event() ) d.set_event( *dmsg.get_event() );
    mem_obj.send( device, d );
    if ( d.get_event() ) dmsg.set_event( *d.get_event() );
}

template <typename T>
typename std::enable_if<is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device device, const T &t ) {
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    dependency_msg<mem_obj_t, typename mem_obj_t::opencl_factory_type> dmsg( mem_obj );
    mem_obj.send( device, dmsg );
}

template <typename T>
typename std::enable_if<!is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device, const T& ) {};

// receive_if_memory_object checks if the T type has memory_object_type and call the receive method for the object.
template <typename T, typename Factory>
typename std::enable_if<is_memory_object_type<T>::value>::type receive_if_memory_object( const dependency_msg<T, Factory> &dmsg ) {
    const T &t = dmsg.data( false );
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    dependency_msg<mem_obj_t, Factory> d( mem_obj );
    if ( dmsg.get_event() ) d.set_event( *dmsg.get_event() );
    mem_obj.receive( d );
    if ( d.get_event() ) dmsg.set_event( *d.get_event() );
}

template <typename T>
typename std::enable_if<!is_memory_object_type<T>::value>::type  receive_if_memory_object( const T& ) {}

template<typename JP>
struct key_from_policy {
    typedef size_t type;
    typedef std::false_type is_key_matching;
};

template<typename Key>
struct key_from_policy< key_matching<Key> > {
    typedef Key type;
    typedef std::true_type is_key_matching;
};

template<typename Key>
struct key_from_policy< key_matching<Key&> > {
    typedef const Key &type;
    typedef std::true_type is_key_matching;
};

template<typename Key>
class opencl_device_with_key {
    opencl_device my_device;
    typename std::decay<Key>::type my_key;
public:
    // TODO: investigate why defaul ctor is required
    opencl_device_with_key() {}
    opencl_device_with_key( opencl_device d, Key k ) : my_device( d ), my_key( k ) {}
    Key key() const { return my_key; }
    opencl_device device() const { return my_device; }
};

/*
    /---------------------------------------- opencl_node ---------------------------------------\
    |                                                                                            |
    |   /--------------\   /----------------------\   /-----------\   /----------------------\   |
    |   |              |   |    (device_with_key) O---O           |   |                      |   |
    |   |              |   |                      |   |           |   |                      |   |
    O---O indexer_node O---O device_selector_node O---O join_node O---O      kernel_node     O---O
    |   |              |   | (multifunction_node) |   |           |   | (multifunction_node) |   |
    O---O              |   |                      O---O           |   |                      O---O
    |   \--------------/   \----------------------/   \-----------/   \----------------------/   |
    |                                                                                            |
    \--------------------------------------------------------------------------------------------/
*/

template<typename JP, typename Factory, typename... Ports>
class opencl_node< tuple<Ports...>, JP, Factory > : public composite_node < tuple<dependency_msg<Ports, Factory>...>, tuple<dependency_msg<Ports, Factory>...> >{
    typedef tuple<dependency_msg<Ports, Factory>...> input_tuple;
    typedef input_tuple output_tuple;
    typedef typename key_from_policy<JP>::type key_type;
    typedef composite_node<input_tuple, output_tuple> base_type;
    static const size_t NUM_INPUTS = tuple_size<input_tuple>::value;
    static const size_t NUM_OUTPUTS = tuple_size<output_tuple>::value;

    typedef typename internal::make_sequence<NUM_INPUTS>::type input_sequence;
    typedef typename internal::make_sequence<NUM_OUTPUTS>::type output_sequence;

    typedef indexer_node<dependency_msg<Ports, Factory>...> indexer_node_type;
    typedef typename indexer_node_type::output_type indexer_node_output_type;
    typedef tuple<opencl_device_with_key<key_type>, dependency_msg<Ports, Factory>...> kernel_input_tuple;
    typedef multifunction_node<indexer_node_output_type, kernel_input_tuple> device_selector_node;
    typedef multifunction_node<kernel_input_tuple, output_tuple> kernel_multifunction_node;

    template <int... S>
    typename base_type::input_ports_type get_input_ports( internal::sequence<S...> ) {
        return std::tie( internal::input_port<S>( my_indexer_node )... );
    }

    template <int... S>
    typename base_type::output_ports_type get_output_ports( internal::sequence<S...> ) {
        return std::tie( internal::output_port<S>( my_kernel_node )... );
    }

    typename base_type::input_ports_type get_input_ports() {
        return get_input_ports( input_sequence() );
    }

    typename base_type::output_ports_type get_output_ports() {
        return get_output_ports( output_sequence() );
    }

    template <int N>
    int make_Nth_edge() {
        make_edge( internal::output_port<N>( my_device_selector_node ), internal::input_port<N>( my_join_node ) );
        return 0;
    }

    template <int... S>
    void make_edges( internal::sequence<S...> ) {
        make_edge( my_indexer_node, my_device_selector_node );
        make_edge( my_device_selector_node, my_join_node );
        ignore_return_values( make_Nth_edge<S + 1>()... );
        make_edge( my_join_node, my_kernel_node );
    }

    void make_edges() {
        make_edges( input_sequence() );
    }

    class device_selector_base {
    public:
        virtual void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) = 0;
        virtual device_selector_base *clone( opencl_node &n ) const = 0;
        virtual ~device_selector_base() {}
    };

    template <typename UserFunctor>
    class device_selector : public device_selector_base, tbb::internal::no_assign {
    public:
        device_selector( UserFunctor uf, opencl_node &n, Factory &f ) : my_user_functor( uf ), my_node(n), my_factory( f ) {
            my_port_epoches.fill( 0 );
        }

        /* override */ void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) {
            send_and_put( my_port_epoches[v.tag()], v, op, input_sequence() );
            __TBB_ASSERT( (std::is_same<typename key_from_policy<JP>::is_key_matching, std::false_type>::value) || my_port_epoches[v.tag()] == 0, "Epoch is changed when key matching is requested" );
        }

        /* override */ device_selector_base *clone( opencl_node &n ) const {
            return new device_selector( my_user_functor, n, my_factory );
        }
    private:
        template <int... S>
        void send_and_put( size_t &epoch, const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op, internal::sequence<S...> ) {
            typedef void(device_selector<UserFunctor>::*send_and_put_fn)(size_t &, const indexer_node_output_type &, typename device_selector_node::output_ports_type &);
            static std::array <send_and_put_fn, NUM_INPUTS > dispatch = { { &device_selector<UserFunctor>::send_and_put_impl<S>... } };
            (this->*dispatch[v.tag()])( epoch, v, op );
        }

        template <typename T>
        key_type get_key( std::false_type, const T &, size_t &epoch ) {
            __TBB_STATIC_ASSERT( (std::is_same<key_type, size_t>::value), "" );
            return epoch++;
        }

        template <typename T>
        key_type get_key( std::true_type, const T &t, size_t &/*epoch*/ ) {
            using tbb::flow::key_from_message;
            return key_from_message<key_type>( t );
        }

        template <int N>
        void send_and_put_impl( size_t &epoch, const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) {
            typedef typename tuple_element<N + 1, typename device_selector_node::output_ports_type>::type::output_type elem_type;
            elem_type e = internal::cast_to<elem_type>( v );
            opencl_device device = get_device( get_key( typename key_from_policy<JP>::is_key_matching(), e, epoch ), get<0>( op ) );
            send_if_memory_object( device, e );
            get<N + 1>( op ).try_put( e );
        }

        template< typename DevicePort >
        opencl_device get_device( key_type key, DevicePort& dp ) {
            typename std::unordered_map<typename std::decay<key_type>::type, epoch_desc>::iterator it = my_devices.find( key );
            if ( it == my_devices.end() ) {
                opencl_device d = my_user_functor( my_factory.devices() );
                std::tie( it, std::ignore ) = my_devices.insert( std::make_pair( key, d ) );
                bool res = dp.try_put( opencl_device_with_key<key_type>( d, key ) );
                __TBB_ASSERT_EX( res, NULL );
                my_node.notify_new_device( d );
            }
            epoch_desc &e = it->second;
            opencl_device d = e.my_device;
            if ( ++e.my_request_number == NUM_INPUTS ) my_devices.erase( it );
            return d;
        }

        struct epoch_desc {
            epoch_desc( opencl_device d ) : my_device( d ), my_request_number( 0 ) {}
            opencl_device my_device;
            size_t my_request_number;
        };

        std::unordered_map<typename std::decay<key_type>::type, epoch_desc> my_devices;
        std::array<size_t, NUM_INPUTS> my_port_epoches;
        UserFunctor my_user_functor;
        opencl_node &my_node;
        Factory &my_factory;
    };

    class device_selector_body {
    public:
        device_selector_body( device_selector_base *d ) : my_device_selector( d ) {}

        /* override */ void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) {
            (*my_device_selector)(v, op);
        }
    private:
        device_selector_base *my_device_selector;
    };

    // Forward declaration.
    class ndranges_mapper_base;

    class opencl_kernel_base : tbb::internal::no_copy {
        cl_kernel clone_kernel() const {
            size_t ret_size;

            std::vector<char> kernel_name;
            for ( size_t curr_size = 32;; curr_size <<= 1 ) {
                kernel_name.resize( curr_size <<= 1 );
                enforce_cl_retcode( clGetKernelInfo( my_kernel, CL_KERNEL_FUNCTION_NAME, curr_size, kernel_name.data(), &ret_size ), "Failed to get kernel info" );
                if ( ret_size < curr_size ) break;
            }

            cl_program program;
            enforce_cl_retcode( clGetKernelInfo( my_kernel, CL_KERNEL_PROGRAM, sizeof( program ), &program, &ret_size ), "Failed to get kernel info" );
            __TBB_ASSERT( ret_size == sizeof( program ), NULL );

            return opencl_program<Factory>( program ).get_kernel( kernel_name.data(), my_factory );
        }

        // ------------- NDRange getters ------------- //
        template <typename NDRange>
        NDRange ndrange_value( NDRange&& r, const kernel_input_tuple& ) const { return r; }
        template <int N>
        typename tuple_element<N+1,kernel_input_tuple>::type::value_type ndrange_value( port_ref_impl<N,N>, const kernel_input_tuple& ip ) const { 
            // "+1" since get<0>(ip) is opencl_device.
            return get<N+1>(ip).data(false);
        }
        template <int N1,int N2>
        void ndrange_value( port_ref_impl<N1,N2>, const kernel_input_tuple& ip ) const { 
            __TBB_STATIC_ASSERT( N1==N2, "Do not use a port_ref range (e.g. port_ref<0,2>) as an argument for the set_ndranges routine" );
        }
        template <int N>
        typename tuple_element<N+1,kernel_input_tuple>::type::value_type ndrange_value( port_ref_impl<N,N>(*)(), const kernel_input_tuple& ip ) const { 
            return ndrange_value(port_ref<N,N>(), ip);
        }
        template <int N1,int N2>
        void ndrange_value( port_ref_impl<N1,N2>(*)(), const kernel_input_tuple& ip ) const { 
            return ndrange_value(port_ref<N1,N2>(), ip);
        }
        // ------------------------------------------- //
    public:
        typedef typename kernel_multifunction_node::output_ports_type output_ports_type;

        virtual void enqueue( const ndranges_mapper_base *ndranges_mapper, const kernel_input_tuple &ip, output_ports_type &op, graph &g ) = 0;
        virtual void send_memory_objects( opencl_device d ) = 0;
        virtual opencl_kernel_base *clone() const = 0;
        virtual ~opencl_kernel_base () {
            enforce_cl_retcode( clReleaseKernel( my_kernel ), "Failed to release a kernel" );
        }

        template <typename GlbNDRange, typename LclNDRange>
        cl_event enqueue( GlbNDRange&& glb_range, LclNDRange&& lcl_range, int num_events, std::array<cl_event, NUM_INPUTS> events, const kernel_input_tuple& ip ) {
            return my_factory.enqueue_kernel( get<0>( ip ).device(), my_kernel, ndrange_value( glb_range, ip ), ndrange_value( lcl_range, ip ), num_events, events.data() );
        }
    protected:
        opencl_kernel_base( const opencl_program<Factory>& p, const std::string& kernel_name, Factory &f )
            : my_kernel( p.get_kernel( kernel_name, f ) ), my_factory( f )
        {}

        opencl_kernel_base( const opencl_kernel_base &k )
            : my_kernel( k.clone_kernel() ), my_factory( k.my_factory )
        {}

        const cl_kernel my_kernel;
        Factory &my_factory;
    };

    // Container for ndrandes. It can contain either port references or real ndranges.
    class ndranges_mapper_base {
    public:
        virtual cl_event enqueue_kernel( opencl_kernel_base *k, const kernel_input_tuple& ip, int num_events, const std::array<cl_event, NUM_INPUTS> &events ) const = 0;
        virtual ndranges_mapper_base *clone() const = 0;
        virtual ~ndranges_mapper_base() {}
    };

    template <typename... Args>
    class opencl_kernel : public opencl_kernel_base {
        typedef typename opencl_kernel_base::output_ports_type output_ports_type;
        // --------- Kernel argument helpers --------- //
        template <int Place, typename T>
        void set_one_kernel_arg(const T& t) {
            auto p = get_native_object( t );
            enforce_cl_retcode( clSetKernelArg( this->my_kernel, Place, sizeof( p ), &p ), "Failed to set a kernel argument" );
        }

        template <int Place, int N>
        int set_one_arg_from_range( const kernel_input_tuple& ip ) {
            // "+1" since get<0>(ip) is opencl_device
            set_one_kernel_arg<Place>( get<N + 1>( ip ).data( false ) );
            return 0;
        }

        template <int Place, int Start, int... S>
        void set_args_range(const kernel_input_tuple& ip, internal::sequence<S...>) {
            ignore_return_values( set_one_arg_from_range<Place + S, Start + S>( ip )... );
        }

        template <int Place, int N1, int N2>
        void set_arg_impl( const kernel_input_tuple& ip, port_ref_impl<N1, N2> ) {
            set_args_range<Place,N1>( ip, typename internal::make_sequence<port_ref_impl<N1, N2>::size>::type() );
        }

        template <int Place, int N1, int N2>
        void set_arg_impl( const kernel_input_tuple& ip, port_ref_impl<N1, N2>(*)() ) {
            set_arg_impl<Place>( ip, port_ref<N1, N2>() );
        }

        template <int Place, typename T>
        void set_arg_impl( const kernel_input_tuple&, const T& t ) {
            set_one_kernel_arg<Place>( t );
        }

        template <int>
        void set_args( const kernel_input_tuple& ) {}

        template <int Place, typename T, typename... Rest>
        void set_args( const kernel_input_tuple& ip, const T& t, Rest&&... rest ) {
            set_arg_impl<Place>( ip, t );
            set_args<Place+num_arguments<T>::value>( ip, std::forward<Rest>(rest)... );
        }
        // ------------------------------------------- //

        // -------- Kernel event list helpers -------- //
        int add_event_to_list( std::array<cl_event, NUM_INPUTS> &events, int &num_events, const cl_event *e ) {
            __TBB_ASSERT( (static_cast<typename std::array<cl_event, NUM_INPUTS>::size_type>(num_events) < events.size()), NULL );
            if ( e ) events[num_events++] = *e;
            return 0;
        }

        template <int... S>
        int generate_event_list( std::array<cl_event, NUM_INPUTS> &events, const kernel_input_tuple& ip, internal::sequence<S...> ) {
            int num_events = 0;
            ignore_return_values( add_event_to_list( events, num_events, get<S + 1>( ip ).get_event() )... );
            return num_events;
        }
        // ------------------------------------------- //

        // ---------- Update events helpers ---------- //
        template <int N>
        bool update_event_and_try_put( graph &g, cl_event e, const kernel_input_tuple& ip, output_ports_type &op ) {
            auto t = get<N + 1>( ip );
            t.set_event( e );
            t.set_graph( g );
            auto &port = get<N>( op );
            return port.try_put( t );
        }

        template <int... S>
        bool update_events_and_try_put( graph &g, cl_event e, const kernel_input_tuple& ip, output_ports_type &op, internal::sequence<S...> ) {
            return or_return_values( update_event_and_try_put<S>( g, e, ip, op )... );
        }
        // ------------------------------------------- //

        class set_args_func : tbb::internal::no_assign {
        public:
            set_args_func( opencl_kernel &k, const kernel_input_tuple &ip ) : my_opencl_kernel( k ), my_ip( ip ) {}
            // It is immpossible to use Args... because a function pointer cannot be casted to a function reference implicitly.
            // Allow the compiler to deduce types for function pointers automatically.
            template <typename... A> 
            void operator()( A&&... a ) {
                my_opencl_kernel.set_args<0>( my_ip, std::forward<A>( a )... );
            }
        private:
            opencl_kernel &my_opencl_kernel;
            const kernel_input_tuple &my_ip;
        };

        class send_func : tbb::internal::no_assign {
        public:
            send_func( opencl_device d ) : my_device( d ) {}
            void operator()() {}
            template <typename T, typename... Rest> 
            void operator()( T &&t, Rest&&... rest ) {
                send_if_memory_object( my_device, std::forward<T>( t ) );
                (*this)( std::forward<Rest>( rest )... );
            }
        private:
            opencl_device my_device;
        };

        static void CL_CALLBACK decrement_wait_count_callback( cl_event, cl_int event_command_exec_status, void *data ) {
            tbb::internal::suppress_unused_warning( event_command_exec_status );
            __TBB_ASSERT( event_command_exec_status == CL_COMPLETE, NULL );
            graph &g = *static_cast<graph*>(data);
            g.decrement_wait_count();
        }

    public:
        opencl_kernel( const opencl_program<Factory>& p, const std::string &kernel_name, Factory &f, Args&&... args )
            : opencl_kernel_base( p, kernel_name, f )
            , my_args_pack( std::forward<Args>( args )... )
        {}

        opencl_kernel( const opencl_kernel_base &k ) : opencl_kernel_base( k ), my_args_pack( k.my_args_pack ) {}

        opencl_kernel( const opencl_kernel_base &k, Args&&... args ) : opencl_kernel_base( k ), my_args_pack( std::forward<Args>(args)... ) {}

        /* override */ void enqueue( const ndranges_mapper_base *ndrange_mapper, const kernel_input_tuple &ip, output_ports_type &op, graph &g ) {
            // Set arguments for the kernel.
            tbb::internal::call( set_args_func( *this, ip ), my_args_pack );

            // Gather events from all ports to an array.
            std::array<cl_event, NUM_INPUTS> events;
            int num_events = generate_event_list( events, ip, input_sequence() );

            // Enqueue the kernel. ndrange_mapper is used only to obtain ndrange. Actually, it calls opencl_kernel_base::enqueue.
            cl_event e = ndrange_mapper->enqueue_kernel( this, ip, num_events, events );

            // Update events in dependency messages and try_put them to the output ports.
            if ( !update_events_and_try_put( g, e, ip, op, input_sequence() ) ) {
                // No one message was passed to successors so set a callback to extend the graph lifetime until the kernel completion.
                g.increment_wait_count();
                enforce_cl_retcode( clSetEventCallback( e, CL_COMPLETE, decrement_wait_count_callback, &g ), "Failed to set a callback" );
                this->my_factory.flush( get<0>( ip ).device() );
            }
            // Release our own reference to cl_event.
            enforce_cl_retcode( clReleaseEvent( e ), "Failed to release an event" );
        }

        virtual void send_memory_objects( opencl_device d ) {
            // Find opencl_buffer and send them to the devece.
            tbb::internal::call( send_func( d ), my_args_pack );
        }

        /* override */ opencl_kernel_base *clone() const {
            // Create new opencl_kernel with copying constructor.
            return new opencl_kernel<Args...>( *this );
        }

    private:
        tbb::internal::stored_pack<Args...> my_args_pack;
    };

    template <typename GlbNDRange, typename LclNDRange>
    class ndranges_mapper : public ndranges_mapper_base, tbb::internal::no_assign {
    public:
        template <typename GRange, typename LRange>
        ndranges_mapper( GRange&& glb, LRange&& lcl ) : my_global_work_size( glb ), my_local_work_size( lcl ) {}

        /*override*/ cl_event enqueue_kernel( opencl_kernel_base *k, const kernel_input_tuple &ip, int num_events, const std::array<cl_event, NUM_INPUTS> &events ) const {
            return k->enqueue( my_global_work_size, my_local_work_size, num_events, events, ip );
        }

        /*override*/ ndranges_mapper_base *clone() const {
            return new ndranges_mapper<GlbNDRange, LclNDRange>( my_global_work_size, my_local_work_size );
        }

    private:
        GlbNDRange my_global_work_size;
        LclNDRange my_local_work_size;
    };

    void enqueue_kernel( const kernel_input_tuple &ip, typename opencl_kernel_base::output_ports_type &op ) const {
        __TBB_ASSERT(my_ndranges_mapper, "NDRanges are not set. Call set_ndranges before running opencl_node.");
        my_opencl_kernel->enqueue( my_ndranges_mapper, ip, op, this->my_graph );
    }

    // Body for kernel_multifunction_node.
    class kernel_body : tbb::internal::no_assign {
    public:
        kernel_body( const opencl_node &node ) : my_opencl_node( node ) {}
        void operator()( const kernel_input_tuple &ip, typename opencl_kernel_base::output_ports_type &op ) {
            my_opencl_node.enqueue_kernel( ip, op );
        }
    private:
        const opencl_node &my_opencl_node;
    };

    template <typename... Args>
    opencl_kernel_base *make_opencl_kernel( const opencl_program<Factory> &p, const std::string &kernel_name, Factory &f, Args&&... args ) const {
        return new opencl_kernel<Args...>( p, kernel_name, f, std::forward<Args>( args )... );
    }

    template <typename GlobalNDRange, typename LocalNDRange = std::array<size_t,3>>
    void set_ndranges_impl( GlobalNDRange&& global_work_size, LocalNDRange&& local_work_size = std::array<size_t, 3>( { { 0, 0, 0 } } ) ) {
        if ( my_ndranges_mapper ) delete my_ndranges_mapper;
        my_ndranges_mapper = new ndranges_mapper<typename std::decay<GlobalNDRange>::type, typename std::decay<LocalNDRange>::type>
                ( std::forward<GlobalNDRange>( global_work_size ), std::forward<LocalNDRange>( local_work_size ) );
    }

    void notify_new_device( opencl_device d ) {
        my_opencl_kernel->send_memory_objects( d );
    }

public:
    template <typename DeviceSelector>
    opencl_node( opencl_graph &g, const opencl_program<Factory> &p, const std::string &kernel_name, DeviceSelector d, Factory &f )
        : base_type( g )
        , my_indexer_node( g )
        , my_device_selector( new device_selector<DeviceSelector>( d, *this, f ) )
        , my_device_selector_node( g, serial, device_selector_body( my_device_selector ) )
        , my_join_node( g )
        , my_kernel_node( g, serial, kernel_body( *this ) )
        // By default, opencl_node maps all its ports to the kernel arguments on a one-to-one basis.
        , my_opencl_kernel( make_opencl_kernel( p , kernel_name, f, port_ref<0, NUM_INPUTS - 1>()) )
        , my_ndranges_mapper( NULL )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
    }

    opencl_node( opencl_graph &g, const opencl_program<Factory> &p, const std::string &kernel_name, Factory &f )
        : opencl_node( g, p, kernel_name, g.get_opencl_foundation().get_default_device_selector(), f )
        {}


    opencl_node( const opencl_node &node )
        : base_type( node.my_graph )
        , my_indexer_node( node.my_indexer_node )
        , my_device_selector( node.my_device_selector->clone( *this ) )
        , my_device_selector_node( node.my_graph, serial, device_selector_body( my_device_selector ) )
        , my_join_node( node.my_join_node )
        , my_kernel_node( node.my_graph, serial, kernel_body( *this ) )
        , my_opencl_kernel( node.my_opencl_kernel->clone() )
        , my_ndranges_mapper( node.my_ndranges_mapper ? node.my_ndranges_mapper->clone() : NULL )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
    }

    opencl_node( opencl_node &&node )
        : base_type( node.my_graph )
        , my_indexer_node( std::move( node.my_indexer_node ) )
        , my_device_selector( node.my_device_selector->clone(*this) )
        , my_device_selector_node( node.my_graph, serial, device_selector_body( my_device_selector ) )
        , my_join_node( std::move( node.my_join_node ) )
        , my_kernel_node( node.my_graph, serial, kernel_body( *this ) )
        , my_opencl_kernel( node.my_opencl_kernel )
        , my_ndranges_mapper( node.my_ndranges_mapper )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
        // Set moving node mappers to NULL to prevent double deallocation.
        node.my_opencl_kernel = NULL;
        node.my_ndranges_mapper = NULL;
    }

    ~opencl_node() {
        if ( my_opencl_kernel ) delete my_opencl_kernel;
        if ( my_ndranges_mapper ) delete my_ndranges_mapper;
        if ( my_device_selector ) delete my_device_selector;
    }

    template <typename T>
    void set_ndranges( std::initializer_list<T> global_work_size ) {
        set_ndranges_impl( internal::initializer_list_wrapper<T>( global_work_size ) );
    }

    template <typename GlobalNDRange>
    void set_ndranges( GlobalNDRange&& global_work_size ) {
        set_ndranges_impl( std::forward<GlobalNDRange>( global_work_size ) );
    }

    template <typename T, typename LocalNDRange>
    void set_ndranges( std::initializer_list<T> global_work_size, LocalNDRange&& local_work_size ) {
        set_ndranges_impl( internal::initializer_list_wrapper<T>(global_work_size), std::forward<LocalNDRange>( local_work_size ) );
    }

    template <typename T1, typename T2 = T1>
    void set_ndranges( std::initializer_list<T1> global_work_size, std::initializer_list<T2> local_work_size ) {
        set_ndranges_impl( internal::initializer_list_wrapper<T1>(global_work_size), internal::initializer_list_wrapper<T2>(local_work_size) );
    }

    template <typename GlobalNDRange, typename LocalNDRange>
    void set_ndranges( GlobalNDRange&& global_work_size, LocalNDRange&& local_work_size ) {
        set_ndranges_impl( std::forward<GlobalNDRange>(global_work_size), std::forward<LocalNDRange>(local_work_size) );
    }

    template <typename GlobalNDRange, typename T>
    void set_ndranges( GlobalNDRange&& global_work_size, std::initializer_list<T> local_work_size ) {
        set_ndranges_impl( std::forward<GlobalNDRange>( global_work_size ), internal::initializer_list_wrapper<T>( local_work_size ) );
    }

    template <typename... Args>
    void set_args( Args&&... args ) {
        // Copy the base class of opencl_kernal and create new storage for "Args...".
        opencl_kernel_base *new_opencl_kernel = new opencl_kernel<Args...>( *my_opencl_kernel, std::forward<Args>( args )... );
        delete my_opencl_kernel;
        my_opencl_kernel = new_opencl_kernel;
    }

protected:
    /* override */ void reset_node( reset_flags = rf_reset_protocol ) { __TBB_ASSERT( false, "Not implemented yet" ); }

private:
    indexer_node_type my_indexer_node;
    device_selector_base *my_device_selector;
    device_selector_node my_device_selector_node;
    join_node<kernel_input_tuple, JP> my_join_node;
    kernel_multifunction_node my_kernel_node;

    opencl_kernel_base *my_opencl_kernel;
    ndranges_mapper_base *my_ndranges_mapper;
};

template<typename JP, typename... Ports>
class opencl_node< tuple<Ports...>, JP > : public opencl_node < tuple<Ports...>, JP, default_opencl_factory > {
    typedef opencl_node < tuple<Ports...>, JP, default_opencl_factory > base_type;
public:
    opencl_node( opencl_graph &g, const std::string &kernel )
        : base_type( g, kernel, g.get_opencl_foundation().get_default_device_selector(), g.get_opencl_foundation().get_default_opencl_factory() )
    {}
    opencl_node( opencl_graph &g, const opencl_program<default_opencl_factory> &p, const std::string &kernel )
        : base_type( g, p, kernel,
                g.get_opencl_foundation().get_default_device_selector(), g.get_opencl_foundation().get_default_opencl_factory() )
    {}
    template <typename DeviceSelector>
    opencl_node( opencl_graph &g, const opencl_program<default_opencl_factory> &p, const std::string &kernel, DeviceSelector d )
        : base_type( g, p , kernel, d, g.get_opencl_foundation().get_default_opencl_factory() )
    {}
};

template<typename... Ports>
class opencl_node< tuple<Ports...> > : public opencl_node < tuple<Ports...>, queueing, default_opencl_factory > {
    typedef opencl_node < tuple<Ports...>, queueing, default_opencl_factory > base_type;
public:
    opencl_node( opencl_graph &g, const std::string &kernel )
        : base_type( g, kernel, g.get_opencl_foundation().get_default_device_selector(), g.get_opencl_foundation().get_default_opencl_factory() )
    {}
    opencl_node( opencl_graph &g, const opencl_program<default_opencl_factory> &p, const std::string &kernel )
        : base_type( g, p, kernel,
                g.get_opencl_foundation().get_default_device_selector(), g.get_opencl_foundation().get_default_opencl_factory() )
    {}
    template <typename DeviceSelector>
    opencl_node( opencl_graph &g, const opencl_program<default_opencl_factory> &p, const std::string &kernel, DeviceSelector d )
        : base_type( g, p, kernel, d, g.get_opencl_foundation().get_default_opencl_factory() )
    {}
};

} // namespace interface8

using interface8::opencl_graph;
using interface8::opencl_node;
using interface8::read_only;
using interface8::read_write;
using interface8::write_only;
using interface8::opencl_buffer;
using interface8::opencl_device;
using interface8::opencl_device_list;
using interface8::opencl_program;
using interface8::opencl_program_type;
using interface8::dependency_msg;
using interface8::port_ref;
using interface8::opencl_factory;

} // namespace flow
} // namespace tbb
#endif /* __TBB_PREVIEW_OPENCL_NODE */

#endif // __TBB_flow_graph_opencl_node_H
