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

#ifndef _FGT_GRAPH_TRACE_IMPL_H
#define _FGT_GRAPH_TRACE_IMPL_H

#include "../tbb_profiling.h"

namespace tbb {
    namespace internal {

#if TBB_PREVIEW_FLOW_GRAPH_TRACE

static inline void fgt_internal_create_input_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_INPUT_PORT, node, FLOW_NODE, name_index );
}

static inline void fgt_internal_create_output_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_OUTPUT_PORT, node, FLOW_NODE, name_index );
}

template<typename InputType>
void register_input_port(void *node, tbb::flow::receiver<InputType>* port, string_index name_index) {
    //TODO: Make fgt_internal_create_input_port a function template?
    fgt_internal_create_input_port( node, port, name_index);
}    

template < typename PortsTuple, int N >
struct fgt_internal_input_helper {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_INPUT_PORT_0 + N - 1) );
        fgt_internal_input_helper<PortsTuple, N-1>::register_port( node, ports );
    } 
};

template < typename PortsTuple >
struct fgt_internal_input_helper<PortsTuple, 1> {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<0>(ports)), FLOW_INPUT_PORT_0 );
    } 
};

template<typename OutputType>
void register_output_port(void *node, tbb::flow::sender<OutputType>* port, string_index name_index) {
    //TODO: Make fgt_internal_create_output_port a function template?
    fgt_internal_create_output_port( node, static_cast<void *>(port), name_index);
}    

template < typename PortsTuple, int N >
struct fgt_internal_output_helper {
    static void register_port( void *node, PortsTuple &ports ) {
        register_output_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_OUTPUT_PORT_0 + N - 1) ); 
        fgt_internal_output_helper<PortsTuple, N-1>::register_port( node, ports );
    } 
};

template < typename PortsTuple >
struct fgt_internal_output_helper<PortsTuple,1> {
    static void register_port( void *node, PortsTuple &ports ) {
        register_output_port( node, &(tbb::flow::get<0>(ports)), FLOW_OUTPUT_PORT_0 ); 
    } 
};

template< typename NodeType >
void fgt_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::receiver< typename NodeType::input_type > * >(const_cast< NodeType *>(node)) ); 
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc ); 
}

template< typename NodeType >
void fgt_multiinput_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  const_cast<NodeType *>(node); 
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc ); 
}

template< typename NodeType >
static inline void fgt_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::sender< typename NodeType::output_type > * >(const_cast< NodeType *>(node)) ); 
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc ); 
}

static inline void fgt_graph_desc( void *g, const char *desc ) {
    itt_metadata_str_add( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, FLOW_OBJECT_NAME, desc ); 
}

static inline void fgt_body( void *node, void *body ) {
    itt_relation_add( ITT_DOMAIN_FLOW, body, FLOW_BODY, __itt_relation_is_child_of, node, FLOW_NODE );
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node( string_index t, void *g, void *input_port, PortsTuple &ports ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t ); 
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 ); 
    fgt_internal_output_helper<PortsTuple, N>::register_port( input_port, ports ); 
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( string_index t, void *g, void *input_port, PortsTuple &ports, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t ); 
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 ); 
    fgt_internal_output_helper<PortsTuple, N>::register_port( input_port, ports ); 
    fgt_body( input_port, body );
}

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( string_index t, void *g, PortsTuple &ports, void *output_port) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t ); 
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 ); 
    fgt_internal_input_helper<PortsTuple, N>::register_port( output_port, ports ); 
}

static inline void fgt_node( string_index t, void *g, void *output_port ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t ); 
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 ); 
}

static inline void fgt_node_with_body( string_index t, void *g, void *output_port, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t ); 
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 ); 
    fgt_body( output_port, body );
}


static inline void fgt_node( string_index t, void *g, void *input_port, void *output_port ) {
    fgt_node( t, g, output_port );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 );
}

static inline void fgt_node_with_body( string_index t, void *g, void *input_port, void *output_port, void *body ) {
    fgt_node_with_body( t, g, output_port, body );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 ); 
}


static inline void  fgt_node( string_index t, void *g, void *input_port, void *decrement_port, void *output_port ) {
    fgt_node( t, g, input_port, output_port );
    fgt_internal_create_input_port( output_port, decrement_port, FLOW_INPUT_PORT_1 ); 
}

static inline void fgt_make_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_predecessor_to, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_remove_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_sibling_of, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_graph( void *g ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, NULL, FLOW_NULL, FLOW_GRAPH ); 
}

static inline void fgt_begin_body( void *body ) {
    itt_task_begin( ITT_DOMAIN_FLOW, body, FLOW_BODY, NULL, FLOW_NULL, FLOW_BODY );
}

static inline void fgt_end_body( void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_try_put_begin( void *node, void *port ) {
    itt_task_begin( ITT_DOMAIN_FLOW, port, FLOW_OUTPUT_PORT, node, FLOW_NODE, FLOW_OUTPUT_PORT );
}

static inline void fgt_async_try_put_end( void *, void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_reserve( void *node, void *graph ) {
    itt_region_begin( ITT_DOMAIN_FLOW, node, FLOW_NODE, graph, FLOW_GRAPH, FLOW_NULL );
}

static inline void fgt_async_commit( void *node, void *graph ) {
    itt_region_end( ITT_DOMAIN_FLOW, node, FLOW_NODE );
}

#else // TBB_PREVIEW_FLOW_GRAPH_TRACE

static inline void fgt_graph( void * /*g*/ ) { }

template< typename NodeType >
static inline void fgt_multioutput_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

template< typename NodeType >
static inline void fgt_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

static inline void fgt_graph_desc( void * /*g*/, const char * /*desc*/ ) { }

static inline void fgt_body( void * /*node*/, void * /*body*/ ) { }

template< int N, typename PortsTuple > 
static inline void fgt_multioutput_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/, void * /*body*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( string_index /*t*/, void * /*g*/, PortsTuple & /*ports*/, void * /*output_port*/ ) { }

static inline void fgt_node( string_index /*t*/, void * /*g*/, void * /*output_port*/ ) { } 
static inline void fgt_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/ ) { } 
static inline void  fgt_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*decrement_port*/, void * /*output_port*/ ) { }

static inline void fgt_node_with_body( string_index /*t*/, void * /*g*/, void * /*output_port*/, void * /*body*/ ) { }
static inline void fgt_node_with_body( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/, void * /*body*/ ) { }

static inline void fgt_make_edge( void * /*output_port*/, void * /*input_port*/ ) { }
static inline void fgt_remove_edge( void * /*output_port*/, void * /*input_port*/ ) { }

static inline void fgt_begin_body( void * /*body*/ ) { }
static inline void fgt_end_body( void *  /*body*/) { }
static inline void fgt_async_try_put_begin( void * /*node*/, void * /*port*/ ) { }
static inline void fgt_async_try_put_end( void * /*node*/ , void * /*port*/ ) { }
static inline void fgt_async_reserve( void * /*node*/, void * /*graph*/ ) { }
static inline void fgt_async_commit( void * /*node*/, void * /*graph*/ ) { }

#endif // TBB_PREVIEW_FLOW_GRAPH_TRACE

    } // namespace internal
} // namespace tbb

#endif
