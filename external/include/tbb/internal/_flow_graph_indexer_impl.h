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

#ifndef __TBB__flow_graph_indexer_impl_H
#define __TBB__flow_graph_indexer_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "tbb/internal/_flow_graph_types_impl.h"

namespace internal {

    // Output of the indexer_node is a tbb::flow::tagged_msg, and will be of
    // the form  tagged_msg<tag, result>
    // where the value of tag will indicate which result was put to the
    // successor.  
    
    template<typename IndexerNodeBaseType, typename T, size_t K>
    task* do_try_put(const T &v, void *p) {
        typename IndexerNodeBaseType::output_type o(K, v);
        return reinterpret_cast<IndexerNodeBaseType *>(p)->try_put_task(&o);
    }

    template<typename TupleTypes,int N>
    struct indexer_helper {
        template<typename IndexerNodeBaseType, typename PortTuple>
        static inline void set_indexer_node_pointer(PortTuple &my_input, IndexerNodeBaseType *p) {
            typedef typename tuple_element<N-1, TupleTypes>::type T;
            task *(*indexer_node_put_task)(const T&, void *) = do_try_put<IndexerNodeBaseType, T, N-1>;
            tbb::flow::get<N-1>(my_input).set_up(p, indexer_node_put_task);
            indexer_helper<TupleTypes,N-1>::template set_indexer_node_pointer<IndexerNodeBaseType,PortTuple>(my_input, p);
        }
        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            indexer_helper<TupleTypes,N-1>::reset_inputs(my_input, f);
            tbb::flow::get<N-1>(my_input).reset_receiver(f);
        }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        template<typename InputTuple>
        static inline void extract(InputTuple &my_input) {
            indexer_helper<TupleTypes,N-1>::extract(my_input);
            tbb::flow::get<N-1>(my_input).extract_receiver();
        }
#endif
    };

    template<typename TupleTypes>
    struct indexer_helper<TupleTypes,1> {
        template<typename IndexerNodeBaseType, typename PortTuple>
        static inline void set_indexer_node_pointer(PortTuple &my_input, IndexerNodeBaseType *p) {
            typedef typename tuple_element<0, TupleTypes>::type T;
            task *(*indexer_node_put_task)(const T&, void *) = do_try_put<IndexerNodeBaseType, T, 0>;
            tbb::flow::get<0>(my_input).set_up(p, indexer_node_put_task);
        }
        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            tbb::flow::get<0>(my_input).reset_receiver(f);
        }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        template<typename InputTuple>
        static inline void extract(InputTuple &my_input) {
            tbb::flow::get<0>(my_input).extract_receiver();
        }
#endif
    };

    template<typename T>
    class indexer_input_port : public receiver<T> {
    private:
        void* my_indexer_ptr;
        typedef task* (* forward_function_ptr)(T const &, void* );
        forward_function_ptr my_try_put_task;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        spin_mutex my_pred_mutex;
        typedef typename receiver<T>::built_predecessors_type built_predecessors_type;
        built_predecessors_type my_built_predecessors;
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */
    public:
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        indexer_input_port() : my_pred_mutex() {}
        indexer_input_port( const indexer_input_port & /*other*/ ) : receiver<T>(), my_pred_mutex() {
        }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */
        void set_up(void *p, forward_function_ptr f) {
                my_indexer_ptr = p;
                my_try_put_task = f;
            }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        typedef typename receiver<T>::predecessor_list_type predecessor_list_type;

        /*override*/ built_predecessors_type &built_predecessors() { return my_built_predecessors; }

        /*override*/size_t predecessor_count() {
            spin_mutex::scoped_lock l(my_pred_mutex);
            return my_built_predecessors.edge_count();
        }
        /*override*/void internal_add_built_predecessor(sender<T> &p) {
            spin_mutex::scoped_lock l(my_pred_mutex);
            my_built_predecessors.add_edge(p);
        }
        /*override*/void internal_delete_built_predecessor(sender<T> &p) {
            spin_mutex::scoped_lock l(my_pred_mutex);
            my_built_predecessors.delete_edge(p);
        }
        /*override*/void copy_predecessors( predecessor_list_type &v) {
            spin_mutex::scoped_lock l(my_pred_mutex);
            return my_built_predecessors.copy_edges(v);
        }
        /*override*/void clear_predecessors() {
            spin_mutex::scoped_lock l(my_pred_mutex);
            my_built_predecessors.clear();
        }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */
    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class internal::broadcast_cache;
        template<typename X, typename Y> friend class internal::round_robin_cache;
        task *try_put_task(const T &v) {
            return my_try_put_task(v, my_indexer_ptr);
        }

    public:
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        /*override*/void reset_receiver(reset_flags f) { if(f&rf_clear_edges) my_built_predecessors.clear(); }
#else
        /*override*/void reset_receiver(reset_flags /*f*/) { }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        void extract_receiver() { my_built_predecessors.receiver_extract(*this); }
#endif
    };

    template<typename InputTuple, typename OutputType, typename StructTypes>
    class indexer_node_FE {
    public:
        static const int N = tbb::flow::tuple_size<InputTuple>::value;
        typedef OutputType output_type;
        typedef InputTuple input_type;

        // Some versions of Intel C++ compiler fail to generate an implicit constructor for the class which has std::tuple as a member.
        indexer_node_FE() : my_inputs() {}

        input_type &input_ports() { return my_inputs; }
    protected:
        input_type my_inputs;
    };

    //! indexer_node_base
    template<typename InputTuple, typename OutputType, typename StructTypes>
    class indexer_node_base : public graph_node, public indexer_node_FE<InputTuple, OutputType,StructTypes>,
                           public sender<OutputType> {
    protected:
       using graph_node::my_graph;
    public:
        static const size_t N = tbb::flow::tuple_size<InputTuple>::value;
        typedef OutputType output_type;
        typedef StructTypes tuple_types;
        typedef receiver<output_type> successor_type;
        typedef indexer_node_FE<InputTuple, output_type,StructTypes> input_ports_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        typedef typename sender<output_type>::built_successors_type built_successors_type;
        typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    private:
        // ----------- Aggregator ------------
        enum op_type { reg_succ, rem_succ, try__put_task
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            , add_blt_succ, del_blt_succ,
             blt_succ_cnt, blt_succ_cpy
#endif
        };
        enum op_stat {WAIT=0, SUCCEEDED, FAILED};
        typedef indexer_node_base<InputTuple,output_type,StructTypes> class_type;

        class indexer_node_base_operation : public aggregated_operation<indexer_node_base_operation> {
        public:
            char type;
            union {
                output_type const *my_arg;
                successor_type *my_succ;
                task *bypass_t;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
                size_t cnt_val;
                successor_list_type *succv;
#endif
            };
            indexer_node_base_operation(const output_type* e, op_type t) :
                type(char(t)), my_arg(e) {}
            indexer_node_base_operation(const successor_type &s, op_type t) : type(char(t)), 
                my_succ(const_cast<successor_type *>(&s)) {}
            indexer_node_base_operation(op_type t) : type(char(t)) {}
        };

        typedef internal::aggregating_functor<class_type, indexer_node_base_operation> handler_type;
        friend class internal::aggregating_functor<class_type, indexer_node_base_operation>;
        aggregator<handler_type, indexer_node_base_operation> my_aggregator;

        void handle_operations(indexer_node_base_operation* op_list) {
            indexer_node_base_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {

                case reg_succ:
                    my_successors.register_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;

                case rem_succ:
                    my_successors.remove_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case try__put_task: {
                        current->bypass_t = my_successors.try_put_task(*(current->my_arg));
                        __TBB_store_with_release(current->status, SUCCEEDED);  // return of try_put_task actual return value
                    }
                    break;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
                case add_blt_succ:
                    my_successors.internal_add_built_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case del_blt_succ:
                    my_successors.internal_delete_built_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_succ_cnt:
                    current->cnt_val = my_successors.successor_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_succ_cpy:
                    my_successors.copy_successors(*(current->succv));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */
                }
            }
        }
        // ---------- end aggregator -----------
    public:
        indexer_node_base(graph& g) : graph_node(g), input_ports_type() {
            indexer_helper<StructTypes,N>::set_indexer_node_pointer(this->my_inputs, this);
            my_successors.set_owner(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        indexer_node_base(const indexer_node_base& other) : graph_node(other.my_graph), input_ports_type(), sender<output_type>() {
            indexer_helper<StructTypes,N>::set_indexer_node_pointer(this->my_inputs, this);
            my_successors.set_owner(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        bool register_successor(successor_type &r) {
            indexer_node_base_operation op_data(r, reg_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool remove_successor( successor_type &r) {
            indexer_node_base_operation op_data(r, rem_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        task * try_put_task(output_type const *v) {
            indexer_node_base_operation op_data(v, try__put_task);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES

        built_successors_type &built_successors() { return my_successors.built_successors(); }

        void internal_add_built_successor( successor_type &r) {
            indexer_node_base_operation op_data(r, add_blt_succ);
            my_aggregator.execute(&op_data);
        }

        void internal_delete_built_successor( successor_type &r) {
            indexer_node_base_operation op_data(r, del_blt_succ);
            my_aggregator.execute(&op_data);
        }

        size_t successor_count() {
            indexer_node_base_operation op_data(blt_succ_cnt);
            my_aggregator.execute(&op_data);
            return op_data.cnt_val;
        }

        void copy_successors( successor_list_type &v) {
            indexer_node_base_operation op_data(blt_succ_cpy);
            op_data.succv = &v;
            my_aggregator.execute(&op_data);
        } 
        void extract() {
            my_successors.built_successors().sender_extract(*this);
            indexer_helper<StructTypes,N>::extract(this->my_inputs);
        }
#endif /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */
    protected:
        /*override*/void reset_node(reset_flags f) {
            if(f & rf_clear_edges) {
                my_successors.clear();
                indexer_helper<StructTypes,N>::reset_inputs(this->my_inputs,f);
            }
        }

    private:
        broadcast_cache<output_type, null_rw_mutex> my_successors;
    };  //indexer_node_base


    template<int N, typename InputTuple> struct input_types;

    template<typename InputTuple>
    struct input_types<1, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename internal::tagged_msg<size_t, first_type > type;
    };

    template<typename InputTuple>
    struct input_types<2, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type> type;
    };

    template<typename InputTuple>
    struct input_types<3, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type> type;
    };
    
    template<typename InputTuple>
    struct input_types<4, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type> type;
    };
    
    template<typename InputTuple>
    struct input_types<5, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type> type;
    };
    
    template<typename InputTuple>
    struct input_types<6, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename tuple_element<5, InputTuple>::type sixth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type> type;
    };
    
    template<typename InputTuple>
    struct input_types<7, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename tuple_element<5, InputTuple>::type sixth_type;
        typedef typename tuple_element<6, InputTuple>::type seventh_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type> type;
    };


    template<typename InputTuple>
    struct input_types<8, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename tuple_element<5, InputTuple>::type sixth_type;
        typedef typename tuple_element<6, InputTuple>::type seventh_type;
        typedef typename tuple_element<7, InputTuple>::type eighth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type> type;
    };

 
    template<typename InputTuple>
    struct input_types<9, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename tuple_element<5, InputTuple>::type sixth_type;
        typedef typename tuple_element<6, InputTuple>::type seventh_type;
        typedef typename tuple_element<7, InputTuple>::type eighth_type;
        typedef typename tuple_element<8, InputTuple>::type nineth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type, nineth_type> type;
    };

    template<typename InputTuple>
    struct input_types<10, InputTuple> {
        typedef typename tuple_element<0, InputTuple>::type first_type;
        typedef typename tuple_element<1, InputTuple>::type second_type;
        typedef typename tuple_element<2, InputTuple>::type third_type;
        typedef typename tuple_element<3, InputTuple>::type fourth_type;
        typedef typename tuple_element<4, InputTuple>::type fifth_type;
        typedef typename tuple_element<5, InputTuple>::type sixth_type;
        typedef typename tuple_element<6, InputTuple>::type seventh_type;
        typedef typename tuple_element<7, InputTuple>::type eighth_type;
        typedef typename tuple_element<8, InputTuple>::type nineth_type;
        typedef typename tuple_element<9, InputTuple>::type tenth_type;
        typedef typename internal::tagged_msg<size_t, first_type, second_type, third_type,
                                                      fourth_type, fifth_type, sixth_type,
                                                      seventh_type, eighth_type, nineth_type,
                                                      tenth_type> type;
    };

    // type generators
    template<typename OutputTuple>
    struct indexer_types : public input_types<tuple_size<OutputTuple>::value, OutputTuple> {
        static const int N = tbb::flow::tuple_size<OutputTuple>::value;
        typedef typename input_types<N, OutputTuple>::type output_type;
        typedef typename wrap_tuple_elements<N,indexer_input_port,OutputTuple>::type input_ports_type;
        typedef internal::indexer_node_FE<input_ports_type,output_type,OutputTuple> indexer_FE_type;
        typedef internal::indexer_node_base<input_ports_type, output_type, OutputTuple> indexer_base_type;
    };

    template<class OutputTuple>
    class unfolded_indexer_node : public indexer_types<OutputTuple>::indexer_base_type {
    public:
        typedef typename indexer_types<OutputTuple>::input_ports_type input_ports_type;
        typedef OutputTuple tuple_types;
        typedef typename indexer_types<OutputTuple>::output_type output_type;
    private:
        typedef typename indexer_types<OutputTuple>::indexer_base_type base_type;
    public:
        unfolded_indexer_node(graph& g) : base_type(g) {}
        unfolded_indexer_node(const unfolded_indexer_node &other) : base_type(other) {}
    };

} /* namespace internal */

#endif  /* __TBB__flow_graph_indexer_impl_H */
