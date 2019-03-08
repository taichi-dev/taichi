//#####################################################################
// Copyright 2007, Andrew Selle.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __PTHREAD_QUEUE__
#define __PTHREAD_QUEUE__
#include "POINTER_QUEUE.h"
#include <pthread.h>
#include <iostream>
namespace PhysBAM{

class PTHREAD_QUEUE
{
    const int number_of_threads;
    pthread_t* const threads;
    pthread_attr_t attr;
    pthread_mutex_t queue_lock;
    pthread_cond_t done_condition,todo_condition;
    int active_threads,inactive_threads;

public:
    struct TASK
    {
        virtual ~TASK(){};
        virtual void Run()=0;
    };

    struct EXITER:public TASK
    {
        void Run()
        {pthread_exit(0);}
    };
    
private:
    POINTER_QUEUE queue;
public:

    PTHREAD_QUEUE(const int thread_count)
        :number_of_threads(thread_count),threads(new pthread_t[thread_count]),active_threads(thread_count),inactive_threads(0),queue(65535)
    {
        pthread_attr_init(&attr);
        pthread_cond_init(&done_condition,0);
        pthread_cond_init(&todo_condition,0);
        pthread_mutex_init(&queue_lock,0);

        for(int i=0;i<number_of_threads;i++) pthread_create(&threads[i],0,Thread_Routine,this);
    }

    ~PTHREAD_QUEUE()
    {
        for(int i=0;i<number_of_threads;i++) Queue(new EXITER());
        pthread_cond_destroy(&done_condition);
        pthread_cond_destroy(&todo_condition);
        pthread_mutex_destroy(&queue_lock);
        pthread_attr_destroy(&attr);
        delete[] threads;
    }

    void Queue(TASK* task)
    {pthread_mutex_lock(&queue_lock);
    if(inactive_threads) pthread_cond_signal(&todo_condition);
    queue.Enqueue(task);
    pthread_mutex_unlock(&queue_lock);}

    static void* Thread_Routine(void* data)
    {
        PTHREAD_QUEUE& queue=*(PTHREAD_QUEUE*)data;
        while(1){
            pthread_mutex_lock(&queue.queue_lock);
            while(queue.queue.Empty()){
                queue.active_threads--;
                if(queue.active_threads==0) pthread_cond_signal(&queue.done_condition);
                queue.inactive_threads++;
                pthread_cond_wait(&queue.todo_condition,&queue.queue_lock);
                queue.active_threads++;queue.inactive_threads--;}
            TASK* work=(TASK*)queue.queue.Dequeue();
            pthread_mutex_unlock(&queue.queue_lock);
            work->Run();
            delete work;
        }
        return 0;
    }

    void Wait()
    {
        pthread_mutex_lock(&queue_lock);
        while(!queue.Empty() || active_threads!=0) pthread_cond_wait(&done_condition,&queue_lock);
        pthread_mutex_unlock(&queue_lock);
    }
    

//#####################################################################
};
}
#endif
