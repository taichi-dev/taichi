//#####################################################################
// Copyright 2004-2007, Ron Fedkiw, Geoffrey Irving, Eftychios Sifakis, Rachel Weinstein.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class POINTER_QUEUE
//#####################################################################
#ifndef __POINTER_QUEUE__
#define __POINTER_QUEUE__

#include <cassert>

namespace PhysBAM{

class POINTER_QUEUE
{
public:
    const int array_size;
    void** const array;
    int front,back;

    explicit POINTER_QUEUE(const int size)
         :array_size(size+1),array(new void*[size+1]-1),front(1),back(1)
    {}

    ~POINTER_QUEUE()
    {delete[] (array+1);}

    void Enqueue(void* const element)
    {array[back]=element;if(++back>array_size) back=1;
    assert(back!=front);} // dies if you run out of room

    void* Dequeue()
    {assert(!Empty());int index=front;if(++front>array_size) front=1;return array[index];}

    void* Peek() const
    {return array[front];}

    void* operator()(const int i)
    {assert(i<=Size());int index=front+i-1;if(index>array_size) index=index-array_size;return array[index];}

    int Size() const
    {if(back<front) return back+array_size-front;else return back-front;}

    bool Empty() const
    {return back==front;}

    bool Full() const
    {return Size()==array_size-1;}

    void Remove_All_Entries()
    {front=1;back=1;}

//#####################################################################
};
}
#endif
