#include "Constraints.h"

MemoryAllocator<Contact, 8196> contactMemoryAllocator;

void *Contact::operator new(size_t _) {
    return contactMemoryAllocator.Allocate();
}

void Contact::operator delete(void *p, size_t _) {
    if (p != NULL)
        contactMemoryAllocator.Dispose(p);
}
