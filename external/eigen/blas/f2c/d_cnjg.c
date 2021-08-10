#include "datatypes.h"    

void d_cnjg(doublecomplex *r, doublecomplex *z) {
    r->r = z->r;
    r->i = -(z->i);
}
