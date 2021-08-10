/* This file has been modified to use the standard gfortran calling
   convention, rather than the f2c calling convention.

   It does not require -ff2c when compiled with gfortran.
*/

/* complexdots.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "datatypes.h"

complex cdotc_(integer *n, complex *cx, integer 
	*incx, complex *cy, integer *incy)
{
    complex res;
    extern /* Subroutine */ int cdotcw_(integer *, complex *, integer *, 
	    complex *, integer *, complex *);

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    cdotcw_(n, &cx[1], incx, &cy[1], incy, &res);
    return res;
} /* cdotc_ */

complex cdotu_(integer *n, complex *cx, integer 
	*incx, complex *cy, integer *incy)
{
    complex res;
    extern /* Subroutine */ int cdotuw_(integer *, complex *, integer *, 
	    complex *, integer *, complex *);

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    cdotuw_(n, &cx[1], incx, &cy[1], incy, &res);
    return res;
} /* cdotu_ */

doublecomplex zdotc_(integer *n, doublecomplex *cx, integer *incx, 
                     doublecomplex *cy, integer *incy)
{
    doublecomplex res;
    extern /* Subroutine */ int zdotcw_(integer *, doublecomplex *, integer *,
	     doublecomplex *, integer *, doublecomplex *);

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    zdotcw_(n, &cx[1], incx, &cy[1], incy, &res);
    return res;
} /* zdotc_ */

doublecomplex zdotu_(integer *n, doublecomplex *cx, integer *incx, 
                     doublecomplex *cy, integer *incy)
{
    doublecomplex res;
    extern /* Subroutine */ int zdotuw_(integer *, doublecomplex *, integer *,
	     doublecomplex *, integer *, doublecomplex *);

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    zdotuw_(n, &cx[1], incx, &cy[1], incy, &res);
    return res;
} /* zdotu_ */

