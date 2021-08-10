//=============================================================================
// File      : utilities.h
// Created   : mar jun 19 13:18:14 CEST 2001
// Author    : Antoine YESSAYAN, Paul RASCLE, EDF
// Project   : SALOME
// Copyright : EDF 2001
// $Header$
//=============================================================================

/* ---  Definition macros file to print information if _DEBUG_ is defined --- */

# ifndef UTILITIES_H
# define UTILITIES_H

# include <stdlib.h>
//# include <iostream> ok for gcc3.01
# include <iostream>

/* ---  INFOS is always defined (without _DEBUG_): to be used for warnings, with release version --- */

# define HEREWEARE cout<<flush ; cerr << __FILE__ << " [" << __LINE__ << "] : " << flush ;
# define INFOS(chain) {HEREWEARE ; cerr << chain << endl ;}
# define PYSCRIPT(chain) {cout<<flush ; cerr << "---PYSCRIPT--- " << chain << endl ;}

/* --- To print date and time of compilation of current source on stdout --- */

# if defined ( __GNUC__ )
# define COMPILER		"g++" ;
# elif defined ( __sun )
# define COMPILER		"CC" ;
# elif defined ( __KCC )
# define COMPILER		"KCC" ;
# elif defined ( __PGI )
# define COMPILER		"pgCC" ;
# else
# define COMPILER		"undefined" ;
# endif

# ifdef INFOS_COMPILATION
# error INFOS_COMPILATION already defined
# endif
# define INFOS_COMPILATION	{\
					cerr << flush;\
					cout << __FILE__ ;\
					cout << " [" << __LINE__ << "] : " ;\
					cout << "COMPILED with " << COMPILER ;\
					cout << ", " << __DATE__ ; \
					cout << " at " << __TIME__ << endl ;\
					cout << "\n\n" ;\
					cout << flush ;\
				}

# ifdef _DEBUG_

/* --- the following MACROS are useful at debug time --- */

# define HERE cout<<flush ; cerr << "- Trace " << __FILE__ << " [" << __LINE__ << "] : " << flush ;
# define SCRUTE(var) HERE ; cerr << #var << "=" << var << endl ;
# define MESSAGE(chain) {HERE ; cerr << chain << endl ;}
# define INTERRUPTION(code) HERE ; cerr << "INTERRUPTION return code= " << code << endl ; exit(code) ;

# ifndef ASSERT
# define ASSERT(condition) if (!(condition)){ HERE ; cerr << "CONDITION " << #condition << " NOT VERIFIED"<< endl ; INTERRUPTION(1) ;}
# endif /* ASSERT */

#define REPERE cout<<flush ; cerr << "   --------------" << endl << flush ;
#define BEGIN_OF(chain) {REPERE ; HERE ; cerr << "Begin of: " << chain << endl ; REPERE ; }
#define END_OF(chain) {REPERE ; HERE ; cerr << "Normal end of: " << chain << endl ; REPERE ; }



# else /* ifdef _DEBUG_*/

# define HERE
# define SCRUTE(var)
# define MESSAGE(chain)
# define INTERRUPTION(code)

# ifndef ASSERT
# define ASSERT(condition)
# endif /* ASSERT */

#define REPERE
#define BEGIN_OF(chain)
#define END_OF(chain)


# endif /* ifdef _DEBUG_*/

# endif /* ifndef UTILITIES_H */
