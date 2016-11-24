/** @file assert.h
 */
#ifndef AI_DEBUG_H_INC
#define AI_DEBUG_H_INC

#ifdef ASSIMP_BUILD_DEBUG  
#	include <assert.h>
#	define	ai_assert(expression) assert(expression)
#else
#	define	ai_assert(expression)
#endif


#endif
