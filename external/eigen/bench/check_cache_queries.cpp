
#define EIGEN_INTERNAL_DEBUG_CACHE_QUERY
#include <iostream>
#include "../Eigen/Core"

using namespace Eigen;
using namespace std;

#define DUMP_CPUID(CODE) {\
  int abcd[4]; \
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;\
  EIGEN_CPUID(abcd, CODE, 0); \
  std::cout << "The code " << CODE << " gives " \
              << (int*)(abcd[0]) << " " << (int*)(abcd[1]) << " " \
              << (int*)(abcd[2]) << " " << (int*)(abcd[3]) << " " << std::endl; \
  }
  
int main()
{
  cout << "Eigen's L1    = " << internal::queryL1CacheSize() << endl;
  cout << "Eigen's L2/L3 = " << internal::queryTopLevelCacheSize() << endl;
  int l1, l2, l3;
  internal::queryCacheSizes(l1, l2, l3);
  cout << "Eigen's L1, L2, L3       = " << l1 << " " << l2 << " " << l3 << endl;
  
  #ifdef EIGEN_CPUID

  int abcd[4];
  int string[8];
  char* string_char = (char*)(string);

  // vendor ID
  EIGEN_CPUID(abcd,0x0,0);
  string[0] = abcd[1];
  string[1] = abcd[3];
  string[2] = abcd[2];
  string[3] = 0;
  cout << endl;
  cout << "vendor id = " << string_char << endl;
  cout << endl;
  int max_funcs = abcd[0];

  internal::queryCacheSizes_intel_codes(l1, l2, l3);
  cout << "Eigen's intel codes L1, L2, L3 = " << l1 << " " << l2 << " " << l3 << endl;
  if(max_funcs>=4)
  {
    internal::queryCacheSizes_intel_direct(l1, l2, l3);
    cout << "Eigen's intel direct L1, L2, L3 = " << l1 << " " << l2 << " " << l3 << endl;
  }
  internal::queryCacheSizes_amd(l1, l2, l3);
  cout << "Eigen's amd L1, L2, L3         = " << l1 << " " << l2 << " " << l3 << endl;
  cout << endl;
  
  // dump Intel direct method
  if(max_funcs>=4)
  {
    l1 = l2 = l3 = 0;
    int cache_id = 0;
    int cache_type = 0;
    do {
      abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
      EIGEN_CPUID(abcd,0x4,cache_id);
      cache_type  = (abcd[0] & 0x0F) >> 0;
      int cache_level = (abcd[0] & 0xE0) >> 5;  // A[7:5]
      int ways        = (abcd[1] & 0xFFC00000) >> 22; // B[31:22]
      int partitions  = (abcd[1] & 0x003FF000) >> 12; // B[21:12]
      int line_size   = (abcd[1] & 0x00000FFF) >>  0; // B[11:0]
      int sets        = (abcd[2]);                    // C[31:0]
      int cache_size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1);
      
      cout << "cache[" << cache_id << "].type       = " << cache_type << "\n";
      cout << "cache[" << cache_id << "].level      = " << cache_level << "\n";
      cout << "cache[" << cache_id << "].ways       = " << ways << "\n";
      cout << "cache[" << cache_id << "].partitions = " << partitions << "\n";
      cout << "cache[" << cache_id << "].line_size  = " << line_size << "\n";
      cout << "cache[" << cache_id << "].sets       = " << sets << "\n";
      cout << "cache[" << cache_id << "].size       = " << cache_size << "\n";
      
      cache_id++;
    } while(cache_type>0 && cache_id<16);
  }
  
  // dump everything
  std::cout << endl <<"Raw dump:" << endl;
  for(int i=0; i<max_funcs; ++i)
    DUMP_CPUID(i);

  DUMP_CPUID(0x80000000);
  DUMP_CPUID(0x80000001);
  DUMP_CPUID(0x80000002);
  DUMP_CPUID(0x80000003);
  DUMP_CPUID(0x80000004);
  DUMP_CPUID(0x80000005);
  DUMP_CPUID(0x80000006);
  DUMP_CPUID(0x80000007);
  DUMP_CPUID(0x80000008);
  #else
  cout << "EIGEN_CPUID is not defined" << endl;
  #endif
  return 0;
}
