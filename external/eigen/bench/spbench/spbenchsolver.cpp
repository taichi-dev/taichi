#include <bench/spbench/spbenchsolver.h>

void bench_printhelp()
{
    cout<< " \nbenchsolver : performs a benchmark of all the solvers available in Eigen \n\n";
    cout<< " MATRIX FOLDER : \n";
    cout<< " The matrices for the benchmark should be collected in a folder specified with an environment variable EIGEN_MATRIXDIR \n";
    cout<< " The matrices are stored using the matrix market coordinate format \n";
    cout<< " The matrix and associated right-hand side (rhs) files are named respectively \n";
    cout<< " as MatrixName.mtx and MatrixName_b.mtx. If the rhs does not exist, a random one is generated. \n";
    cout<< " If a matrix is SPD, the matrix should be named as MatrixName_SPD.mtx \n";
    cout<< " If a true solution exists, it should be named as MatrixName_x.mtx; \n"     ;
    cout<< " it will be used to compute the norm of the error relative to the computed solutions\n\n";
    cout<< " OPTIONS : \n"; 
    cout<< " -h or --help \n    print this help and return\n\n";
    cout<< " -d matrixdir \n    Use matrixdir as the matrix folder instead of the one specified in the environment variable EIGEN_MATRIXDIR\n\n"; 
    cout<< " -o outputfile.xml \n    Output the statistics to a xml file \n\n";
    cout<< " --eps <RelErr> Sets the relative tolerance for iterative solvers (default 1e-08) \n\n";
    cout<< " --maxits <MaxIts> Sets the maximum number of iterations (default 1000) \n\n";
    
}
int main(int argc, char ** args)
{
  
  bool help = ( get_options(argc, args, "-h") || get_options(argc, args, "--help") );
  if(help) {
    bench_printhelp();
    return 0;
  }

  // Get the location of the test matrices
  string matrix_dir;
  if (!get_options(argc, args, "-d", &matrix_dir))
  {
    if(getenv("EIGEN_MATRIXDIR") == NULL){
      std::cerr << "Please, specify the location of the matrices with -d mat_folder or the environment variable EIGEN_MATRIXDIR \n";
      std::cerr << " Run with --help to see the list of all the available options \n";
      return -1;
    }
    matrix_dir = getenv("EIGEN_MATRIXDIR");
  }
     
  std::ofstream statbuf;
  string statFile ;
  
  // Get the file to write the statistics
  bool statFileExists = get_options(argc, args, "-o", &statFile);
  if(statFileExists)
  {
    statbuf.open(statFile.c_str(), std::ios::out);
    if(statbuf.good()){
      statFileExists = true; 
      printStatheader(statbuf);
      statbuf.close();
    }
    else
      std::cerr << "Unable to open the provided file for writting... \n";
  }       
  
  // Get the maximum number of iterations and the tolerance
  int maxiters = 1000; 
  double tol = 1e-08; 
  string inval; 
  if (get_options(argc, args, "--eps", &inval))
    tol = atof(inval.c_str()); 
  if(get_options(argc, args, "--maxits", &inval))
    maxiters = atoi(inval.c_str()); 
  
  string current_dir; 
  // Test the real-arithmetics matrices
  Browse_Matrices<double>(matrix_dir, statFileExists, statFile,maxiters, tol);
  
  // Test the complex-arithmetics matrices
  Browse_Matrices<std::complex<double> >(matrix_dir, statFileExists, statFile, maxiters, tol); 
  
  if(statFileExists)
  {
    statbuf.open(statFile.c_str(), std::ios::app); 
    statbuf << "</BENCH> \n";
    cout << "\n Output written in " << statFile << " ...\n";
    statbuf.close();
  }

  return 0;
}

      
