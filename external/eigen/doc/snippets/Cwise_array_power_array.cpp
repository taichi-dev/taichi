Array<double,1,3> x(8,25,3),
                  e(1./3.,0.5,2.);
cout << "[" << x << "]^[" << e << "] = " << x.pow(e) << endl; // using ArrayBase::pow
cout << "[" << x << "]^[" << e << "] = " << pow(x,e) << endl; // using Eigen::pow
