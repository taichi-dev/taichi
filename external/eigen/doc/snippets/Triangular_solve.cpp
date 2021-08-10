Matrix3d m = Matrix3d::Zero();
m.triangularView<Eigen::Upper>().setOnes();
cout << "Here is the matrix m:\n" << m << endl;
Matrix3d n = Matrix3d::Ones();
n.triangularView<Eigen::Lower>() *= 2;
cout << "Here is the matrix n:\n" << n << endl;
cout << "And now here is m.inverse()*n, taking advantage of the fact that"
        " m is upper-triangular:\n"
     << m.triangularView<Eigen::Upper>().solve(n) << endl;
cout << "And this is n*m.inverse():\n"
     << m.triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(n);
