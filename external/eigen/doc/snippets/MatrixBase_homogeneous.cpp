Vector3d v = Vector3d::Random(), w;
Projective3d P(Matrix4d::Random());
cout << "v                                   = [" << v.transpose() << "]^T" << endl;
cout << "h.homogeneous()                     = [" << v.homogeneous().transpose() << "]^T" << endl;
cout << "(P * v.homogeneous())               = [" << (P * v.homogeneous()).transpose() << "]^T" << endl;
cout << "(P * v.homogeneous()).hnormalized() = [" << (P * v.homogeneous()).eval().hnormalized().transpose() << "]^T" << endl;