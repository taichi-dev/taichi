VectorXd v(10);
v.resize(3);
RowVector3d w;
w.resize(3); // this is legal, but has no effect
cout << "v: " << v.rows() << " rows, " << v.cols() << " cols" << endl;
cout << "w: " << w.rows() << " rows, " << w.cols() << " cols" << endl;
