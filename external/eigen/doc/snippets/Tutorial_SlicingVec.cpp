RowVectorXf v = RowVectorXf::LinSpaced(20,0,19);
cout << "Input:" << endl << v << endl;
Map<RowVectorXf,0,InnerStride<2> > v2(v.data(), v.size()/2);
cout << "Even:" << v2 << endl;