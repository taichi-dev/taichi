Vector3i v = Vector3i::Random();
cout << "Here is the vector v:" << endl << v << endl;
cout << "v.rowwise().replicate(5) = ..." << endl;
cout << v.rowwise().replicate(5) << endl;
