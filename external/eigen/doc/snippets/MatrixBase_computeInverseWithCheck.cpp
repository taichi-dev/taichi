Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix3d inverse;
bool invertible;
m.computeInverseWithCheck(inverse,invertible);
if(invertible) {
  cout << "It is invertible, and its inverse is:" << endl << inverse << endl;
}
else {
  cout << "It is not invertible." << endl;
}
