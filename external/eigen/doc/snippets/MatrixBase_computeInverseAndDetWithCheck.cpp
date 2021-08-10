Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix3d inverse;
bool invertible;
double determinant;
m.computeInverseAndDetWithCheck(inverse,determinant,invertible);
cout << "Its determinant is " << determinant << endl;
if(invertible) {
  cout << "It is invertible, and its inverse is:" << endl << inverse << endl;
}
else {
  cout << "It is not invertible." << endl;
}
