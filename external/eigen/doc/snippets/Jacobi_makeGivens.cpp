Vector2f v = Vector2f::Random();
JacobiRotation<float> G;
G.makeGivens(v.x(), v.y());
cout << "Here is the vector v:" << endl << v << endl;
v.applyOnTheLeft(0, 1, G.adjoint());
cout << "Here is the vector J' * v:" << endl << v << endl;