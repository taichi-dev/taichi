Vector3f boxMin(Vector3f::Zero()), boxMax(Vector3f::Ones());
Vector3f p0 = Vector3f::Random(), p1 = Vector3f::Random().cwiseAbs();
// let's check if p0 and p1 are inside the axis aligned box defined by the corners boxMin,boxMax:
cout << "Is (" << p0.transpose() << ") inside the box: "
     << ((boxMin.array()<p0.array()).all() && (boxMax.array()>p0.array()).all()) << endl;
cout << "Is (" << p1.transpose() << ") inside the box: "
     << ((boxMin.array()<p1.array()).all() && (boxMax.array()>p1.array()).all()) << endl;
