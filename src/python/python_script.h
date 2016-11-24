#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#endif
#ifndef _DEBUG
#include <Python.h>
#endif
#include <string>
#include <vector>

TC_NAMESPACE_BEGIN

using std::string;
using std::vector;

int call_python_function(string arg1, string arg2, vector<string> arguments) {
	PyObject *pName, *pModule, *pFunc;
	PyObject *pArgs, *pValue;

	Py_Initialize();
	PyObject* sysPath = PySys_GetObject((char*)"path");
	PyObject* programName = PyString_FromString("C:/Users/koshi/Documents/Visual Studio 2015/Projects/Taichi/scripts/");
	PyList_Append(sysPath, programName);
	Py_DECREF(programName);

	pName = PyString_FromString(arg1.c_str());
	/* Error checking of pName left out */

	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	if (pModule != NULL) {
		pFunc = PyObject_GetAttrString(pModule, arg2.c_str());
		/* pFunc is a new reference */

		if (pFunc && PyCallable_Check(pFunc)) {
			pArgs = PyTuple_New((int)arguments.size());
			for (int i = 0; i < (int)arguments.size(); ++i) {
				pValue = PyInt_FromLong(atoi(arguments[i].c_str()));
				if (!pValue) {
					Py_DECREF(pArgs);
					Py_DECREF(pModule);
					fprintf(stderr, "Cannot convert argument\n");
					return 1;
				}
				/* pValue reference stolen here: */
				PyTuple_SetItem(pArgs, i, pValue);
			}
			pValue = PyObject_CallObject(pFunc, pArgs);
			Py_DECREF(pArgs);
			if (pValue != NULL) {
				printf("Result of call: %ld\n", PyInt_AsLong(pValue));
				Py_DECREF(pValue);
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr, "Call failed\n");
				return 1;
			}
		}
		else {
			if (PyErr_Occurred())
				PyErr_Print();
			fprintf(stderr, "Cannot find function \"%s\"\n", arg2.c_str());
		}
		Py_XDECREF(pFunc);
		Py_DECREF(pModule);
	}
	else {
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", arg1.c_str());
		return 1;
	}
	Py_Finalize();
	return 0;
}

void test_python() {
	int ret = call_python_function("test", "multiply", { "2", "3" });
	getchar();
	exit(0);
}

TC_NAMESPACE_END
