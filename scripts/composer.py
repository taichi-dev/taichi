#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
from PyQt4.QtGui import *

# Create an PyQT4 application object.
a = QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = QWidget()

# Set window size.
w.resize(320, 240)

# Set window title
w.setWindowTitle("Taichi Composer")

# Create textbox
textbox = QLineEdit(w)
textbox.move(20, 20)
textbox.resize(280, 20)


combo = QComboBox(w)
combo.addItem("Python")
combo.addItem("Perl")
combo.addItem("Java")
combo.addItem("C++")
combo.move(20, 60)

# Show window
w.show()

sys.exit(a.exec_())