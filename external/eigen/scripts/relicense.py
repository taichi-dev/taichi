# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2012 Keir Mierle <mierle@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: mierle@gmail.com (Keir Mierle)
#
# Make the long-awaited conversion to MPL.

lgpl3_header = '''
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.
'''

mpl2_header = """
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import os
import sys

exclusions = set(['relicense.py'])

def update(text):
  if text.find(lgpl3_header) == -1:
    return text, False
  return text.replace(lgpl3_header, mpl2_header), True

rootdir = sys.argv[1]
for root, sub_folders, files in os.walk(rootdir):
    for basename in files:
        if basename in exclusions:
          print 'SKIPPED', filename
          continue
        filename = os.path.join(root, basename)
        fo = file(filename)
        text = fo.read()
        fo.close()

        text, updated = update(text)
        if updated:
          fo = file(filename, "w")
          fo.write(text)
          fo.close()
          print 'UPDATED', filename
        else:
          print '       ', filename
