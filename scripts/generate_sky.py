import sys
import array
import OpenEXR
import Imath
import numpy as np
import cv2

# Open the input file
file = OpenEXR.InputFile(sys.argv[1])

# Compute the size
dw = file.header()['dataWindow']
sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
print(sz)

# Read the three color channels as 32-bit floats
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
(R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

R = np.array(R).reshape(sz)
G = np.array(G).reshape(sz)
B = np.array(B).reshape(sz)
img = np.stack([R, G, B], axis=2)[:sz[0] // 2]
cv2.imshow('sky', img[:, :, ::-1])
cv2.waitKey()

