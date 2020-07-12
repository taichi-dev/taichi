# Start in pytest mode:
import sys
sys._called_from_pytest = True

# let pytest see our fixtures & configuators
from taichi.testing import *
