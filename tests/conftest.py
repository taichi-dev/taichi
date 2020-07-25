# let pytest see our fixtures & configuators
from taichi.testing import get_conftest

get_conftest(globals())
