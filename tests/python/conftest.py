# let pytest see the fixtures
import taichi.testing

taichi_archs = taichi.testing._get_taichi_archs_fixture()
