import os


def pytest_addoption(parser):
    group = parser.getgroup("IB-WMLES PTX regression")
    group.addoption("--simfinity-repo", default=os.environ.get("SIMFINITY_MONO"))
    group.addoption("--ptx-mode", choices=("reference", "optimized", "verify"), default="optimized")
    group.addoption("--record-ptx-gold", action="store_true", default=False)
