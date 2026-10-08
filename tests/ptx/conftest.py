import os


def pytest_addoption(parser):
    group = parser.getgroup("IB-WMLES PTX regression")
    group.addoption("--simfinity-repo", default=os.environ.get("SIMFINITY_MONO"))
    group.addoption("--ptx-mode", choices=("reference", "optimized", "verify"), default="optimized")
    group.addoption("--record-ptx-gold", action="store_true", default=False)
    group.addoption("--ptx-profile", action="store_true", help="Print native pass timings for each target kernel")
    group.addoption("--ptx-verify-forwarding", action="store_true", help="Compare each indexed forwarding lookup with the original search")
