import libpysal


def pytest_configure(config):
    # Fetch the remote example datasets once, before pytest-xdist starts its
    # workers. Otherwise several workers download the same dataset at the same
    # time and some of them read it before it is complete.
    if hasattr(config, "workerinput"):
        return
    for name in ("NCOVR", "Natregimes", "chicagoSDOH"):
        libpysal.examples.load_example(name)
