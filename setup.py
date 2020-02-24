# coding: utf-8

from setuptools import setup, find_packages

from distutils.command.build_py import build_py

import os

with open("README.md", encoding="utf8") as file:
    long_description = file.read()

with open("spreg/__init__.py", "r") as f:
    exec(f.readline())


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, "r") as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def setup_package():
    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    _groups_files = {
        "base": "requirements.txt",
        "tests": "requirements_tests.txt",
        "plus": "requirements_plus.txt",
        "docs": "requirements_docs.txt",
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop("base")
    extras_reqs = reqs

    setup(
        name="spreg",
        version=__version__,
        description="PySAL Spatial Econometrics Package",
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        maintainer="PySAL Developers",
        maintainer_email="pysal-dev@googlegroups.com",
        url="http://pysal.org",
        download_url="https://pypi.python.org/pypi/spreg",
        license="BSD",
        py_modules=["spreg"],
        packages=find_packages(),
        keywords="spatial statistics",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        # package_data={'libpysal':list(example_data_files)},
        install_requires=install_reqs,
        extras_require=extras_reqs,
        cmdclass={"build_py": build_py},
    )


if __name__ == "__main__":
    setup_package()

