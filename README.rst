========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-framestructure/badge/?style=flat
    :target: https://python-framestructure.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/fonganthonym/python-framestructure.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/fonganthonym/python-framestructure

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/fonganthonym/python-framestructure?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/fonganthonym/python-framestructure

.. |requires| image:: https://requires.io/github/fonganthonym/python-framestructure/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/fonganthonym/python-framestructure/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/fonganthonym/python-framestructure/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/fonganthonym/python-framestructure

.. |version| image:: https://img.shields.io/pypi/v/framestructure.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/framestructure

.. |wheel| image:: https://img.shields.io/pypi/wheel/framestructure.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/framestructure

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/framestructure.svg
    :alt: Supported versions
    :target: https://pypi.org/project/framestructure

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/framestructure.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/framestructure

.. |commits-since| image:: https://img.shields.io/github/commits-since/fonganthonym/python-framestructure/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/fonganthonym/python-framestructure/compare/v0.1.0...master



.. end-badges

Objects for organizing data in frame structures.

* Free software: BSD 2-Clause License

Installation
============

::

    pip install framestructure

You can also install the in-development version with::

    pip install https://github.com/fonganthonym/python-framestructure/archive/master.zip


Documentation
=============


https://python-framestructure.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
