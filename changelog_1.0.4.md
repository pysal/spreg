# Changes

Version 1.0.4 (2018-08-24)

We closed a total of 16 issues (enhancements and bug fixes) through 7 pull requests, since our last release on 2018-05-11.

## Main Release Enhancements. 

This release adds additional estimators and tests for [seemingly-unrelated regression](https://en.wikipedia.org/wiki/Seemingly_unrelated_regressions) models with endogenous spatial lag & spatial error structures. These methods in `spreg.SURerrorGM`, and `spreg.SURerrorML` have also been extended to incorporate spatial regimes regression. 

## Issues Closed
  - Libpysal refresh (#11)
  - update docstrings for libpysal API changes (#9)
  - Merging in spanel & spreg2 code necessary for new spatial panel & GeoDaSpace (#10)
  - move to silence_warnings from current libpysal (#7)
  - add init to ensure tests are shipped (#6)
  - weights typechecking will only accept things from `pysal`.  (#3)
  - relax error checking in check_weights (#4)
  - simplify testing (#5)
  - Convert spreg to common subset 2,3 code (#2)

## Pull Requests
  - Libpysal refresh (#11)
  - Merging in spanel & spreg2 code necessary for new spatial panel & GeoDaSpace (#10)
  - move to silence_warnings from current libpysal (#7)
  - add init to ensure tests are shipped (#6)
  - relax error checking in check_weights (#4)
  - simplify testing (#5)
  - Convert spreg to common subset 2,3 code (#2)

The following individuals contributed to this release:

- Luc Anselin (@lanselin)
- Pedro V. Amaral (@pedrovma)

  - Levi John Wolf (@levijohnwolf)