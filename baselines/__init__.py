"""The baselines of Check 10 (plan 5.5 Check 10): the readings of six subjects
over the same questions, and the record each question gets.

This package is test infrastructure and is not shipped. It reads the table in
``fixture/table/check10.toml`` and holds nothing of the tool: no module of it
imports ``waterwhite``. ``baselines.interfaces`` holds the words, the marks and the
types every other part builds on; ``baselines.table10`` loads the table and expands
its groups into the questions.
"""
