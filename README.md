# Fiscal_Rules_with_Discretion_Economic_Union
This repository contains the code to compute the examples in Section 5 of the following paper: 
Guillaume Sublet (2022) "Fiscal Rules with Discretion for an Economic Union".

The code is in [Python](https://www.python.org).

Structure of the repository:
----------------------------
* `src` contains the source code which defines the class Class_FiscRule.py
* `Execution_non_financial_sanctions.ipynb` is a Jupyter notebook that runs Class_FiscRule.py to produce Figure 1
* `Execution_financial_sanctions.ipynb` is a Jupyter notebook that runs Class_FiscRule.py to produce Figure 2
* `output` contains the figures produced by running the `Execution_...` codes

To run the code:
----------------
The [QuantEcon](https://quantecon.org/quantecon-py/) library is needed for the interpolation routine. The core Python library is [Anaconda](https://www.anaconda.com/products/individual).

The code runs in few seconds on a personal computer.

For comments and questions, please email guillaume.sublet@gmail.com.
