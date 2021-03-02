============================================================
SLSDT (Stochastic Local Search Decision Tree) documentation
============================================================

Oblique decision tree using the LAHC heuristic. 

Decision tree is a predictive modelling aproach used in machine learning, data mining and statistics. In the decision tree each internal node represents a test on a feature and each terminal (or leaf) node represents a class label. Oblique Decision Tree is a variation of traditional decision trees, which allows multivariate tests in its internal nodes in the form of a combination of the features.

Our research, SLSDT, is a method for induction oblique decision trees using a stochastic local search method called Late Acceptance Hill-Climbing (LAHC) to try to find the best combination of features in each internal node.

This project also provides a utility to read csv files and convert to the format accepted by the SLSDT method.

Please, see also our :ref:`intro-install` and our :ref:`quickstart` for a quick start.

:ref:`license` is EPL 2.0.

.. toctree::
   :caption: Getting started
   :hidden:

   intro/install
   intro/quickstart

.. toctree::
   :maxdepth: 2
   :hidden:

   api_reference
   license