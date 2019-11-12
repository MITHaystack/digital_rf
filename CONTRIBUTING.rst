==========================
Contributing to digital_rf
==========================

Bug Reports and Feature Requests
================================

Please submit bug reports and feature requests to the GitHub issue tracker at https://github.com/MITHaystack/digital_rf/issues.


Code Contributions
==================

Code contributions are welcome in the form of pull requests to the ``master`` branch on GitHub at https://github.com/MITHaystack/digital_rf/pulls. Commits should be made to a feature branch off of the current ``master``. Please follow the coding style specified below. If the change is newsworthy, include a file in the `news <news>`_ directory as documented below.


Coding Style
============

For Python code, we strive to follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ with the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format for docstrings. In particular, we strongly prefer the use of `Black <https://black.readthedocs.io/en/stable/>`_ for automatically formatting code into a uniform style. Please run this formatter on all Python code before submitting a pull request.

Hooks using `pre-commit <https://pre-commit.com/>`_ for automatically running code checks including ``black`` are included in the git repository. To add them to your git hooks that run pre-push, install ``pre-commit`` and then run::

    pre-commit install -t pre-push


News Items for Changes
======================

If a commit or pull request introduces a feature, change, or fix that should be mentioned in the changelog, please add a news file documenting the change(s) following the `rever news workflow <https://regro.github.io/rever-docs/news.html>`_. First, copy the `TEMPLATE.rst <news/TEMPLATE.rst>`_ file in the `news <news>`_ directory and rename it to a unique filename (using, for instance, the branch name). Then edit the news file to add entries for the changes under the appropriate headings. Finally, commit the news file and include it in your pull request. When a new release is made, these news files will be merged and added to `CHANGELOG.rst <CHANGELOG.rst>`_.
