====================
Releasing digital_rf
====================

Releases and all accompanying changes (version bumping, changelog, tagging, etc.) are handled with `rever <https://regro.github.io/rever-docs/>`_. The ``rever`` configuration particular to ``digital_rf`` can be found in the `rever.xsh <rever.xsh>`_ file.

Before making a release, check the following:
  * Make sure that all tests pass, locally and through the CI hooks on GitHub
  * Increment the library version for ``libdigital_rf`` in `c/include/digital_rf_version.h <c/include/digital_rf_version.h>`_ if there have been any feature additions or breaking changes since the last release

To make a new release, run the ``rever`` command from the package base directory::

    rever VERSION

This will do the following:
  * Change the version number where it is hard-coded in the source files
  * Generate a list of authors since the last release
  * Update the bibtex entry for citing the new release
  * Merge and remove files in the 'news' directory to update the changelog
  * Run the Python test suite in a Docker container
  * Create a git tag for the release
  * Push the release tag to Github
  * Create a Github release containing a source archive and a list of changes
  * Upload the Python source distribution to PyPI
  * Submit a pull request to update the conda-forge recipe
