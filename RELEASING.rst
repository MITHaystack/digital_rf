====================
Releasing digital_rf
====================

Releases and all accompanying changes (version bumping, changelog, tagging, etc.) are handled with `rever <https://regro.github.io/rever-docs/>`_. The ``rever`` configuration particular to ``digital_rf`` can be found in the `rever.xsh <rever.xsh>`_ file.

Before making a release, check the following:
  * Make sure that all tests pass, locally and through the CI hooks on GitHub
  * Increment the library version for ``libdigital_rf`` in `c/include/digital_rf_version.h <c/include/digital_rf_version.h>`_ if there have been any feature additions or breaking changes since the last release

To make a new release, switch to a new git branch and run the ``rever`` command from the package base directory::

    rever VERSION

This will do the following:
  * Change the version number where it is hard-coded in the source files
  * Generate a list of authors since the last release
  * Update the bibtex entry for citing the new release
  * Merge and remove files in the 'news' directory to update the changelog

From there, make a pull request to the upstream Digital RF repository and merge it if all checks pass. Create a release on the main branch with a tag equal to the version number. This will result in a tarball being added to the release and the wheel build CI running and uploading artifacts to PyPI.
