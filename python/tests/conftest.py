# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "firstonly(fixturename1, fixturename2, ...): Generate a test only for the first parameter of the listed fixtures.",
    )


def pytest_collection_modifyitems(items):
    selected_items = []

    for item in items:
        for firstonly in item.iter_markers("firstonly"):
            for param in firstonly.args:
                # check if specified param is not first and skip
                idx = item.callspec.indices.get(param, None)
                if idx is not None and idx > 0:
                    break
            else:
                # if item is not skipped on the basis of this set of params,
                # move on to next mark in loop
                continue
            # if we broke out of the previous for loop and are skipping the
            # item, we have to break out of this loop too
            break
        else:
            selected_items.append(item)

    items[:] = selected_items
