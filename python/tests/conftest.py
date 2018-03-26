# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------


def pytest_collection_modifyitems(items):
    selected_items = []

    for item in items:
        firstonly = item.get_marker('firstonly')
        if firstonly is not None:
            for param in firstonly.args:
                idx = item.callspec.indices.get(param, None)
                if idx is not None and idx > 0:
                    break
            else:
                selected_items.append(item)
        else:
            selected_items.append(item)

    items[:] = selected_items
