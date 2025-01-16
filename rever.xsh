import datetime
import os

import rever
import xonsh

$PROJECT = "digital_rf"
$WEBSITE_URL = "https://github.com/MITHaystack/digital_rf"
$GITHUB_ORG = "MITHaystack"
$GITHUB_REPO = "digital_rf"

$ACTIVITIES = [
    "version_bump",
    "authors",
    "bibtex",
    "changelog",
]

$VERSION_BUMP_PATTERNS = [
    (
        "CMakeLists.txt",
        r"set\(digital_rf_VERSION [0-9\.]*\)",
        "set(digital_rf_VERSION $VERSION)",
    ),
    (
        "README.rst",
        r"Volz, R.* \(Version .*\).*",
        (lambda ver: "Volz, R., Rideout, W. C., Swoboda, J., Vierinen, J. P., & Lind, F. D. ({yr}). Digital RF (Version {ver}). MIT Haystack Observatory. Retrieved from https://github.com/MITHaystack/digital_rf".format(ver=ver, yr=datetime.datetime.now().year)),
    ),
]

$AUTHORS_TEMPLATE = """\
History
=======

The Digital RF project was started in 2014 by Juha Vierinen, Frank Lind, and
Bill Rideout to provide a disk storage and archival format for radio signals
for use in projects at MIT Haystack Observatory. The concept was born out of
years of experience in collecting and wrangling RF data and incorporates many
lessons learned over that time. In 2017, the project was officially released to
the community under an open source BSD license.

People
======

{authors}

"""
$AUTHORS_FORMAT = "- {name} ({email})\n"

$BIBTEX_PROJECT_NAME = "Digital RF"
$BIBTEX_AUTHORS = [
    "Volz, Ryan",
    "Rideout, William C.",
    "Swoboda, John",
    "Vierinen, Juha P.",
    "Lind, Frank D.",
]

$CHANGELOG_FILENAME = "CHANGELOG.rst"
$CHANGELOG_TEMPLATE = "TEMPLATE.rst"
