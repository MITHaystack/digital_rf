$PROJECT = "digital_rf"
$WEBSITE_URL = "https://github.com/MITHaystack/digital_rf"
$GITHUB_ORG = "MITHaystack"
$GITHUB_REPO = "digital_rf"

$ACTIVITIES = [
    "version_bump",
    "authors",
    "bibtex",
    "changelog",
    "pytest",
#    "sphinx",
    "tag",
    "push_tag",
#    "pypi",
#    "conda_forge",
    "ghrelease",
]

#$VERSION_BUMP_PATTERNS = [
#    ("rever/__init__.py", "__version__\s*=.*", "__version__ = '$VERSION'"),
#    ("setup.py", "version\s*=.*,", "version='$VERSION',")
#]

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

#$DOCKER_APT_DEPS = []
#with open("requirements/tests.txt") as f:
#    $DOCKER_CONDA_DEPS = f.read().split()
#with open("requirements/docs.txt") as f:
#    $DOCKER_CONDA_DEPS += f.read().split()
#$DOCKER_CONDA_DEPS = [d.lower() for d in set($DOCKER_CONDA_DEPS)]
#$DOCKER_PIP_DEPS = ["xonda"]
#$DOCKER_INSTALL_COMMAND = "git clean -fdx && mkdir build-rever && pushd build-rever && cmake .. && make && make install"
