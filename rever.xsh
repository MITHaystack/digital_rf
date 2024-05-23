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
#    "tag",
    "pytest",
    "copy_dist",
#    "push_tag",
#    "ghrelease",
#    "pypi_upload",
]

$VERSION_BUMP_PATTERNS = [
    (
        "python/CMakeLists.txt",
        r"set\(digital_rf_VERSION .*\)",
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

$DOCKER_APT_DEPS = [
    "cmake",
    "git",
    "libhdf5-dev",
    "python3-dateutil",
    "python3-dev",
    "python3-h5py",
    "python3-mako",
    "python3-numpy",
    "python3-packaging",
    "python3-pkgconfig",
    "python3-pytest",
    "python3-setuptools",
    "python3-six",
    "python3-tz",
]
$DOCKER_INSTALL_COMMAND = "git clean -fdx && mkdir build-rever && cd build-rever && cmake .. && make && make install && make sdist && cp -a dist $HOME/ && cd .. && rm -rf build-rever"

$PYTEST_COMMAND = "pytest-3"

@rever.activity.activity(deps={"pytest"})
def copy_dist():
    """Copy dist tarballs from install docker container to rever dist directory."""
    dist_dir = os.path.join($REVER_DIR, "dist")
    if os.path.exists(dist_dir):
        xonsh.lib.os.rmtree(dist_dir, force=True)
    $install_image = xonsh.tools.expand_path($DOCKER_INSTALL_IMAGE)
    $install_container_id = $(docker ps -aq --filter ancestor=$install_image).strip()
    docker cp $install_container_id:$DOCKER_HOME/dist $REVER_DIR/
    del $install_image
    del $install_container_id
copy_dist.requires = {"commands": {"docker": "docker"}}

@rever.activity.activity(deps={"version_bump", "copy_dist"})
def pypi_upload():
    """Uploads packages from the rever dist directory to the Python Package Index."""
    $dist_dir = os.path.join($REVER_DIR, "dist")
    p = ![twine upload --sign $dist_dir/*]
    if p.rtn != 0:
        raise RuntimeError("PyPI upload failed!")
    del $dist_dir
pypi_upload.requires = {"commands": {"twine": "twine"}}
