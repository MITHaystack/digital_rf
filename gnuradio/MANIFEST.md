title: gr_drf
brief: Digital RF module for GNU Radio
tags:
  - recording
  - HDF5
author:
  - Ryan Volz <rvolz@mit.edu>
copyright_owner:
  - Massachusetts Institute of Technology
dependencies:
  - gnuradio, libdigital_rf
license: GPLv3
#repo: https://github.com/<>/gr_drf
#website: <module_website> # If you have a separate project website, put it here
#icon: <icon_url> # Put a URL to a square image here that will be used as an icon on CGRAN
---
Read and write files in Digital RF format of HDF5 using GNU Radio.

Digital RF is a disk storage and archival format for radio signals. It uses HDF5
files with embedded metadata and a predictable naming scheme to produce a self-
describing data format suitable to a variety of use cases.
