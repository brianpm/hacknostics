---
layout: post
author: brianpm
title:  "New Examples for Pivoters"
date:   2019-10-30 01:00:00
categories: update, NCL, python, pivot
---

Several important `hacknostics` updates.

First is a restructuring of the repository. We now have a dedicated Examples directory. This directory is where we are going to put examples that have been adapted from NCL's examples to python. Within the Examples directory, please check out the [README][https://github.com/brianpm/hacknostics/blob/master/Examples/README.md] file; I am trying to include the high-level documentation of each example in that file. Besides the README, I have separated the Examples into [Notebooks][https://github.com/brianpm/hacknostics/tree/master/Examples/Notebooks] and [scripts][https://github.com/brianpm/hacknostics/tree/master/Examples/scripts]. That pretty well describes what's inside those directories. I'm doing a mix of both Jupyter Notebooks and plain py scripts. I use both formats depending on how I feel when I decide to write the code. Notebooks are great for exploration and interactivity. Many times, however, I find that simple scripts are easier for more complex and well-defined workflows. I also think that for those who are pivoting to python from NCL, seeing simple scripts that are *very* similar to the NCL ones will be helpful.


Second, along with restructuring, I have added several new examples to the Examples directory. Following the NCL examples pages, I have recently added `explore_grib.py` which was developed while I was trying to follow some of the `latlon_subset` NCL examples that used a GRIB file. This script shows some important differences between the python/xarray GRIB interface and NCL's. They both work well, but both have strengths and weaknesses. I think the python/xarray interface is a little easier because it doesn't muck with variable names, but it does require some exploration of the file to figure out the correct `typeOfLevel`. In the Notebooks, I've also added the NCL scatter plot examples, a notebook showing NCL's shapefile examples, and two of NCL's climate indices examples. As in NCL's examples, there are some useful general nuggets in most of these, which I have tried to point out in the README. At some point I will try to more systematically document the features demonstrated in different scripts. 

Third, I set up this blog for hacknostics. My hope is to provide some detailed descriptions of examples and some commentary that might be useful for those transitioning to python. The blog uses Jekyll as a static site generator, which is convenient because GitHub builds the site for me whenever I push new posts. I just got it working, so I will probably need to do some tinkering.

