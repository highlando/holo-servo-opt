#!/bin/bash
# use this script by calling `source get-upd-gh-deps.sh` in the console

mkdir matlibplots
cd matlibplots
rm conv_plot_utils.py
wget https://raw.githubusercontent.com/highlando/mat-lib-plots/master/conv_plot_utils.py
touch __init__.py
cd ..
