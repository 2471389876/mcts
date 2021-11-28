#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

os.remove("logs/trace.log")
os.system('python algo_compare.py --Algo "never" &')
os.system('python algo_compare.py --Algo "always" &')
os.system('python algo_compare.py --Algo "heuristic" &')
os.system('python algo_compare.py --Algo "random" &')