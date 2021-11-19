#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

os.remove("logs/trace.log")
os.system('python test.py --Algo "never" &')
os.system('python test.py --Algo "always" &')
os.system('python test.py --Algo "heuristic" &')
os.system('python test.py --Algo "random" &')