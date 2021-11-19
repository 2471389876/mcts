#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

os.remove("logs/trace.log")
os.system("python compare_always_qianyi.py &")
os.system("python compare_never_qianyi.py &")
os.system("python compare_qifa.py &")
os.system("python compare_random_qianyi.py &")