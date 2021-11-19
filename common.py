#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os

CurrentPath = os.path.abspath(__file__)
RootPath = os.path.dirname(CurrentPath)
LogsPath = RootPath + "/logs"

LoggingPath = LogsPath + "/trace.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=LoggingPath, level=logging.DEBUG, format=LOG_FORMAT)