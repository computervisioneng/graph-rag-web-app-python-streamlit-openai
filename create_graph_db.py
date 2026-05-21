__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import asyncio
import os

import graphrag.api as api

from utils import DATA_PATH, get_config


for dir_ in os.listdir(DATA_PATH):
	dir_path = os.path.join(DATA_PATH, dir_)

	# create config

	#  default graphrag config
	config = get_config(dir_)

	# build index
	outputs = asyncio.run(
		api.build_index(config)
	)
