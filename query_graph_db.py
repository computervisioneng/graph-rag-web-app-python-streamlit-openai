__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import asyncio
from typing import Any
import os

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag.data_model.data_reader import DataReader
import graphrag.api as api

from utils import get_config


def extract_info(config):

	dataframe_dict = _resolve_output_files(
		config=config,
		output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
		optional_list=["covariates"]
	)

	communities = dataframe_dict["communities"]
	community_reports = dataframe_dict["community_reports"]
	text_units = dataframe_dict["text_units"]
	relationships = dataframe_dict["relationships"]
	entities = dataframe_dict["entities"]
	covariates = dataframe_dict["covariates"]

	return communities, community_reports, text_units, relationships, entities, covariates


def get_local_response(config, query, communities, community_reports, text_units, relationships,
		                                            entities, covariates):

	local_response, context_data_local_response = asyncio.run(
		api.query.local_search(
			config=config,
			entities=entities,
			relationships=relationships,
			text_units=text_units,
			community_reports=community_reports,
			communities=communities,
			covariates=covariates,
			community_level=2,
			query=query,
			response_type="Multiple Paragraphs"
		)
	)
	return local_response, context_data_local_response


def get_global_response(config, query, communities, community_reports, entities):
	global_response, context_data_global_response = asyncio.run(
		api.query.global_search(
			config=config,
			entities=entities,
			community_reports=community_reports,
			communities=communities,
			query=query,
			response_type="Multiple Paragraphs",
			community_level=2,
			dynamic_community_selection=False
		)
	)
	return global_response, context_data_global_response


def create_graphrag_response(config, query, mode="local"):

	communities, community_reports, text_units, relationships, entities, covariates = extract_info(config)

	if mode in ['local']:
		response, context_data = get_local_response(config, query, communities, community_reports, text_units,
		                                            relationships, entities, covariates)
	elif mode in ['global']:
		response, context_data = get_global_response(config, query, communities, community_reports, entities)

	return response, context_data


def _resolve_output_files(
	config: GraphRagConfig,
	output_list: list[str],
	optional_list: list[str] | None = None,
) -> dict[str, Any]:
	"""Read indexing output files to a dataframe dict, with correct column types."""
	dataframe_dict = {}
	storage_obj = create_storage(config.output_storage)
	table_provider = create_table_provider(config.table_provider, storage=storage_obj)
	reader = DataReader(table_provider)
	for name in output_list:
		df_value = asyncio.run(getattr(reader, name)())
		dataframe_dict[name] = df_value

	# for optional output files, set the dict entry to None instead of erroring out if it does not exist
	if optional_list:
		for optional_file in optional_list:
			file_exists = asyncio.run(table_provider.has(optional_file))
			if file_exists:
				df_value = asyncio.run(getattr(reader, optional_file)())
				dataframe_dict[optional_file] = df_value
			else:
				dataframe_dict[optional_file] = None
	return dataframe_dict


if __name__ == "__main__":
	dir_ = "prague_castle"
	query = "Summarize the document."

	# create config
	config = get_config(dir_)

	# load graph data


	# local search
	create_graphrag_response(config, query, mode="local")

	# global search
	# create_graphrag_response(config, query, mode="global")

