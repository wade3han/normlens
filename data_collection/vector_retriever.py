"""Base vector store index query."""
from pathlib import Path
from typing import List, Optional

from llama_index import QueryBundle, StorageContext, load_index_from_storage
from llama_index.data_structs import NodeWithScore, IndexDict
from llama_index.indices.utils import log_vector_store_query_result
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.vector_stores import FaissVectorStore
from llama_index.vector_stores.types import VectorStoreQuery


class FaissVectorIndexRetriever(VectorIndexRetriever):
    """Vector index retriever.

    Args:
        index (GPTVectorStoreIndex): vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    @llm_token_counter("retrieve")
    def _retrieve(
            self,
            query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None:
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            doc_ids=self._doc_ids,
            query_str=query_bundle.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
        )
        query_result = self._vector_store.query(query, **self._kwargs)

        # NOTE: vector store does not keep text and returns node indices.
        # Need to recover all nodes from docstore
        if query_result.ids is None:
            raise ValueError(
                "Vector store query result should return at "
                "least one of nodes or ids."
            )
        assert isinstance(self._index.index_struct, IndexDict)
        node_ids = [
            self._doc_ids[int(idx)] for idx in query_result.ids
        ]
        nodes = self._docstore.get_nodes(node_ids)
        query_result.nodes = nodes

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node, score=score))

        return node_with_scores


def get_retriever(root_dir):
    datatypes = ['sherlock', 'coco', 'narratives']
    retrievers = {}
    for datatype in datatypes:
        if datatype == 'sherlock':
            datapath = f'{root_dir}/sherlock_dataset/sherlock_train_v1_1.json'
        elif datatype == 'narratives':
            datapath = f'{root_dir}/openimages_localized_narratives/open_images_train_v6_captions.jsonl'
        elif datatype == 'coco':
            datapath = f'{root_dir}/coco/dataset_coco.json'
        else:
            raise NotImplementedError

        try:
            persist_dir = str(Path(datapath).parent / f'{datatype}_index')

            vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
            index = load_index_from_storage(storage_context=storage_context)

            retriever = FaissVectorIndexRetriever(index,
                                                  doc_ids=list(index.index_struct.nodes_dict.values()),
                                                  similarity_top_k=10)
            retrievers[datatype] = retriever
        except Exception as e:
            print(f'Failed to load {datatype} retriever, {e}')
    return retrievers
