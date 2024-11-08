import jieba
from rank_bm25 import BM25Okapi
from typing import Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


def get_hybrid_retrieve_pd(vector_retrieve_pd, bm25_retrieve_pd) -> pd.DataFrame:
    merged_retrieve = pd.merge(bm25_retrieve_pd, vector_retrieve_pd, on=['qid', 'chunk_id', 'file', 'query', 'sentence'],
                               suffixes=('_x', '_y'))
    merged_retrieve['score'] = 0.5 * merged_retrieve['score_x'] + 0.5 * merged_retrieve['score_y']
    merged_retrieve = merged_retrieve.drop(columns=['score_x', 'score_y'])
    return merged_retrieve


def get_retrieve_pd(corpus_dict: dict[int, str],
                    queries: list[str],
                    retrieve: list[list[dict[str, Union[int, float]]]]) -> pd.DataFrame:
    row = []
    for query_id, sentences_info in enumerate(retrieve):
        query = queries[query_id]
        for sentence_info in sentences_info:
            corpus_id = sentence_info['corpus_id']
            score = sentence_info['score']
            sentence = corpus_dict[corpus_id]
            row.append(
                {
                    'qid': query_id,
                    'query': query,
                    'corpus_id': corpus_id,
                    'score': score,
                    'sentence': sentence
                }
            )
    retrieve: pd.DataFrame = pd.DataFrame(row)
    # retrieve['normalized_score'] = retrieve.groupby('qid')['score'].transform(
    #     lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    # )
    return retrieve


class BM25Retrieve:
    @staticmethod
    def retrieve_aux(qs: str, source: list[int], corpus_df: pd.DataFrame) \
            -> tuple[list[float], dict[int, str]]:

        filtered_chunk_to_id: dict[int,str] = corpus_df[corpus_df['file'].isin(source)].set_index('chunk')['chunk_id'].to_dict()
        filtered_chunks = list(filtered_chunk_to_id.keys())
        if len(filtered_chunks) == 0:
            print(f'BM25')
            print(f'source: {source}, filtered_chunks: {filtered_chunks}')
        tokenized_corpus = [list(jieba.cut_for_search(chunk)) for chunk in filtered_chunks]
        tokenized_query = list(jieba.cut_for_search(qs))
        bm25 = BM25Okapi(tokenized_corpus)
        score_list = sorted(bm25.get_scores(tokenized_query), reverse=True)
        score_list = (score_list - np.mean(score_list))/np.std(score_list) if np.std(score_list) != 0 else [0]*len(score_list)
        top_k = len(filtered_chunks)
        top_k_chunks = bm25.get_top_n(tokenized_query, filtered_chunks, top_k)
        top_k_chunks_id = [filtered_chunk_to_id[chunk] for chunk in top_k_chunks]
        top_k_id_to_chunk = dict(zip(top_k_chunks, top_k_chunks_id))
        # print(f'score: {bm25_score_list}'
        #       f'top_n: {bm25_top_n}')

        return score_list, top_k_id_to_chunk

    @staticmethod
    # retrieve top n candidates for all queries
    def retrieve(queries: list[str], sources: list[list[int]], corpus_df: pd.DataFrame) \
            -> pd.DataFrame:
        chunk_to_file_dict = corpus_df.set_index('chunk')['file'].to_dict()
        # retrieved_data_for_all_query: list[list[dict[str, Union[int, float]]]] = []
        row = []
        for i, (query, source) in enumerate(tqdm(zip(queries, sources), total=len(queries)), start=1):
            #print(f'query: {query}, source: {source}')
            score_list, top_k_id_to_chunk = BM25Retrieve.retrieve_aux(query, source, corpus_df)
            for score, (chunk, chunk_id) in zip(score_list, top_k_id_to_chunk.items()):
                row.append(
                    {
                        'qid': i,
                        'chunk_id': chunk_id,
                        'file': chunk_to_file_dict[chunk],
                        'score': score,
                        'query': query,
                        'sentence': chunk
                    }
                )

        retrieved_info_df = pd.DataFrame(row)
        return retrieved_info_df

    # @staticmethod
    # def retrieve_one_sample(qs: str, source: list[int], corpus_dict: dict[int, str]):
    #     _, bm25_top_n = BM25Retrieve.retrieve_aux(qs, source, corpus_dict, top_n=1)
    #     ans_sentence = bm25_top_n[0]
    #     file_key = [key for key, sentence in corpus_dict.items() if sentence == ans_sentence]
    #     return file_key[0]


class VectorRetriever:
    @staticmethod
    def retrieve_aux(embedder, query: str, source: list[int], corpus_df: pd.DataFrame) \
            -> tuple[list[float], list[str]]:
        filtered_chunk_to_id: dict[int,str] = corpus_df[corpus_df['file'].isin(source)].set_index('chunk')['chunk_id'].to_dict()
        filtered_chunks = list(filtered_chunk_to_id.keys())
        corpus_emb = embedder.encode(filtered_chunks, convert_to_tensor=True)
        query_emb = embedder.encode(query, convert_to_tensor=True)
        similarity_scores = embedder.similarity(query_emb, corpus_emb)[0]
        top_k = len(filtered_chunks)
        scores, indices = torch.topk(similarity_scores, k=top_k)
        score_list = list(scores.cpu())
        score_list = (score_list - np.mean(score_list))/np.std(score_list) if np.std(score_list) != 0 else [0]*len(score_list)
        top_k_chunks = [filtered_chunks[index] for index in indices]
        top_k_chunks_id = [filtered_chunk_to_id[chunk] for chunk in top_k_chunks]
        top_k_id_to_chunk = dict(zip(top_k_chunks, top_k_chunks_id))
        
        return score_list, top_k_id_to_chunk

    
    @staticmethod
    # retrieve top n candidates for all queries
    def retrieve(queries: list[str], sources: list[list[int]], corpus_df: pd.DataFrame) \
            -> pd.DataFrame:
        embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        chunk_to_file_dict = corpus_df.set_index('chunk')['file'].to_dict()
        row = []
        for i, (query, source) in enumerate(tqdm(zip(queries, sources), total=len(queries)), start=1):

            # filtered_corpus: list = corpus_df[corpus_df['file'].isin(source)]['chunk'].to_list()
            # if len(filtered_corpus) == 0:
            #     print(f'vec')
            #     print(f'source: {source}, filtered_corpus: {filtered_corpus}')
            
            score_list, id_to_top_k_chunk = VectorRetriever.retrieve_aux(embedder, query=query, source=source, corpus_df=corpus_df)
            for score, (chunk, chunk_id) in zip(score_list, id_to_top_k_chunk.items()):
                row.append(
                    {
                        'qid': i,
                        'chunk_id': chunk_id,
                        'file': chunk_to_file_dict[chunk],
                        'score': score,
                        'query': query,
                        'sentence': chunk
                    }
                )
        retrieved_info_df = pd.DataFrame(row)
        return retrieved_info_df
