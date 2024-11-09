import jieba
from rank_bm25 import BM25Okapi
from typing import Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from utils.Loader import *

def generate_answer_json(top_1_retrieve_df_list: list[pd.DataFrame]) -> dict[int ,int]:
    answers_dict = {'answers':[]}
    for df in top_1_retrieve_df_list:
        ans_df = df[['qid','retrieve']]
        answer_dict = ans_df.to_dict(orient='records')
        answers_dict['answers'].extend(answer_dict)
    with open(Config.my_prediction_path, 'w', encoding='utf8') as f:
        json.dump(answers_dict, f, ensure_ascii=False, indent=4)
    return answers_dict


def apply_reranking(reranker, hybrid_retrieve: pd.DataFrame):
    query_passage_pair = hybrid_retrieve[['query','sentence_vec']].values.tolist()
    scores = reranker.compute_score(query_passage_pair)
    hybrid_retrieve['reranking_score'] = scores
    hybrid_retrieve = hybrid_retrieve.drop(columns=['ranking_bm25','ranking_vec','score_vec','weighted_sum','rrf_score'])
    return hybrid_retrieve

def get_top_1_retrieve_pd(hybrid_retrieve: pd.DataFrame, score_field: str):
    top_1_idx = hybrid_retrieve.groupby('qid')[score_field].idxmax()
    top_1_retrieve = hybrid_retrieve.loc[top_1_idx].reset_index(drop = True)
    top_1_retrieve['retrieve'] = top_1_retrieve['file']
    top_1_retrieve = top_1_retrieve.drop(columns=[score_field])
    return top_1_retrieve

def get_hybrid_retrieve_pd(vector_retrieve_pd: pd.DataFrame, bm25_retrieve_pd: pd.DataFrame, bm25_weight = 0.5, vector_weight = 0.5) -> pd.DataFrame:
    vector_retrieve_pd = vector_retrieve_pd
    bm25_retrieve_pd = bm25_retrieve_pd
    merged_retrieve = pd.merge(bm25_retrieve_pd, vector_retrieve_pd, on=['qid', 'file', 'category', 'query'],suffixes=('_bm25', '_vec'))
    merged_retrieve['weighted_sum'] = merged_retrieve['score_bm25'] + merged_retrieve['score_vec']
    return merged_retrieve

def get_RRF_score(hybrid_retrieve: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    hybrid_retrieve['rrf_score'] = (1 / (k + hybrid_retrieve['ranking_vec'])) + (1 / (k + hybrid_retrieve['ranking_bm25']))
    return hybrid_retrieve

class BM25Retrieve:
    @staticmethod
    def retrieve_aux(qs: str, source: list[int], corpus_df: pd.DataFrame) \
            -> tuple[list[float], dict[int, str]]:
        filtered_chunks = list(corpus_df[corpus_df['file'].isin(source)]['chunk'])
        #filtered_chunk_to_id: dict[int,str] = corpus_df[corpus_df['file'].isin(source)].set_index('chunk')['chunk_id'].to_dict()
        #filtered_chunks = list(filtered_chunk_to_id.keys())
        if len(filtered_chunks) == 0:
            print(f'BM25')
            print(f'source: {source}, filtered_chunks: {filtered_chunks}')
        tokenized_corpus = [list(jieba.cut_for_search(chunk)) for chunk in filtered_chunks]
        tokenized_query = list(jieba.cut_for_search(qs))
        bm25 = BM25Okapi(tokenized_corpus)
        score_list = sorted(bm25.get_scores(tokenized_query), reverse=True)
        # max_val, min_val = np.min(score_list), np.max(score_list)
        # score_list = (score_list - min_val) / (max_val - min_val)  if (max_val - min_val) != 0 else [0]*len(score_list)
        # score_list = (score_list - np.mean(score_list))/np.std(score_list) if np.std(score_list) != 0 else [0]*len(score_list)
        top_k = len(filtered_chunks)
        top_k_chunks = bm25.get_top_n(tokenized_query, filtered_chunks, top_k)
        #top_k_chunks_id = [filtered_chunk_to_id[chunk] for chunk in top_k_chunks]
        #top_k_id_to_chunk = dict(zip(top_k_chunks, top_k_chunks_id))
        # print(f'score: {score_list}'
        #       f'top_n: {top_k_chunks}')

        return score_list, top_k_chunks

    @staticmethod
    # retrieve top n candidates for all queries
    def retrieve(queries_info: pd.DataFrame , corpus_df: pd.DataFrame) \
            -> pd.DataFrame:
        chunk_to_file_dict = corpus_df.set_index('chunk')['file'].to_dict()
        # retrieved_data_for_all_query: list[list[dict[str, Union[int, float]]]] = []
        row = []
        for qid, query, source, category in tqdm(zip(queries_info['qid'], queries_info['query'], queries_info['source'], queries_info['category']), total=len(queries_info['qid'])):
            #print(f'query: {query}, source: {source}')
            score_list, top_k_chunk = BM25Retrieve.retrieve_aux(query, source, corpus_df)
            for score, chunk in zip(score_list, top_k_chunk):
                row.append(
                    {
                        'qid': qid,
                        'file': chunk_to_file_dict[chunk],
                        'score': score,
                        'query': query,
                        'sentence': chunk,
                        'category': category
                    }
                )

        retrieved_info_df = pd.DataFrame(row)
        retrieved_info_df['ranking'] = retrieved_info_df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
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
        filtered_corpus = corpus_df[corpus_df['file'].isin(source)]
        filtered_chunks = list(filtered_corpus['chunk'])
        filtered_files = list(filtered_corpus['file'])
        


        corpus_emb = embedder.encode(filtered_chunks, convert_to_tensor=True)
        query_emb = embedder.encode(query, convert_to_tensor=True)
        similarity_scores = embedder.similarity(query_emb, corpus_emb)[0]
        top_k = len(filtered_chunks)
        scores, indices = torch.topk(similarity_scores, k=top_k)
        
        score_list = list(scores.cpu())
        #score_list = (score_list - np.mean(score_list))/np.std(score_list) if np.std(score_list) != 0 else [0]*len(score_list)
        #max_val, min_val = np.min(score_list), np.max(score_list)
        #score_list = (score_list - min_val) / (max_val - min_val)  if (max_val - min_val) != 0 else [0]*len(score_list)
        top_k_chunks = [filtered_chunks[index] for index in indices]
        top_k_files = [filtered_files[index] for index in indices]
        #top_k_chunks_id = [filtered_chunk_to_id[chunk] for chunk in top_k_chunks]
        #top_k_id_to_chunk = dict(zip(top_k_chunks, top_k_chunks_id))
        
        return score_list, top_k_chunks, top_k_files

    
    @staticmethod
    # retrieve top n candidates for all queries
    def retrieve(embedder, queries_info: pd.DataFrame, corpus_df: pd.DataFrame) \
            -> pd.DataFrame:
        chunk_to_file_dict = corpus_df.set_index('chunk')['file'].to_dict()
        row = []
        for qid, query, source, category in tqdm(zip(queries_info['qid'], queries_info['query'], queries_info['source'], queries_info['category']), total=len(queries_info['qid'])):
            # filtered_corpus: list = corpus_df[corpus_df['file'].isin(source)]['chunk'].to_list()
            # if len(filtered_corpus) == 0:
            #     print(f'vec')
            #     print(f'source: {source}, filtered_corpus: {filtered_corpus}')
            
            score_list, top_k_chunks, top_k_files = VectorRetriever.retrieve_aux(embedder, query=query, source=source, corpus_df=corpus_df)
            for score, chunk, file in zip(score_list, top_k_chunks, top_k_files):
                row.append(
                    {
                        'qid': qid,
                        'file': file,
                        'score': score,
                        'query': query,
                        'sentence': chunk,
                        'category': category
                    }
                )
        retrieved_info_df = pd.DataFrame(row)
        highest_file_score_idx = retrieved_info_df.groupby(['qid','file'])['score'].idxmax()
        unique_file_vector_retrieve = retrieved_info_df.loc[highest_file_score_idx]
        unique_file_vector_retrieve['ranking'] = unique_file_vector_retrieve.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
        unique_file_vector_retrieve = unique_file_vector_retrieve.sort_values(by=['qid', 'ranking'], ascending=[True, True]).reset_index(drop=True)

        return unique_file_vector_retrieve
