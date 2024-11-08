import os
import json
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
import pandas as pd
import re
from pathlib import Path
class Config:
    faq_path = Path('../data/reference/faq')
    finance_path = Path('../data/reference/finance')
    insurance_path = Path('../data/reference/insurance')
    test_faq_path = Path('../data/reference/test_faq')
    test_finance_path = Path('../data/reference/test_finance')
    test_insurance = Path('../data/reference/test_insurance')
    truth_path = Path('../data/dataset/preliminary/ground_truths_example.json')
    prediction_path = Path('../data/dataset/preliminary/pred_retrieve.json')
    queries_info_path = Path('../data/dataset/preliminary/questions_example.json')

def get_queried_info(queries_info_path: str) -> pd.DataFrame:
    with open(queries_info_path, 'rb') as f:
        queries_info = json.load(f)['questions']
        row = []
        for query_info in queries_info:
            row.append(
                {
                    'qid': query_info['qid'],
                    'source': query_info['source'],
                    'query': query_info['query'],
                    'category': query_info['category']
                }
            )
    queries_info_df = pd.DataFrame(row)
    return queries_info_df

def preprocess(text):
    # 去除標點符號
    text = re.sub(r'[^\w\s]', '', text)
    # 可以添加更多預處理步驟，如去除數字、轉換大小寫等
    return text


def load_data(source_path) -> pd.DataFrame:
    row = []
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    chunk_id = 0
    for file in tqdm(masked_file_ls):
        chunks = read_pdf(os.path.join(source_path, file))
        for chunk in chunks:
            row.append({
                'chunk_id': chunk_id,
                'file': int(file.replace('.pdf', '')),
                'chunk': chunk if chunk != '' else '0'
            })
            chunk_id = chunk_id + 1
    return pd.DataFrame(row)


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None) -> list[str]:
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    chunks = chunking(pdf_text, chunk_size=500, overlap=200)

    return chunks  # 返回萃取出的文本


def chunking(text: str, chunk_size: int = 500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < chunk_size:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks


def evaluation(predict_path, truth_path) -> None:
    faq: list = []
    finance: list = []
    insurance: list = []

    with open(predict_path, 'rb') as f:
        predict_list = json.load(f)['answers']
    with open(truth_path, 'rb') as f:
        truth_list = json.load(f)['ground_truths']

    for predict_dict, truth_dict in zip(predict_list, truth_list):
        if truth_dict['category'] == 'insurance':
            insurance.append(predict_dict['retrieve'] == truth_dict['retrieve'])
        elif truth_dict['category'] == 'finance':
            finance.append(predict_dict['retrieve'] == truth_dict['retrieve'])
        elif truth_dict['category'] == 'faq':
            faq.append(predict_dict['retrieve'] == truth_dict['retrieve'])

    total = insurance + finance + faq

    print(f'insurance: {sum(insurance) / len(insurance):.4f}\n'
          f'finance: {sum(finance) / len(finance):.4f}\n'
          f'faq: {sum(faq) / len(faq):.4f}\n'
          f'total: {sum(total) / len(total)}')
