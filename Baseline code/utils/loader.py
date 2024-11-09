import os
import json
from tqdm import tqdm
import pdfplumber 
import pandas as pd
import re
from pathlib import Path
from concurrent.futures import as_completed, ProcessPoolExecutor
from PIL import Image
import torch
import shutil
from pathlib import Path

class Config:
    f = Path('../data/reference/faq/pid_map_content.json')
    faq_path = Path('../data/reference/faq/pid_map_content.json')
    finance_path = Path('../data/reference/finance')
    insurance_path = Path('../data/reference/insurance')
    test_faq_path = Path('../data/reference/test_faq')
    test_finance_path = Path('../data/reference/test_finance')
    test_insurance = Path('../data/reference/test_insurance')
    truth_path = Path('../data/dataset/preliminary/ground_truths_example.json')
    prediction_path = Path('../data/dataset/preliminary/pred_retrieve.json')
    my_prediction_path = Path('../data/dataset/my_ans/pred_retrieve.json')
    queries_info_path = Path('../data/dataset/preliminary/questions_example.json')
    formal_queries_info_path = Path('../data/dataset/preliminary/questions_preliminary.json')
    img_path = Path('../data/img')            
        
def remove_punctuation(text):
    # 定義中文標點符號的範圍
    punctuation = r'[^\w\s]'
    # 使用正則表達式替換
    return re.sub(punctuation, '', text)
    
class Reader:
    def __init__(self, vision_model, processor):
        self.vision_model = vision_model
        self.processor = processor
    
    def convert_jpg_to_df(self) -> pd.DataFrame:
        processor, model = self.processor, self.model
        self.sort_img_dir()
        def sort_key(filename):
            page_number = int(filename.split('_page_')[-1].split('.')[0])
            main_number = int(filename.split('_page_')[0])
            return main_number, page_number
        with torch.no_grad():
            img_dir = os.listdir(Config.img_path/'finance')
            row = []
            for file in tqdm(img_dir):
                all_txt = ''
                for jpg in sorted(os.listdir(Config.img_path/'finance'/file), key = sort_key):
                    img = Image.open(Config.img_path/'finance'/file/jpg)
                    prompt = '''
                    把這張圖片轉換為文字(繁體中文)，你只能說繁體中文
                    '''
                    messages = [
                        {
                            'role': 'user','content':[
                                {'type': 'image'},
                                {'type': 'text', 'text': prompt}
                            ]
                        }
                    ]

                    input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
                    inputs = processor(img, input_text, return_tensors='pt').to(model.device)
                    output = model.generate(**inputs, max_new_tokens=530)
                    decoded_output = processor.decode(output[0], skip_special_tokens=True)
                    cleaned_output = re.sub(f'(user|assistant|把這張圖片轉換為文字(繁體中文)，你只能說繁體中文)', '', decoded_output).strip()
                    all_txt = all_txt + cleaned_output
                row.append(
                    {
                        'content': all_txt,
                        'file': file
                    }
                )
        return pd.DataFrame(row)

    def sort_img_dir(self):

        image_folder = Config.img_path/'finance'

        if not image_folder.exists():
            print("指定的資料夾不存在")
        else:
            for file_path in image_folder.glob("*.jpg"):
                prefix = file_path.stem.split('_')[0]
                
                prefix_folder = image_folder / prefix
                prefix_folder.mkdir(exist_ok=True)
                
                shutil.move(str(file_path), prefix_folder / file_path.name)


    def convert_pdf_to_jpg(self, category, pdf_path, resolution=600):

        os.makedirs(Config.img_path, exist_ok=True)
        with pdfplumber.open(pdf_path) as pdf:
            file_name=os.path.splitext(os.path.basename(pdf_path))[0]
            for i, page in enumerate(tqdm(pdf.pages,desc='convert_pdf_to_jpg'), start=1):
                image = page.to_image(resolution=resolution)
                pil_image = image.original
                output_path =  Config.img_path/f'{category}'/f'{file_name}_page_{i}.jpg'
                pil_image = pil_image.convert("RGB") 
                pil_image.save(output_path, format="JPEG")
                print(f"已儲存: {output_path}")
            
    def load_faq(source_path: str) -> pd.DataFrame:
        with open(source_path, 'rb') as f:
            key_to_source_dict = json.load(f)  
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        row = []
        for pid, content in tqdm(key_to_source_dict.items(), desc="Loading data"):
            row.append(
                {
                    'file': pid,
                    'chunk': str(content[0])
                }
            )
        return pd.DataFrame(row)
        
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


    def process_file(model = None, processor = None, category = None, file: str = None, source_path: str = None, is_chunking: bool = None, sheet = None) -> list[dict]:
        file_path = os.path.join(source_path, file)
        rows = []
        chunks = read_pdf(model=model, processor=processor, category=category, pdf_loc=file_path, is_chunking=is_chunking, sheet=sheet)
        file_num = int(os.path.splitext(file)[0])  # 去除文件擴展名並轉換為整數
        for chunk in chunks:
            rows.append({
                'file': file_num,
                'chunk': chunk.strip() if chunk.strip() else '0'
            })
        return rows

def load_data(model, processor, source_path: str, is_chunking: bool = False, sheet = None) -> pd.DataFrame:
    masked_file_ls = os.listdir(source_path)
    category = os.path.basename(source_path)
    all_rows = []
    
    max_workers = 64

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, model, processor, category, file, source_path, is_chunking, sheet): file
            for file in masked_file_ls
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            file = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return pd.DataFrame(all_rows)



# 讀取單個PDF文件並返回其文本內容
def read_pdf(model = None, processor =None, category = None, sheet = None, pdf_loc = None, page_infos: list = None, is_chunking = False) -> list[str]:

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    with pdfplumber.open(pdf_loc) as pdf:  # 打開指定的PDF文件
        file_name=os.path.splitext(os.path.basename(pdf_loc))[0]
        #print(f'pdf_loc: {file_name}')
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''.join([page.extract_text() for page in pages])
    if pdf_text == '':
        if sheet[sheet['file'] == file_name]['content'][0]:
            pdf_text = sheet[sheet['file'] == file_name]['content'][0]
        else:
            convert_pdf_to_jpg(category=category, pdf_path=pdf_loc)
    if is_chunking:
        return chunking(pdf_text, chunk_size=500, overlap=200)
    else:
        return [pdf_text]

def chunking(text: str, chunk_size: int = 500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
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
