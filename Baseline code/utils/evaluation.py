import json
import pandas as pd
from utils.Loader import *

def get_ground_truth_df(truth_path: str) -> pd.DataFrame:
    with open(truth_path, 'rb') as f:
        ground_truth_df = pd.DataFrame(json.load(f)['ground_truths'])
        
    return ground_truth_df
        
        
def evaluation_df(finance_top_1_retrieve_df = None, 
                  insurance_top_1_retrieve_df = None, 
                  faq_top_1_retrieve_df = None,
                  category = ['insurance', 'finance', 'faq']):
    ground_truth_df = get_ground_truth_df(Config.truth_path)
    if('insurance' in category):
        insurance_ground_truth = list(ground_truth_df[ground_truth_df['category'] == 'insurance']['retrieve'])
        insurance_predictions: list = insurance_top_1_retrieve_df['retrieve']
        insurance_correctness_list = [prediction == truth for prediction, truth in zip(insurance_predictions, insurance_ground_truth)]
        insurance_accuracy = sum(insurance_correctness_list)/len(insurance_correctness_list)
        print(f'insurance: {insurance_accuracy:.4f}')
    if('finance' in category):
        finance_ground_truth = list(ground_truth_df[ground_truth_df['category'] == 'finance']['retrieve'])
        finance_predictions: list = finance_top_1_retrieve_df['retrieve']
        finance_correctness_list = [prediction == truth for prediction, truth in zip(finance_predictions, finance_ground_truth)]
        finance_accuracy = sum(finance_correctness_list)/len(finance_correctness_list)
        print(f'finance: {finance_accuracy:.4f}')
    if('faq' in category):
        faq_ground_truth = list(ground_truth_df[ground_truth_df['category'] == 'faq']['retrieve'])
        faq_predictions: list = faq_top_1_retrieve_df['retrieve']
        faq_correctness_list = [prediction == truth for prediction, truth in zip(faq_predictions, faq_ground_truth)]
        faq_accuracy = sum(faq_correctness_list)/len(faq_correctness_list)
        print(f'faq: {faq_accuracy:.4f}')
    if('insurance' in category and 'finance' in category and 'faq' in category):
        total = insurance_correctness_list + finance_correctness_list + faq_correctness_list
        print(f'total: {sum(total) / len(total)}')
        

def evaluation_json(predict_path, truth_path) -> None:
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
