import json


def evaluation(predict_path, truth_path) -> None:
    faq: list = []
    finance: list = []
    insurance: list = []

    with open(predict_path, 'rb') as f:
        predict_list = json.load(f)['answers']
    with open(truth_path, 'rb') as f:
        truth_list = json.load(f)['ground_truths']

    for predict_dict, truth_dict in zip(predict_list, truth_list):
        if truth_dict['category'] == 'insurance ':
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
