import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import  AutoTokenizer
from src.contriever import Contriever
from utils import top_k_top_p_filtering, set_logger
from os.path import join
from flask import Flask, redirect, url_for, request
from knn import get_batch_embeddings
import time
import json
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码

def retrieve_texts(nn_dict):
    line_numbers = [nn[0] for nn in sorted(nn_dict.items(), key=lambda x:x[1])]
    results = []
    for line in line_numbers:
        results.append(json.loads(text_lines[line + 1]))
    return results



@app.route('/retrieve', methods=['GET'])
def retrieve():
    start_time = time.time()
    nn_dict = {}
    query = request.args.get('query', type=str)
    k = request.args.get('k', type=int)
    input = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
    query = model(**input).to(device)
    distances = torch.linalg.vector_norm(embeddings-query,ord=2,dim=1) # 2-norm, i.e. L2 distance of query to each embedding, return in shape [n]
    sorted_dist, indices = torch.sort(distances)
    nn_dict.update(dict(zip(indices[:k].tolist(),sorted_dist[:k].tolist())))
    print(nn_dict)
    neighbors = retrieve_texts(nn_dict)
    use_time = time.time()-start_time
    return {'neighbors':neighbors,'use_time':use_time}


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str,
                        required=False, help='生成设备')
    parser.add_argument('--port', type=int, default=8085, help='服务绑定的端口号')
    parser.add_argument('--log_path', default='log/http_service.log',
                        type=str, required=False, help='日志存放位置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    device = 'cuda:0' if args.cuda else 'cpu'
    # device = 'cpu'

    # 创建日志对象
    logger = set_logger(args.log_path)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('mcontriever')
    print("tokenizer loaded")

    # 加载模型
    model = Contriever.from_pretrained('mcontriever').to(device)
    model.eval()
    print("model loaded")
    
    # load files
    embeddings = []
    for i in range(1):
        embeddings += get_batch_embeddings(i)
    embeddings = torch.row_stack(embeddings).detach().to(device)
    print('embeddings loaded')
    with open('database/text_database.json')as f:
        text_lines = f.readlines()
    print('text_lines loaded')
    
    app.run(debug=False, host="0.0.0.0", port=args.port)
