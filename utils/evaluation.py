from torch.autograd import Variable
import numpy as np
import torch
import logging
from tqdm import tqdm
import json  # <-- [新增] 导入
import os    # <-- [新增] 导入


def ours_compress(train, test, encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def ours_distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = t_encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = s_encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def compress(train, test, encode_discrete, device):
    print("Starting compress function")
    logging.info("Starting compress function")
    retrievalB = list([])
    retrievalL = list([])
    
    # --- [修改] ---
    retrieval_fnames = list([]) 
    if hasattr(train.dataset, 'data'):
        retrieval_fnames = [os.path.basename(f) for f in train.dataset.data] 
    # --- [修改结束] ---
    
    print("Processing training data (Database)...")
    logging.info("Processing training data (Database)...")
    for batch_step, (data, _, target) in enumerate(tqdm(train)):
        var_data = Variable(data.to(device))
        try:
            code = encode_discrete(var_data)
            retrievalB.extend(code.cpu().data.numpy())
            retrievalL.extend(target)
        except Exception as e:
            logging.error(f"Error processing training batch {batch_step}: {str(e)}")
            raise e

    queryB = list([])
    queryL = list([])
    
    # --- [修改] ---
    query_fnames = list([])
    if hasattr(test.dataset, 'data'):
        query_fnames = [os.path.basename(f) for f in test.dataset.data]
    # --- [修改结束] ---
    
    print("Processing test data (Query)...")
    logging.info("Processing test data (Query)...")
    for batch_step, (data, _, target) in enumerate(tqdm(test)):
        var_data = Variable(data.to(device))
        try:
            code = encode_discrete(var_data)
            queryB.extend(code.cpu().data.numpy())
            queryL.extend(target)
        except Exception as e:
            logging.error(f"Error processing test batch {batch_step}: {str(e)}")
            raise e

    print("Converting to arrays...")
    logging.info("Converting to arrays...")
    try:
        retrievalB = np.array(retrievalB)
        retrievalL = np.stack(retrievalL)

        queryB = np.array(queryB)
        queryL = np.stack(queryL)
    except Exception as e:
        logging.error(f"Error converting to arrays: {str(e)}")
        raise e

    print("compress function completed successfully")
    logging.info("compress function completed successfully")
    
    # --- [修改] 返回文件名列表 ---
    return retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames


def distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    
    # --- [修改] ---
    retrieval_fnames = list([]) 
    if hasattr(train.dataset, 'data'):
        retrieval_fnames = [os.path.basename(f) for f in train.dataset.data]
    # --- [修改结束] ---

    for batch_step, (data, _, target) in enumerate(tqdm(train)):
        var_data = Variable(data.to(device))
        code = t_encode_discrete(var_data) # 数据库使用教师模型
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    
    # --- [修改] ---
    query_fnames = list([])
    if hasattr(test.dataset, 'data'):
        query_fnames = [os.path.basename(f) for f in test.dataset.data]
    # --- [修改结束] ---

    for batch_step, (data, _, target) in enumerate(tqdm(test)):
        var_data = Variable(data.to(device))
        code = s_encode_discrete(var_data) # 查询使用学生模型
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    
    # --- [修改] 返回文件名列表 ---
    return retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames


def calculate_hamming(B1, B2):
    q = B2.shape[1] 
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_euclidean(B1, B2):
    return np.sum((B1 - B2) ** 2, axis=1)


# --- [保留] 步骤五: 保留 mAP (用于 层面一: 语义评估) ---
def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    [保留] 
    计算基于 *语义标签* 的 mAP。
    """
    print("Starting calculate_top_map function (Semantic mAP)")
    logging.info("Starting calculate_top_map function (Semantic mAP)")
    num_query = queryL.shape[0]
    topkmap = 0
    print(f"Processing {num_query} queries...")
    logging.info(f"Processing {num_query} queries...")

    for iter in tqdm(range(num_query)):
        try:
            # 核心: gnd (ground-truth) 是通过 *标签* 点积定义的
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = calculate_hamming(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]  # reorder gnd

            tgnd = gnd[0:topk]
            tsum = int(np.sum(tgnd))
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
        except Exception as e:
            logging.error(f"Error processing query {iter}: {str(e)}")
            raise e

    topkmap = topkmap / num_query
    print("calculate_top_map function completed successfully")
    logging.info("calculate_top_map function completed successfully")
    return topkmap


def calculate_top_map_in_euclidean_space(qB, rB, queryL, retrievalL, topk):
    """
    [保留]
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        euc = calculate_euclidean(qB[iter, :], rB)
        ind = np.argsort(euc)
        gnd = gnd[ind]  # reorder gnd

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# --- [新增] 步骤五: 添加 Perceptual Metrics (用于 层面二: 决策评估) ---
def calculate_perceptual_metrics(qB, rB, q_fnames, r_fnames, gnd_json_path, num_bits):
    """
    [新增]
    计算感知哈希指标 (AHD, F1-Score)，基于 'gnd.json' 文件。
    """
    
    print("Starting calculate_perceptual_metrics function (F1, AHD)...")
    logging.info("Starting calculate_perceptual_metrics function (F1, AHD)...")

    # 1. 加载 gnd.json
    if not os.path.exists(gnd_json_path):
        logging.error(f"GND file not found at: {gnd_json_path}. Skipping perceptual metrics.")
        print(f"Error: GND file not found at: {gnd_json_path}")
        return {}
        
    with open(gnd_json_path, 'r') as f:
        gnd_data = json.load(f) #
    
    # 2. [关键更新] 定义所有在 image_attack.py 中生成的攻击键
    ATTACK_KEYS = [
        "jpegqual", "crops", "blur", "noise", 
        "brightness", "contrast", "paint", "hybrid"
    ]

    # 3. 创建从文件名到哈希索引的快速查找
    q_base_map = {os.path.splitext(f)[0]: i for i, f in enumerate(q_fnames)} 
    r_base_map = {os.path.splitext(f)[0]: i for i, f in enumerate(r_fnames)} 

    gnd_qimlist = gnd_data['qimlist'] #
    gnd_imlist = gnd_data['imlist']   #
    gnd = gnd_data['gnd']             #

    if len(q_base_map) == 0 or len(r_base_map) == 0:
         logging.error("Could not get filenames from data loaders. Skipping perceptual metrics.")
         print("Error: Could not get filenames from data loaders. Make sure dataset.data contains file paths.")
         return {}

    # 4. 构建 (距离, 真实标签) 对
    true_labels = []
    distances = []
    
    positive_distances = []
    negative_distances = []
    
    print(f"Calculating all-pairs Hamming distances for {len(gnd_qimlist)} queries...")
    
    all_pair_distances = 0.5 * (num_bits - np.dot(qB, rB.transpose()))
    
    print("Matching queries to database using gnd.json...")
    for q_gnd_idx, q_base_name in enumerate(tqdm(gnd_qimlist)):
        if q_base_name not in q_base_map:
            continue 
            
        q_hash_idx = q_base_map[q_base_name] 
        
        positive_r_gnd_indices = set()
        query_gnd_attacks = gnd[q_gnd_idx] #
        
        for key in ATTACK_KEYS: #
            if key in query_gnd_attacks:
                positive_r_gnd_indices.update(query_gnd_attacks[key])
        
        positive_r_base_names = {gnd_imlist[idx] for idx in positive_r_gnd_indices} #
        
        for r_base_name, r_hash_idx in r_base_map.items():
            
            dist = all_pair_distances[q_hash_idx, r_hash_idx]
            distances.append(dist)
            
            if r_base_name in positive_r_base_names:
                true_labels.append(1) 
                positive_distances.append(dist)
            else:
                true_labels.append(0) 
                negative_distances.append(dist)

    y_true = np.array(true_labels)
    y_dist = np.array(distances)
    
    if len(positive_distances) == 0:
        logging.error("No positive pairs found based on gnd.json! Check gnd file and data loaders.")
        print("Error: No positive pairs found based on gnd.json!")
        return {}

    # 5. 计算 AHD 指标
    ahd_positive = np.mean(positive_distances)
    ahd_negative = np.mean(negative_distances)
    
    # 6. 通过扫描阈值找到最佳 F1-Score
    best_f1 = 0
    best_threshold = 0
    
    print("Sweeping thresholds to find best F1-Score...")
    for t in tqdm(range(0, num_bits + 1)):
        y_pred = (y_dist <= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            

    metrics = {
        'Best F1-Score': float(best_f1),
        'Best Threshold (Hamming)': int(best_threshold),
        'AHD-Positive (Robustness)': float(ahd_positive),
        'AHD-Negative (Discriminability)': float(ahd_negative)
    }
    print("--- Perceptual Metrics (F1-Score / AHD) ---")
    print(json.dumps(metrics, indent=4))
    print("-------------------------------------------------")
    
    return metrics



# 7.3 为了回复review，增加了topk的计算
def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test,
                                      topK + batch_size).fill_(n_bits + 1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits,
                                         trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(
            torch.cuda.LongTensor).unsqueeze(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices


def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK,
                           is_single_label=True):
    n_test = query_labels.size(0)

    Indices = retrieved_indices[:, :topK]
    if is_single_label:
        print(query_labels)
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(
            torch.cuda.ShortTensor)
    else:
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels,
                                   dim=0).type(torch.cuda.ShortTensor)
        # --- [修复] ---
        # 修复了 -int_ 的拼写错误
        test_labels = query_labels.unsqueeze(1).expand(
            n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor) # <-- [修复]
        # --- [修复结束] ---
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)

    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(topK)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k