# ============ utils/evaluation.py - PaddlePaddle 版本 ============
"""
评估指标模块 - PaddlePaddle 版本
包含哈希压缩、mAP 计算、感知指标等
"""

import numpy as np
import logging
from tqdm import tqdm
import json
import os
import paddle


def ours_compress(train, test, encode_discrete, device):
    """压缩函数 (旧版本，用于 Ours 训练)"""
    retrievalB = []
    retrievalL = []
    for batch_step, (data, _, target) in enumerate(train):
        var_data = data
        code = encode_discrete(var_data)
        retrievalB.extend(code.numpy())
        retrievalL.extend(target)

    queryB = []
    queryL = []
    for batch_step, (data, _, target) in enumerate(test):
        var_data = data
        code = encode_discrete(var_data)
        queryB.extend(code.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)
    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def ours_distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    """蒸馏压缩函数 (旧版本)"""
    retrievalB = []
    retrievalL = []
    for batch_step, (data, _, target) in enumerate(train):
        var_data = data
        code = t_encode_discrete(var_data)
        retrievalB.extend(code.numpy())
        retrievalL.extend(target)

    queryB = []
    queryL = []
    for batch_step, (data, _, target) in enumerate(test):
        var_data = data
        code = s_encode_discrete(var_data)
        queryB.extend(code.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)
    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def compress(train, test, encode_discrete, device):
    """压缩函数 - 返回哈希码、标签和文件名"""
    print("Starting compress function")
    logging.info("Starting compress function")

    retrievalB = []
    retrievalL = []
    retrieval_fnames = []
    if hasattr(train.dataset, 'data'):
        retrieval_fnames = [os.path.basename(f) for f in train.dataset.data]

    print("Processing training data (Database)...")
    logging.info("Processing training data (Database)...")
    for batch_step, (data, _, target) in enumerate(tqdm(train)):
        try:
            code = encode_discrete(data)
            retrievalB.extend(code.numpy())
            retrievalL.extend(target.numpy() if hasattr(target, 'numpy') else target)
        except Exception as e:
            logging.error(f"Error processing training batch {batch_step}: {str(e)}")
            raise e

    queryB = []
    queryL = []
    query_fnames = []
    if hasattr(test.dataset, 'data'):
        query_fnames = [os.path.basename(f) for f in test.dataset.data]

    print("Processing test data (Query)...")
    logging.info("Processing test data (Query)...")
    for batch_step, (data, _, target) in enumerate(tqdm(test)):
        try:
            code = encode_discrete(data)
            queryB.extend(code.numpy())
            queryL.extend(target.numpy() if hasattr(target, 'numpy') else target)
        except Exception as e:
            logging.error(f"Error processing test batch {batch_step}: {str(e)}")
            raise e

    print("Converting to arrays...")
    logging.info("Converting to arrays...")
    retrievalB = np.array(retrievalB)
    retrievalL = np.array(retrievalL) if retrievalL and hasattr(retrievalL[0], '__len__') else np.stack(retrievalL) if retrievalL else np.array([])
    queryB = np.array(queryB)
    queryL = np.array(queryL) if queryL and hasattr(queryL[0], '__len__') else np.stack(queryL) if queryL else np.array([])

    print("compress function completed successfully")
    logging.info("compress function completed successfully")

    return retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames


def distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    """蒸馏压缩函数"""
    retrievalB = []
    retrievalL = []
    retrieval_fnames = []
    if hasattr(train.dataset, 'data'):
        retrieval_fnames = [os.path.basename(f) for f in train.dataset.data]

    for batch_step, (data, _, target) in enumerate(tqdm(train)):
        code = t_encode_discrete(data)
        retrievalB.extend(code.numpy())
        retrievalL.extend(target.numpy() if hasattr(target, 'numpy') else target)

    queryB = []
    queryL = []
    query_fnames = []
    if hasattr(test.dataset, 'data'):
        query_fnames = [os.path.basename(f) for f in test.dataset.data]

    for batch_step, (data, _, target) in enumerate(tqdm(test)):
        code = s_encode_discrete(data)
        queryB.extend(code.numpy())
        queryL.extend(target.numpy() if hasattr(target, 'numpy') else target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.array(retrievalL) if retrievalL and hasattr(retrievalL[0], '__len__') else np.stack(retrievalL) if retrievalL else np.array([])
    queryB = np.array(queryB)
    queryL = np.array(queryL) if queryL and hasattr(queryL[0], '__len__') else np.stack(queryL) if queryL else np.array([])

    return retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames


def calculate_hamming(B1, B2):
    """计算汉明距离"""
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_euclidean(B1, B2):
    """计算欧氏距离"""
    return np.sum((B1 - B2) ** 2, axis=1)


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """计算基于语义标签的 mAP"""
    print("Starting calculate_top_map function (Semantic mAP)")
    logging.info("Starting calculate_top_map function (Semantic mAP)")

    # 确保 queryL 和 retrievalL 是 numpy 数组
    if hasattr(queryL, 'numpy'):
        queryL = queryL.numpy()
    if hasattr(retrievalL, 'numpy'):
        retrievalL = retrievalL.numpy()

    num_query = queryL.shape[0]
    topkmap = 0
    print(f"Processing {num_query} queries...")
    logging.info(f"Processing {num_query} queries...")

    for i in tqdm(range(num_query)):
        try:
            gnd = (np.dot(queryL[i, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = calculate_hamming(qB[i, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]
            tsum = int(np.sum(tgnd))
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
        except Exception as e:
            logging.error(f"Error processing query {i}: {str(e)}")
            raise e

    topkmap = topkmap / num_query
    print("calculate_top_map function completed successfully")
    logging.info("calculate_top_map function completed successfully")
    return topkmap


def calculate_top_map_in_euclidean_space(qB, rB, queryL, retrievalL, topk):
    """计算欧氏空间中的 mAP"""
    if hasattr(queryL, 'numpy'):
        queryL = queryL.numpy()
    if hasattr(retrievalL, 'numpy'):
        retrievalL = retrievalL.numpy()

    num_query = queryL.shape[0]
    topkmap = 0
    for i in range(num_query):
        gnd = (np.dot(queryL[i, :], retrievalL.transpose()) > 0).astype(np.float32)
        euc = calculate_euclidean(qB[i, :], rB)
        ind = np.argsort(euc)
        gnd = gnd[ind]

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


def calculate_perceptual_metrics(qB, rB, q_fnames, r_fnames, gnd_json_path, num_bits):
    """计算感知哈希指标 (AHD, F1-Score)"""
    print("Starting calculate_perceptual_metrics function (F1, AHD)...")
    logging.info("Starting calculate_perceptual_metrics function (F1, AHD)...")

    if not os.path.exists(gnd_json_path):
        logging.error(f"GND file not found at: {gnd_json_path}. Skipping perceptual metrics.")
        print(f"Error: GND file not found at: {gnd_json_path}")
        return {}

    with open(gnd_json_path, 'r') as f:
        gnd_data = json.load(f)

    ATTACK_KEYS = [
        "jpegqual", "crops", "blur", "noise",
        "brightness", "contrast", "paint", "hybrid"
    ]

    q_base_map = {os.path.splitext(f)[0]: i for i, f in enumerate(q_fnames)}
    r_base_map = {os.path.splitext(f)[0]: i for i, f in enumerate(r_fnames)}

    gnd_qimlist = gnd_data['qimlist']
    gnd_imlist = gnd_data['imlist']
    gnd = gnd_data['gnd']

    if len(q_base_map) == 0 or len(r_base_map) == 0:
        logging.error("Could not get filenames from data loaders. Skipping perceptual metrics.")
        print("Error: Could not get filenames from data loaders.")
        return {}

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
        query_gnd_attacks = gnd[q_gnd_idx]

        for key in ATTACK_KEYS:
            if key in query_gnd_attacks:
                positive_r_gnd_indices.update(query_gnd_attacks[key])

        positive_r_base_names = {gnd_imlist[idx] for idx in positive_r_gnd_indices}

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
        logging.error("No positive pairs found based on gnd.json!")
        print("Error: No positive pairs found!")
        return {}

    ahd_positive = np.mean(positive_distances)
    ahd_negative = np.mean(negative_distances)

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


def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    """Top-K 检索 (使用 Paddle 版本)"""
    n_bits = doc_b.shape[1]
    n_train = doc_b.shape[0]
    n_test = query_b.shape[0]

    topScores_list = []
    topIndices_list = []

    testBinmat = paddle.unsqueeze(query_b, axis=2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat = paddle.unsqueeze(trainBinmat, axis=0)
        trainBinmat = paddle.transpose(trainBinmat, [0, 2, 1])
        trainBinmat = paddle.expand(trainBinmat, [testBinmat.shape[0], n_bits, trainBinmat.shape[2]])

        testBinmatExpand = paddle.expand(paddle.unsqueeze(query_b, 1), [e_idx - s_idx, n_bits, query_b.shape[0]])
        testBinmatExpand = paddle.transpose(testBinmatExpand, [2, 1, 0])

        scores = paddle.sum((trainBinmat.astype('bool') ^ testBinmatExpand.astype('bool')).astype('float32'), axis=1)
        indices = paddle.arange(start=s_idx, end=e_idx, dtype='int64')
        indices = paddle.unsqueeze(indices, 0)
        indices = paddle.expand(indices, [n_test, numCandidates])

        topScores_list.append(scores)
        topIndices_list.append(indices)

    all_scores = paddle.concat(topScores_list, axis=1)
    all_indices = paddle.concat(topIndices_list, axis=1)

    sorted_scores, sorted_indices = paddle.topk(all_scores, k=topK, axis=1)
    retrieved_indices = paddle.gather_nd(all_indices, sorted_indices)

    return retrieved_indices


def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK, is_single_label=True):
    """计算 Top-K 精确率"""
    n_test = query_labels.shape[0]
    Indices = retrieved_indices[:, :topK]

    if is_single_label:
        test_labels = paddle.unsqueeze(query_labels, 1)
        test_labels = paddle.expand(test_labels, [n_test, topK])

        topTrainLabels_list = []
        for idx in range(n_test):
            selected = paddle.index_select(doc_labels.astype('int64'), paddle.to_tensor([Indices[idx, :].numpy()]))
            topTrainLabels_list.append(paddle.unsqueeze(selected, 0))
        topTrainLabels = paddle.concat(topTrainLabels_list, axis=0)
        relevances = (test_labels == topTrainLabels).astype('int16')
    else:
        topTrainLabels_list = []
        for idx in range(n_test):
            selected = paddle.index_select(doc_labels.astype('int64'), paddle.to_tensor([Indices[idx, :].numpy()]))
            topTrainLabels_list.append(paddle.unsqueeze(selected, 0))
        topTrainLabels = paddle.concat(topTrainLabels_list, axis=0).astype('int16')
        test_labels = paddle.unsqueeze(query_labels, 1).expand([n_test, topK, topTrainLabels.shape[-1]]).astype('int16')
        relevances = paddle.sum(topTrainLabels & test_labels, axis=2)
        relevances = (relevances > 0).astype('int16')

    true_positive = paddle.sum(relevances.astype('float32'), axis=1)
    true_positive = true_positive / topK
    prec_at_k = paddle.mean(true_positive)
    return prec_at_k
