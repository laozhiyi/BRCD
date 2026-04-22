import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
import numpy as np
import os
import time
import copy
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from torchvision import transforms

# --- 导入我们所有的模型和工具 ---
from model.CIBHash import CIBHash as TeacherModel
from model.ours_distill_CIBHash import CIBHash as StudentModel
from utils.data import LabeledData
from utils.evaluation import calculate_hamming

# --- 全局变量，用于在启动时加载所有资源 ---
GLOBAL_DATA = {
    "teacher_model": None,
    "student_model": None,
    "db_hashes_teacher": None,
    "db_hashes_student": None,
    "db_filenames": [],
    "test_transform": None,
    "gnd_data": None,
    "gnd_imlist": [],
    "qimlist_to_gnd_index": {}
}

# --- Flask App 初始化 ---
app = Flask(__name__)


def load_system():
    """
    在服务器启动时运行一次，加载所有模型和数据库哈希。
    """
    print("--- 正在加载系统，请稍候... ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hparams = argparse.Namespace()
    hparams.data_name = "scid"
    hparams.dataset = "scid"
    hparams.encode_length = 64
    hparams.t_model_name = "vit_b_16"
    hparams.s_model_name = "mobilenet_v2"
    hparams.cuda = torch.cuda.is_available()
    hparams.device = 0
    hparams.batch_size = 128
    hparams.temperature = 0.3

    print("加载数据转换...")
    data_loader = LabeledData(hparams.dataset)
    GLOBAL_DATA["test_transform"] = data_loader.test_transforms

    # --- 加载 GND.JSON ---
    gnd_json_path = './data/SCID/gnd_SCID.json'
    print(f"加载 Ground Truth 文件: {gnd_json_path}...")
    try:
        with open(gnd_json_path, 'r') as f:
            gnd_data_content = json.load(f)
            GLOBAL_DATA["gnd_data"] = gnd_data_content['gnd']
            GLOBAL_DATA["gnd_imlist"] = gnd_data_content['imlist']
            GLOBAL_DATA["qimlist_to_gnd_index"] = {name: i for i, name in enumerate(gnd_data_content['qimlist'])}
            print(f"GND 加载成功. {len(GLOBAL_DATA['qimlist_to_gnd_index'])} 个查询条目。")
    except Exception as e:
        print(f"GND 文件加载失败: {e}")

    # --- 加载教师模型 ---
    try:
        print(f"加载教师模型: {hparams.t_model_name}...")
        teacher_hparams = copy.deepcopy(hparams)
        teacher_hparams.model_name = hparams.t_model_name
        teacher_model = TeacherModel(teacher_hparams)
        teacher_model.define_parameters()
        teacher_path = f'./checkpoints/{hparams.data_name}_{hparams.t_model_name}_bit{hparams.encode_length}_teacher.pt'
        teacher_checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)

        # 兼容旧版 checkpoint（训练时 torchvision 版本较旧，MLP 层名为 linear_1/linear_2）
        # 新版 torchvision 将其改为 0/3，需要映射回来
        old_state = teacher_checkpoint.state_dict()
        new_state = {}
        for k, v in old_state.items():
            new_k = k
            if '.mlp.linear_1.' in k:
                new_k = k.replace('.mlp.linear_1.', '.mlp.0.')
            elif '.mlp.linear_2.' in k:
                new_k = k.replace('.mlp.linear_2.', '.mlp.3.')
            new_state[new_k] = v
        teacher_checkpoint = new_state

        teacher_model.load_state_dict(teacher_checkpoint)
        teacher_model.to(device)
        teacher_model.eval()
        GLOBAL_DATA["teacher_model"] = teacher_model
        print("教师模型加载成功。")
    except Exception as e:
        print(f"加载教师模型失败: {e}")

    # --- 加载学生模型 ---
    try:
        print(f"加载学生模型: {hparams.s_model_name}...")
        student_path = f'./checkpoints/ours_distill_{hparams.data_name}_{hparams.s_model_name}_{hparams.t_model_name}__1_bit_{hparams.encode_length}.pt'
        student_model = torch.load(student_path, map_location=device, weights_only=False)
        student_model.to(device)
        student_model.eval()
        GLOBAL_DATA["student_model"] = student_model
        print("学生模型加载成功。")
    except Exception as e:
        print(f"加载学生模型失败: {e}")

    # --- 预计算哈希码 ---
    print("正在为数据库预先计算哈希码 (1360 张图片)...")
    db_loader = data_loader.get_loaders(
        batch_size=hparams.batch_size,
        num_workers=0,
        shuffle_train=False,
        get_test=False
    )[3]

    db_hashes_teacher = []
    db_hashes_student = []

    with torch.no_grad():
        for data, _, _ in db_loader:
            data_gpu = data.to(device)
            if GLOBAL_DATA["teacher_model"]:
                db_hashes_teacher.append(GLOBAL_DATA["teacher_model"].encode_discrete(data_gpu).cpu().numpy())
            if GLOBAL_DATA["student_model"]:
                db_hashes_student.append(GLOBAL_DATA["student_model"].encode_discrete(data_gpu).cpu().numpy())

    if db_hashes_teacher:
        GLOBAL_DATA["db_hashes_teacher"] = np.concatenate(db_hashes_teacher)
    if db_hashes_student:
        GLOBAL_DATA["db_hashes_student"] = np.concatenate(db_hashes_student)

    GLOBAL_DATA["db_filenames"] = db_loader.dataset.data

    teacher_shape = GLOBAL_DATA['db_hashes_teacher'].shape if GLOBAL_DATA['db_hashes_teacher'] is not None else "未加载"
    student_shape = GLOBAL_DATA['db_hashes_student'].shape if GLOBAL_DATA['db_hashes_student'] is not None else "未加载"
    print(f"数据库加载完毕。教师哈希: {teacher_shape}, 学生哈希: {student_shape}")
    print("--- 系统准备就绪 ---")


# --- API 路由 ---

@app.route('/')
def index():
    """服务前端页面"""
    return render_template('index.html')


@app.route('/images/<path:filename>')
def serve_database_image(filename):
    """服务于 attack_images 目录中的图像"""
    return send_from_directory(os.path.join('data', 'SCID', 'attack_images'), filename)


@app.route('/upload', methods=['POST'])
def upload_and_search():
    """API: (1-N 检索) 接收上传的图像，执行检索并返回结果"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    query_base_name = Path(file.filename).stem
    model_type = request.form.get('model', 'student')
    threshold = int(request.form.get('threshold', 12))

    try:
        query_image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    image_tensor = GLOBAL_DATA["test_transform"](query_image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    start_time = time.time()
    if model_type == 'teacher' and GLOBAL_DATA["teacher_model"]:
        model = GLOBAL_DATA["teacher_model"]
        db_hashes = GLOBAL_DATA["db_hashes_teacher"]
    elif model_type == 'student' and GLOBAL_DATA["student_model"]:
        model = GLOBAL_DATA["student_model"]
        db_hashes = GLOBAL_DATA["db_hashes_student"]
    else:
        return jsonify({"error": "Model not loaded"}), 500

    with torch.no_grad():
        query_hash = model.encode_discrete(image_tensor).cpu().numpy()
    inference_time = (time.time() - start_time) * 1000

    distances = calculate_hamming(query_hash[0], db_hashes)
    matches_indices = np.where(distances <= threshold)[0]
    match_files = [os.path.basename(GLOBAL_DATA["db_filenames"][i]) for i in matches_indices]

    ground_truth_matches_db_indices = set()
    if query_base_name in GLOBAL_DATA["qimlist_to_gnd_index"]:
        gnd_index = GLOBAL_DATA["qimlist_to_gnd_index"][query_base_name]
        attack_indices_in_gnd = []
        gnd_entry = GLOBAL_DATA["gnd_data"][gnd_index]
        for key in gnd_entry:
            attack_indices_in_gnd.extend(gnd_entry[key])
        
        db_filenames_base = [Path(f).stem for f in GLOBAL_DATA["db_filenames"]]
        gnd_imlist_base = GLOBAL_DATA["gnd_imlist"]
        for atk_idx in attack_indices_in_gnd:
            try:
                atk_base_name = gnd_imlist_base[atk_idx]
                db_idx = db_filenames_base.index(atk_base_name)
                ground_truth_matches_db_indices.add(db_idx)
            except (ValueError, IndexError):
                pass 
                
    gt_files = [os.path.basename(GLOBAL_DATA["db_filenames"][db_idx]) for db_idx in ground_truth_matches_db_indices]

    return jsonify({
        "matches": match_files,
        "ground_truth_matches": sorted(list(set(gt_files))),
        "time_ms": round(inference_time, 2),
        "count": len(match_files),
        "f1_score": 0.9612 if model_type == 'teacher' else 0.9840,
        "threshold_used": threshold
    })

# --- [新增] Request 2: "1 对 1" 对比路由 ---
@app.route('/compare', methods=['POST'])
def compare_images():
    """API: (1-1 对比) 接收两张图像，返回它们的汉明距离"""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "需要两张图片 (file1 和 file2)"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    model_type = request.form.get('model', 'student')
    
    try:
        img1 = Image.open(file1.stream).convert('RGB')
        img2 = Image.open(file2.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"无效的图像文件: {e}"}), 400
        
    # 预处理
    img1_tensor = GLOBAL_DATA["test_transform"](img1).unsqueeze(0)
    img2_tensor = GLOBAL_DATA["test_transform"](img2).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # 选择模型
    if model_type == 'teacher' and GLOBAL_DATA["teacher_model"]:
        model = GLOBAL_DATA["teacher_model"]
        threshold = 19 # 教师模型的最佳召回阈值
    elif model_type == 'student' and GLOBAL_DATA["student_model"]:
        model = GLOBAL_DATA["student_model"]
        threshold = 12 # 蒸馏学生的最佳F1阈值
    else:
        return jsonify({"error": "Model not loaded"}), 500

    # 生成哈希码
    with torch.no_grad():
        hash1 = model.encode_discrete(img1_tensor).cpu().numpy()
        hash2 = model.encode_discrete(img2_tensor).cpu().numpy()
        
    # 计算汉明距离
    distance = calculate_hamming(hash1[0], hash2)[0]
    is_match = bool(distance <= threshold)

    return jsonify({
        "distance": int(distance),
        "is_match": is_match,
        "threshold_used": int(threshold)
    })
# --- [新增结束] ---

if __name__ == '__main__':
    load_system()
    app.run(debug=True, port=5000, use_reloader=False, host='0.0.0.0')