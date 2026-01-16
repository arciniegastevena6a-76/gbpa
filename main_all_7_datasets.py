import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# --- 导入模块 ---
from GBPA_Module import GBPA_Module
from FGS_Module import FGS_Module
#  数据集切换器
CURRENT_DATASET = 'wdbc'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_local_data(name):
    print(f"1. Loading Dataset: {name.upper()} from '{DATA_DIR}'...")
    
    X, y = None, None
    target_k = 0
    known_ratio = 0.7

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Folder '{DATA_DIR}' not found.")
        return None, None, None, None, None

    try:
        # 1. IRIS
        if name == 'iris':
            d = load_iris()
            X, y = d.data, d.target
            target_k = 3
            known_ratio = 0.6

        # 2. SEEDS
        elif name == 'seeds':
            file_path = os.path.join(DATA_DIR, 'seeds')
            if not os.path.exists(file_path): file_path += '.txt'
            if not os.path.exists(file_path): file_path = os.path.join(DATA_DIR, 'seeds_dataset.txt')
            
            df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            target_k = 3
            known_ratio = 0.7
        # 3. HABERMAN
        elif name == 'haberman':
            file_path = os.path.join(DATA_DIR, 'haberman.data')
            df = pd.read_csv(file_path, sep=',', header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            target_k = 2
            known_ratio = 0.5
        # 4. GLASS
        elif name == 'glass':
            file_path = os.path.join(DATA_DIR, 'glass.data')
            df = pd.read_csv(file_path, sep=',', header=None)
            X = df.iloc[:, 1:-1].values 
            y = df.iloc[:, -1].values
            target_k = 6
            known_ratio = 0.7
        # 5. WDBC
        elif name == 'wdbc':
            file_path = os.path.join(DATA_DIR, 'wdbc.data')
            if not os.path.exists(file_path):
                from sklearn.datasets import load_breast_cancer
                d = load_breast_cancer()
                X, y = d.data, d.target
            else:
                df = pd.read_csv(file_path, sep=',', header=None)
                y = df.iloc[:, 1].values
                X = df.iloc[:, 2:].values
            target_k = 2
            known_ratio = 0.8

        # 6. ROBOT
        elif name == 'robot':
            file_path = os.path.join(DATA_DIR, 'robot.data')
            df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python') 
            if df.shape[1] == 1: df = pd.read_csv(file_path, sep=',', header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            target_k = 4
            known_ratio = 0.8
        # 7. KNOWLEDGE 
        elif name == 'knowledge':
            all_files = os.listdir(DATA_DIR)
            target_files = [f for f in all_files if 'knowledge' in f.lower() or 'data_user' in f.lower()]
            
            if not target_files:
                print(f" No Knowledge file found in data folder!")
                return None, None, None, None, None
            
            target_file = target_files[0]
            file_path = os.path.join(DATA_DIR, target_file)
            print(f"   > 🎯 Found file: {target_file}")

            # === 智能判断文件类型 ===
            if target_file.lower().endswith(('.xls', '.xlsx')):
                # Excel 模式
                print("   > Mode: Excel")
                xls = pd.ExcelFile(file_path)
                target_sheet = xls.sheet_names[0]
                if len(xls.sheet_names) > 1: target_sheet = xls.sheet_names[1]
                df = pd.read_excel(xls, sheet_name=target_sheet)
            else:
                # CSV/文本 模式 (针对 .csv, .txt 等)
                print("   > Mode: CSV/Text")
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python')
                except:
                    df = pd.read_csv(file_path) 
            # 数据提取
            if 'UNS' in df.columns:
                print("   > Found column 'UNS' as target.")
                y = df['UNS'].values
                X = df.drop(columns=['UNS']).values
            else:
                print("   > Column 'UNS' not found, using last column.")
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
            target_k = 4
            known_ratio = 0.7

        else:
            raise ValueError(f"Dataset '{name}' not configured.")

    except Exception as e:
        print(f"\n❌ [Error] Loading failed for {name}: {e}")
        return None, None, None, None, None

    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f"   > Loaded. Shape: {X.shape}. Classes: {len(np.unique(y))}")
    
    all_classes = np.unique(y)
    np.random.seed(42) 
    unknown_cls = np.random.choice(all_classes)
    known_classes = [c for c in all_classes if c != unknown_cls]
    
    train_idx = []
    test_idx = []
    
    for c in all_classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        if c in known_classes:
            split = int(len(idx) * known_ratio)
            train_idx.extend(idx[:split])
            test_idx.extend(idx[split:])
        else:
            test_idx.extend(idx)
            
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_train_str = np.array([f"Class_{i}" for i in y_train])
    
    print(f"   - Train: {len(X_train)} (Known: {known_classes})")
    print(f"   - Test:  {len(X_test)} (Hidden: {unknown_cls})")
    
    return X_train, y_train_str, X_test, target_k, len(all_classes)

def main():
    print("==============================================")
    print(f"   Current Target: {CURRENT_DATASET.upper()}")
    print("==============================================")
    
    try:
        with open('config.json', 'r') as f: config = json.load(f)
    except: return

    train_data, train_labels, test_data, target_k, true_unknown_count = load_local_data(CURRENT_DATASET)
    if train_data is None: return

    print("\n2. Step 1: GBPA Detection...")
    gbpa = GBPA_Module()
    models = gbpa.train_models(train_data, train_labels)
    _, m_empty = gbpa.calculate_gbpa(test_data, models)
    print(f"   - Avg m(empty): {m_empty:.4f} (Threshold: {config['gbpa']['p_threshold']})")

    print(f"\n3. Step 2: FGS ...")
    print(f"   - Expecting Optimal K = {true_unknown_count}")
    
    fgs = FGS_Module(config)
    opt_k, gap, sk = fgs.run_fgs(test_data)
    print(f"\n[Step 2 Result] Optimal k = {opt_k}")
    
    plt.figure(f"FGS - {CURRENT_DATASET.upper()}")
    plt.errorbar(range(1, len(gap)+1), gap, yerr=sk, fmt='-bo')
    plt.plot(opt_k, gap[opt_k-1], 'rx', markersize=12)
    plt.title(f"{CURRENT_DATASET} (True={true_unknown_count}, Found={opt_k})")
    plt.grid(True)
    
    print("\n4. Step 3: Updating...")
    best_J = float('inf')
    U_final = None
    for _ in range(3):
        _, U, J = fgs.my_fcm(test_data, opt_k, config['fcm']['m_exponent'], 100, 1e-5)
        if J < best_J: best_J = J; U_final = U
    
    preds = np.argmax(U_final, axis=0)
    
    new_data = []
    new_lbls = []
    for k in range(opt_k):
        idx = np.where(preds == k)[0]
        if len(idx)>0:
            new_data.append(test_data[idx])
            new_lbls.extend([f"New_{k}"]*len(idx))
            
    if new_data:
        upd_X = np.vstack([train_data, np.vstack(new_data)])
        upd_y = np.concatenate([train_labels, np.array(new_lbls)])
        print(f"   - Database updated: {len(train_data)} -> {len(upd_X)}")
        
        new_models = gbpa.train_models(upd_X, upd_y)
        _, new_m = gbpa.calculate_gbpa(test_data, new_models)
        print(f"   - Final Conflict: {new_m:.4f}")
        
        if new_m <= config['gbpa']['p_threshold']:
            print("✅ Success: Conflict Resolved.")
        else:
            print("⚠️ Warning: Conflict still high.")

    fig = plt.figure(f"Clusters - {CURRENT_DATASET.upper()}")
    ax = fig.add_subplot(111, projection='3d')
    dim_x, dim_y, dim_z = 0, 1, 2
    if train_data.shape[1] < 3: dim_z = 0
    skip = max(1, len(train_data)//500)
    ax.scatter(train_data[::skip, dim_x], train_data[::skip, dim_y], train_data[::skip, dim_z], c='gray', alpha=0.3, label='Known')
    colors = plt.cm.tab10(np.linspace(0, 1, opt_k))
    for k in range(opt_k):
        idx = np.where(preds == k)[0]
        if len(idx)>0:
            pts = test_data[idx]
            skip_p = max(1, len(pts)//500)
            ax.scatter(pts[::skip_p, dim_x], pts[::skip_p, dim_y], pts[::skip_p, dim_z], color=colors[k], label=f'C{k}')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()