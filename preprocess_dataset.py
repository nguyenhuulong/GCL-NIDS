import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_bot(bot_folder, out_folder, test_size=0.2, random_state=42):
    # Merge CSV
    files = [os.path.join(bot_folder, f"reduced_data_{i}.csv") for i in range(1, 5)]
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    
    # Encode label từ cột "attack"
    if "attack" in df.columns:
        y = df["attack"].astype(int).values
        df = df.drop(columns=["attack"])
    else:
        raise ValueError("Không tìm thấy cột 'attack' trong BoT-IoT CSV")
    
    # Drop category/subcategory nếu không dùng
    drop_cols = ["category", "subcategory", "pkSeqID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # --- Encode categorical ---
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Convert về float
    X = df.values.astype(float)
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Lưu npy
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, "X_train_bot.npy"), X_train)
    np.save(os.path.join(out_folder, "y_train_bot.npy"), y_train)
    np.save(os.path.join(out_folder, "X_test_bot.npy"), X_test)
    np.save(os.path.join(out_folder, "y_test_bot.npy"), y_test)
    
    print(f"[BoT-IoT] Saved to {out_folder}, train={X_train.shape}, test={X_test.shape}")
    
    
def preprocess_cic18(cic18_folder, out_folder, test_size=0.2, random_state=42, chunksize=500_000):
    import time
    start_time = time.time()

    files = sorted([os.path.join(cic18_folder, f)
                   for f in os.listdir(cic18_folder) if f.endswith(".csv")])
    print(f"[CIC18] Tìm thấy {len(files)} file CSV, bắt đầu đọc...")

    df_list = []
    for i, f in enumerate(files, 1):
        print(f"  -> Đang đọc file {i}/{len(files)}: {os.path.basename(f)}")

        # Với file lớn, dùng chunksize
        if os.path.getsize(f) > 1_000_000_000:  # >1GB thì đọc theo chunk
            print(
                f"     [!] File lớn ({os.path.getsize(f)/1e9:.1f} GB), đọc theo chunksize={chunksize}")
            for j, chunk in enumerate(pd.read_csv(f, low_memory=False, chunksize=chunksize)):
                print(f"       - chunk {j}, shape={chunk.shape}")
                df_list.append(chunk)
        else:
            chunk = pd.read_csv(f, low_memory=False)
            df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)
    print(
        f"[CIC18] Đọc xong toàn bộ CSV, shape={df.shape}, mất {time.time()-start_time:.1f}s")

    # Label column
    if "Label" not in df.columns:
        raise ValueError("Không tìm thấy cột 'Label' trong CICIDS2018")
    y = df["Label"].apply(
        lambda x: 0 if "BENIGN" in str(x).upper() else 1).values
    df = df.drop(columns=["Label"])

    # Encode categorical
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"  -> Encode cột {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Convert về float
    X = df.values.astype(float)
    print(f"[CIC18] Convert sang float xong, shape={X.shape}")

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("[CIC18] Scale dữ liệu xong")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[CIC18] Split train={X_train.shape}, test={X_test.shape}")

    # Save
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, "X_train_cic18.npy"), X_train)
    np.save(os.path.join(out_folder, "y_train_cic18.npy"), y_train)
    np.save(os.path.join(out_folder, "X_test_cic18.npy"), X_test)
    np.save(os.path.join(out_folder, "y_test_cic18.npy"), y_test)

    print(
        f"[CIC18] Saved to {out_folder}, tổng thời gian {time.time()-start_time:.1f}s")


import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_unsw(unsw_folder, out_folder):
    # Load npy gốc
    train_path = os.path.join(unsw_folder, "train.npy")
    test_path = os.path.join(unsw_folder, "test.npy")

    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    # Nếu file chứa tuple (X, y)
    if isinstance(train_data, tuple) or train_data.shape[1] > 50:  # tùy theo feature count
        try:
            X_train, y_train = train_data
            X_test, y_test = test_data
        except:
            # Có thể data là 2D: (N, d+1) với cột cuối là label
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]
    else:
        raise ValueError("Không xác định được format UNSW, cần kiểm tra file .npy")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save lại theo format chuẩn
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, "X_train_unsw.npy"), X_train)
    np.save(os.path.join(out_folder, "y_train_unsw.npy"), y_train)
    np.save(os.path.join(out_folder, "X_test_unsw.npy"), X_test)
    np.save(os.path.join(out_folder, "y_test_unsw.npy"), y_test)

    print(f"[UNSW] Saved to {out_folder}, train={X_train.shape}, test={X_test.shape}")


if __name__ == "__main__":
    # preprocess_bot("./bot", "./processed/bot")
    # preprocess_cic18("./cic18", "./processed/cic18")
    preprocess_unsw("./unsw", "./processed/unsw")
