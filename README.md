# GCL-NIDS: Phát hiện xâm nhập mạng bằng Graph Contrastive Learning

Repository này triển khai một framework cho **Network Intrusion Detection System (NIDS)** với các thành phần chính:
- **Graph Neural Networks (GNNs):** biểu diễn lưu lượng mạng dưới dạng đồ thị để nắm bắt quan hệ ngữ cảnh.  
- **Graph Contrastive Learning (GCL):** khai thác dữ liệu chưa gán nhãn nhằm cải thiện khả năng phát hiện.  
- **Explainable AI (XAI):** cung cấp khả năng giải thích quyết định của mô hình, hỗ trợ chuyên gia an ninh mạng.  

Đây là kết quả nghiên cứu trong bài báo gửi tới **VNICT 2025**.  

---

## 📂 Cấu trúc thư mục
GCL-NIDS/

─ GNN_NIDS_pipeline.ipynb # Pipeline cơ bản sử dụng GNN cho phát hiện xâm nhập

─ GCL_NIDS_pipeline.ipynb # Pipeline mở rộng với Graph Contrastive Learning

─ preprocess_dataset.py # Script tiền xử lý dữ liệu CSV thành .npy (không cần dùng vì đã tải sẵn .npy)

─ processed/ # dữ liệu .npy

─ README.md

─ .gitignore

---

## 📥 Dữ liệu

Thí nghiệm sử dụng các bộ dữ liệu:  
- **BoT-IoT (5% subset)**  
- **CICIDS2018**
- **UNSW-NB15**  

Do dung lượng lớn, dữ liệu không được đính kèm trong repo.  
Người dùng cần tải thủ công từ link sau:  

👉 **[Tải dữ liệu từ Google Drive](https://drive.google.com/file/d/1JbczFchDfqDEgEWxpI2OFSuBPh9k-fgP/view?usp=sharing)**  

Sau khi tải về:
- Giải nén dữ liệu trong `processed.zip`  