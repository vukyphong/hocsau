# Học sâu (Deep Learning)

Các bài tập môn học Deep Learning.

- **Sinh viên:** Vũ Kỳ Phong
- **Lớp:** KHDL2

> ⭐ **Bài tập hết môn là bài ANN: Dự đoán bệnh tim bằng mạng nơ-ron nhân tạo**, nằm trong thư mục [`baihetmon/`](baihetmon/).
> Các thư mục trong [`baitaptrenlop/`](baitaptrenlop/) là bài tập làm trên lớp trong quá trình học.

## Cấu trúc thư mục

```
hocsau/
├── README.md
├── baihetmon/                  ⭐ BÀI TẬP HẾT MÔN (ANN)
│   ├── Heart_Disease_Prediction_using_ANN.ipynb
│   ├── Dataset--Heart-Disease-Prediction-using-ANN.csv
│   └── BaoCao_DuDoanBenhTim_ANN.docx   Báo cáo bài hết môn
└── baitaptrenlop/              Bài tập trên lớp
    ├── L1-L15/                 Chuỗi bài L1 → L15 (hồi quy, perceptron, NN, CNN, ứng dụng)
    ├── backpropagation/        Lan truyền tiến / lan truyền ngược
    ├── Optimizer/              Gradient Descent, SGD
    ├── Regularization/         L1, L2, ElasticNet
    ├── CNN/                    Mạng nơ-ron tích chập
    └── RNN/                    RNN, LSTM, GRU
```

---

## ⭐ Bài tập hết môn: Dự đoán bệnh tim bằng ANN

**Thư mục:** [`baihetmon/`](baihetmon/)

| File | Nội dung |
|---|---|
| [`Heart_Disease_Prediction_using_ANN.ipynb`](baihetmon/Heart_Disease_Prediction_using_ANN.ipynb) | Notebook xây dựng, huấn luyện và đánh giá mạng ANN |
| [`Dataset--Heart-Disease-Prediction-using-ANN.csv`](baihetmon/Dataset--Heart-Disease-Prediction-using-ANN.csv) | Bộ dữ liệu bệnh tim (303 mẫu, 13 đặc trưng + nhãn) |
| [`BaoCao_DuDoanBenhTim_ANN.docx`](baihetmon/BaoCao_DuDoanBenhTim_ANN.docx) | Báo cáo Word: bìa, mục lục, danh mục, đặt vấn đề, cơ sở lý thuyết, giải thích từng bước, kết quả – thảo luận, tài liệu tham khảo (Vancouver), phụ lục mã nguồn |

### Bài toán

Phân loại nhị phân: dựa vào các chỉ số lâm sàng của bệnh nhân, dự đoán người đó **có bệnh tim (`target = 1`)** hay **không (`target = 0`)**.

### Dữ liệu

303 dòng, không có giá trị thiếu. 13 cột đặc trưng:

| Cột | Ý nghĩa |
|---|---|
| `age` | Tuổi |
| `sex` | Giới tính (1 = nam, 0 = nữ) |
| `cp` | Loại đau ngực (0–3) |
| `trestbps` | Huyết áp lúc nghỉ (mm Hg) |
| `chol` | Cholesterol huyết thanh (mg/dl) |
| `fbs` | Đường huyết lúc đói > 120 mg/dl (1 = có) |
| `restecg` | Kết quả điện tâm đồ lúc nghỉ (0–2) |
| `thalach` | Nhịp tim tối đa đạt được |
| `exang` | Đau thắt ngực khi gắng sức (1 = có) |
| `oldpeak` | Mức chênh ST khi gắng sức so với lúc nghỉ |
| `slope` | Độ dốc đoạn ST khi gắng sức (0–2) |
| `ca` | Số mạch máu chính nhuộm màu huỳnh quang (0–4) |
| `thal` | Thalassemia (0–3) |
| `target` | **Nhãn**: 1 = có bệnh tim, 0 = không |

### Các bước trong notebook

1. **Import thư viện:** TensorFlow/Keras, pandas, numpy, matplotlib, scikit-learn.
2. **Đọc và khám phá dữ liệu:** `read_csv`, `head()`, `describe()`, kiểm tra giá trị thiếu bằng `isnull().any()`.
3. **Tách đặc trưng và nhãn:** `X` = 13 cột đầu, `y` = cột `target`.
4. **Chia tập train/test:** `train_test_split` theo tỉ lệ 70/30, `random_state = 0`.
5. **Chuẩn hóa dữ liệu:** `StandardScaler` (fit trên tập train, transform tập test).
6. **Xây dựng mô hình ANN** (Keras `Sequential`):

   | Lớp | Số nơ-ron | Hàm kích hoạt |
   |---|---|---|
   | Input | 13 | — |
   | Ẩn 1 (`Dense`) | 8 | ReLU |
   | Ẩn 2 (`Dense`) | 14 | ReLU |
   | Output (`Dense`) | 1 | Sigmoid |

   - Khởi tạo trọng số: `uniform`
   - Optimizer: `adam`, hàm mất mát: `binary_crossentropy`, metric: `accuracy`
7. **Huấn luyện:** `batch_size = 8`, `epochs = 100`.
8. **Dự đoán:** xác suất đầu ra > 0.5 thì gán là có bệnh.
9. **Đánh giá:** ma trận nhầm lẫn (`confusion_matrix`) và độ chính xác.

### Kết quả (theo output đã lưu trong notebook)

Ma trận nhầm lẫn trên tập test (91 mẫu):

|  | Dự đoán 0 | Dự đoán 1 |
|---|---|---|
| **Thực tế 0** | 37 | 7 |
| **Thực tế 1** | 4 | 43 |

- **Độ chính xác trên tập test: ≈ 87.9%**
- Độ chính xác trên tập train ở các epoch cuối: khoảng 90–94%

### Cách chạy

Notebook được viết cho **Google Colab** và đọc dữ liệu từ đường dẫn `/content/Dataset--Heart-Disease-Prediction-using-ANN.csv`:

1. Mở notebook trên Google Colab.
2. Tải file `Dataset--Heart-Disease-Prediction-using-ANN.csv` lên thư mục `/content/` (tab *Files*).
3. Chạy lần lượt các ô (*Runtime → Run all*).

> Nếu chạy trên máy cá nhân, đặt notebook và file CSV cùng thư mục `baihetmon/`, rồi đổi đường dẫn trong ô đọc dữ liệu thành `'Dataset--Heart-Disease-Prediction-using-ANN.csv'`.

---

## Bài tập trên lớp

Thư mục: [`baitaptrenlop/`](baitaptrenlop/)

### `L1-L15/` – Chuỗi bài học theo buổi

| Bài | File | Nội dung |
|---|---|---|
| L1 | `l1.py`, `data_linear.csv`, `data_square.csv` | Hồi quy tuyến tính dự đoán giá nhà theo diện tích, cài đặt gradient descent bằng numpy |
| L2 | `l2.py`, `dataset.csv` | Hồi quy logistic (hàm sigmoid) cho bài toán cho vay / từ chối |
| L3 | `l3.py`, `bitwise.xlsx`, `latex*.txt` | Biểu diễn các phép toán logic AND, OR, XOR bằng perceptron; công thức LaTeX |
| L4 | `l4.py`, `dataset.csv`, `xor.csv` | Tự cài đặt mạng nơ-ron nhiều lớp (lan truyền tiến, lan truyền ngược) bằng numpy |
| L5 | `l5.py` | Đọc và xử lý ảnh xám cơ bản với OpenCV |
| L7 | `mnist.ipynb`, `mnist.html` | Mạng CNN nhận diện chữ số viết tay MNIST bằng Keras |
| L8 | `self-driving-car.ipynb`, `drive.py`, `utils.py`, `model.h5` | Xe tự lái: CNN dự đoán góc lái từ ảnh camera; `drive.py` kết nối model với simulator |
| L12 | `model.py`, `data.py`, `main.py`, `trainUnet.ipynb`, `dataPrepare.ipynb` | Phân vùng ảnh (segmentation) bằng U-Net, có tăng cường dữ liệu (data augmentation) |
| L15 | `Automatic Image Captioning.ipynb` | Sinh mô tả cho ảnh: InceptionV3 trích đặc trưng + LSTM sinh câu |

### `backpropagation/` – Lan truyền tiến và lan truyền ngược

| File | Nội dung |
|---|---|
| `What_is_Forward_Propagation_in_Neural_Networks.ipynb` | Minh họa lan truyền tiến với một nơ-ron sigmoid |
| `Backpropagation_in_Neural_Network.ipynb` | Mạng nơ-ron 1 lớp ẩn, cài đặt lan truyền ngược bằng numpy |
| `6.3.ipynb` | Lớp `NeuralNetwork` tổng quát (nhiều lớp) huấn luyện bằng lan truyền ngược |
| `Untitled0.ipynb` | Phân loại dữ liệu xoắn ốc 3 lớp bằng mạng nơ-ron + softmax |

### `Optimizer/` – Thuật toán tối ưu

| File | Nội dung |
|---|---|
| `GradientDescent.ipynb` | Gradient Descent kết hợp lan truyền ngược trên mạng nơ-ron 1 lớp ẩn |
| `SGD.ipynb` | Stochastic Gradient Descent |

### `Regularization/` – Chính quy hóa

| File | Nội dung |
|---|---|
| `L1&L2.ipynb` | Chống overfitting với Lasso (L1), Ridge (L2) và ElasticNet |

### `CNN/` – Mạng nơ-ron tích chập

| File | Nội dung |
|---|---|
| `1D.ipynb` | Tự cài đặt tích chập 1D, lọc nhiễu tín hiệu sin |
| `2D.ipynb` | Tự cài đặt tích chập 2D (padding, stride) |
| `Convolution_Neural_Network.ipynb` | Áp dụng bộ lọc (kernel) lên ảnh bằng `tf.nn.conv2d` |
| `CNN.ipynb`, `main.py` | Huấn luyện mô hình trên MNIST bằng PyTorch |
| `digits.ipynb` | CNN nhận diện chữ số MNIST + giao diện demo Streamlit chạy trên Colab |
| `catdog.ipynb` | CNN phân loại ảnh chó / mèo (dữ liệu trên Google Drive) |
| `Lung_Cancer_Detection_using_Convolutional_Neural_Network.ipynb` | CNN phát hiện ung thư phổi từ ảnh mô bệnh học (3 lớp) |

### `RNN/` – Mạng nơ-ron hồi quy

| File | Nội dung |
|---|---|
| `lab1.ipynb` | Giới thiệu RNN, sinh văn bản bằng `SimpleRNN` |
| `lab2.ipynb` | Giới thiệu LSTM (Long Short-Term Memory) |
| `lab3.ipynb` | GRU (Gated Recurrent Unit) dự đoán chuỗi thời gian |

---

## Môi trường

- Python 3, Jupyter Notebook / Google Colab
- Thư viện chính: `tensorflow` / `keras`, `torch`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `opencv-python`
- Một số bài cũ (L7, L8, L12, L15) viết cho Keras phiên bản cũ nên có thể cần chỉnh import khi chạy với TensorFlow mới.
