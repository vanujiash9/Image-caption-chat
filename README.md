# ViCaptionTalk 🤖 - Mô hình AI Chú thích Ảnh Tiếng Việt

Đây là một dự án nghiên cứu và phát triển một hệ thống AI tiên tiến có khả năng tự động tạo ra các câu chú thích (caption) bằng tiếng Việt một cách tự nhiên và chính xác cho hình ảnh đầu vào. Dự án này là hành trình từ việc xây dựng một mô hình cơ sở, phân tích điểm yếu, đến việc tối ưu hóa toàn diện để đạt được hiệu suất cao.

  <!-- Bạn có thể thay link này bằng một ảnh chụp màn hình web demo của bạn -->

---

## 🏆 Thành tựu Nổi bật

Sau nhiều lần thử nghiệm và tối ưu, phiên bản mô hình cuối cùng đã đạt được hiệu suất rất ấn tượng trên tập dữ liệu test:

| Chỉ số       | Điểm số    |
|--------------|------------|
| 🔷 **BLEU**      | **35.93%** |
| 🔶 **ROUGE-L**   | **50.23%** |
| 💚 **METEOR**    | **52.41%** |

Kết quả này cho thấy mô hình không chỉ nắm bắt được các đối tượng chính trong ảnh mà còn có khả năng tạo ra các câu văn có cấu trúc, ngữ pháp và ngữ nghĩa gần với cách diễn đạt của con người.

---

## 🛠️ Kiến trúc Mô hình

Hệ thống được xây dựng trên kiến trúc **Encoder-Decoder**, kết hợp sức mạnh của các mô hình nền tảng hàng đầu:

1.  **👁️ Vision Encoder (Bộ mã hóa Thị giác): `Swin Transformer`**
    *   Sử dụng `swin_base_patch4_window7_224.ms_in22k` đã được huấn luyện trước.
    *   Đóng vai trò như "đôi mắt" của hệ thống, trích xuất các đặc trưng hình ảnh đa cấp độ một cách hiệu quả.

2.  **✍️ Language Decoder (Bộ giải mã Ngôn ngữ): `PhoBERT` + `Transformer Decoder`**
    *   Sử dụng `vinai/phobert-base`, mô hình ngôn ngữ chuyên sâu cho tiếng Việt, làm nền tảng.
    *   Một bộ `Transformer Decoder` tiêu chuẩn (6 lớp) sẽ nhận thông tin từ cả Vision Encoder và các từ đã được sinh ra trước đó để tạo ra từ tiếp theo, hình thành một câu chú thích hoàn chỉnh.

 <!-- Bạn có thể tạo một sơ đồ đơn giản và thay link ảnh vào đây -->

---

## 🚀 Hướng dẫn Sử dụng

### 1. Yêu cầu hệ thống

*   Python 3.9+
*   PyTorch & TorchVision
*   Các thư viện được liệt kê trong `requirements.txt`
*   GPU có VRAM >= 8GB để chạy demo (khuyến nghị)

### 2. Cài đặt

1.  **Clone kho lưu trữ này:**
    ```bash
    git clone https://github.com/vanujiash9/chatmovie.git
    cd chatmovie
    ```

2.  **(Khuyến nghị)** Tạo và kích hoạt một môi trường ảo:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Linux/macOS
    # hoặc
    venv\Scripts\activate  # Trên Windows
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chuẩn bị Dữ liệu và Model:**
    *   Tải các file trọng số của model đã huấn luyện (`best_model.pth`) và đặt chúng vào thư mục `checkpoints_phobert/`.
    *   (Tùy chọn cho training/evaluation) Chuẩn bị dữ liệu theo cấu trúc được mô tả trong `config/config_main.yaml` và đặt vào thư mục `data/`.

### 3. Chạy Web Demo

Để trải nghiệm mô hình trên giao diện web, chạy lệnh sau từ thư mục gốc của dự án:
```bash
python app.py
```
Sau đó, mở trình duyệt và truy cập vào địa chỉ `http://127.0.0.1:5000`. Kéo thả một ảnh vào và xem kết quả!

### 4. Test trên một ảnh đơn lẻ

Sử dụng script `predict.py` để nhanh chóng tạo chú thích cho một ảnh từ dòng lệnh:
```bash
python predict.py --image_path duong_dan/den/anh_cua_ban.jpg
```

---

## 🔬 Hành trình Thử nghiệm và Các Bài học Kinh nghiệm

Dự án đã trải qua 3 phiên bản thử nghiệm chính:

1.  **v1 (The Baseline):** Sử dụng `ViT` + `BERT-multilingual`. Đạt `~31.3% BLEU` nhưng bị overfitting sớm.
2.  **v2 (The Over-Correction):** Thử nghiệm chống overfitting "quá tay" với `dropout` và `weight_decay` cao. Kết quả thấp hơn (`~22.3% BLEU`), cho thấy tầm quan trọng của việc tìm ra sự cân bằng.
3.  **v3 (The Ultimate):** Phiên bản đột phá, nâng cấp lên `Swin Transformer` + `PhoBERT`, sử dụng `AdamW` và scheduler hiện đại (`Warmup+Cosine`). Phiên bản này đã giải quyết được vấn đề overfitting và đạt hiệu suất vượt trội.

**Bài học quan trọng nhất:** Việc lựa chọn các mô hình nền tảng **phù hợp với ngôn ngữ và tác vụ cụ thể (PhoBERT cho tiếng Việt)** mang lại hiệu quả cải thiện lớn nhất, quan trọng hơn cả việc tinh chỉnh các siêu tham số nhỏ lẻ.
