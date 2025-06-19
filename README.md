# ViCaptionTalk 🤖 - Hệ thống AI Chú thích Ảnh Tiếng Việt

**Tác giả: Thanh Vân**
<br>
Một dự án nghiên cứu và phát triển chuyên sâu về việc xây dựng mô hình Deep Learning có khả năng phân tích nội dung hình ảnh và tự động sinh ra chú thích (caption) bằng tiếng Việt một cách tự nhiên và chính xác.

---

## 🏆 Kết quả Nổi bật

Sau một quá trình tối ưu hóa và huấn luyện bài bản, mô hình cuối cùng đã đạt được hiệu suất rất cao trên tập dữ liệu test, một tập dữ liệu mà mô hình chưa từng thấy trước đây.

| Metric | Score | Diễn giải |
| :--- | :--- | :--- |
| 🔷 **BLEU** | **35.93%** | Mức độ trùng khớp về các cụm từ (n-grams), cho thấy sự tương đồng cao về cấu trúc và từ vựng so với chú thích của con người. |
| 🔶 **ROUGE-L** | **50.23%** | Đo lường chuỗi con chung dài nhất, cho thấy mô hình có khả năng tái tạo lại cấu trúc câu và ngữ nghĩa cốt lõi. |
| 💚 **METEOR** | **52.41%** | Xem xét cả sự trùng khớp chính xác, từ đồng nghĩa và stemming, cho thấy sự am hiểu sâu sắc về ngữ nghĩa. |

Những con số này không chỉ vượt qua mô hình baseline ban đầu mà còn khẳng định khả năng khái quát hóa mạnh mẽ của kiến trúc đã được lựa chọn.

---

## 🛠️ Phân tích Kiến trúc Kỹ thuật

Hệ thống được xây dựng trên kiến trúc **Encoder-Decoder**, một kiến trúc tiêu chuẩn và mạnh mẽ cho các bài toán Seq2Seq (chuỗi sang chuỗi). Phiên bản tối ưu đã sử dụng các thành phần SOTA (State-of-the-art) để tối đa hóa hiệu suất.

#### 1. Vision Encoder: `Swin Transformer` (`swin_base_patch4_window7_224.ms_in22k`)

-   **Nhiệm vụ:** Trích xuất đặc trưng hình ảnh. Nó hoạt động như "đôi mắt" của hệ thống.
-   **Lý do lựa chọn:** Không giống như ViT-base (sử dụng trong thử nghiệm đầu), Swin Transformer sử dụng cơ chế "cửa sổ dịch chuyển" (shifted window), cho phép nó nắm bắt hiệu quả các mối quan hệ cả ở cấp độ cục bộ (chi tiết nhỏ) và toàn cục (bối cảnh rộng) của hình ảnh. Điều này giúp mô hình hiểu được các tương tác phức tạp giữa các đối tượng.

#### 2. Language Model (Decoder): `PhoBERT` (`vinai/phobert-base`) & `Transformer Decoder`

-   **Nhiệm vụ:** Sinh ra câu chú thích tiếng Việt. Đây là "bộ não ngôn ngữ" của hệ thống.
-   **Lý do lựa chọn (Đây là cải tiến quan trọng nhất):**
    -   `BERT-multilingual` (sử dụng trong thử nghiệm đầu) phải chia sẻ năng lực của mình cho nhiều ngôn ngữ, dẫn đến sự am hiểu hời hợt về tiếng Việt.
    -   `PhoBERT`, ngược lại, được huấn luyện chuyên sâu trên một kho dữ liệu tiếng Việt khổng lồ (hơn 20GB). Nó có một sự am hiểu sâu sắc về ngữ pháp, từ vựng, và các sắc thái tinh tế của tiếng Việt, giúp tạo ra các câu văn tự nhiên và chính xác hơn nhiều.
-   **Cơ chế hoạt động:** Một bộ `Transformer Decoder` (6 lớp) sẽ nhận đặc trưng hình ảnh từ Swin Transformer làm "bộ nhớ" và sử dụng năng lực ngôn ngữ của PhoBERT để tuần tự sinh ra từng từ một.

---

## 🔬 Hành trình Thử nghiệm & Các Quyết định Tối ưu

Quá trình phát triển không phải là một đường thẳng mà là một vòng lặp của **Thử nghiệm -> Phân tích -> Cải tiến**.

| Phiên bản | Kiến trúc & Tham số | Kết quả (Val BLEU) | Phân tích & Bài học |
| :--- | :--- | :--- | :--- |
| **#1 - The Baseline** | `ViT-base` + `BERT-multi`<br>Adam, StepLR | **~31.3%** | **Khả thi nhưng hạn chế:** Chứng minh hướng đi là đúng, nhưng overfitting sớm và bị giới hạn bởi khả năng ngôn ngữ của BERT đa ngôn ngữ. |
| **#2 - Over-Correction**| `ViT-base` + `BERT-multi`<br>Regularization rất mạnh | **~22.3%** | **Bị kìm hãm:** Việc chống overfitting "quá tay" (dropout & weight decay cao) đã ngăn mô hình học hiệu quả. Bài học: Regularization cần sự cân bằng. |
| **#3 - The Ultimate**| `Swin` + `PhoBERT`<br>AdamW, Warmup+Cosine, Data Augmentation cân bằng | **~37.7%** | **Thành công vượt trội:** Sự kết hợp của các model chuyên biệt và chiến lược training hiện đại đã giải quyết được overfitting và phá vỡ ngưỡng hiệu suất cũ. |

Quá trình này cho thấy tầm quan trọng của việc lựa chọn **mô hình nền tảng phù hợp (domain-specific)** và áp dụng các **chiến lược huấn luyện hiện đại** cho các bài toán phức tạp.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu

*   Python 3.9+
*   PyTorch, TorchVision
*   Các thư viện trong `requirements.txt`
*   GPU có VRAM >= 8GB (khuyến nghị)

### Các bước

1.  **Tải Dự án:**
    Toàn bộ dự án, bao gồm mã nguồn, dữ liệu và model đã huấn luyện, được lưu trữ trên Google Drive để tiện cho việc tải về.
    *   ➡️ **[Tải toàn bộ dự án tại đây (Link Google Drive)](https://drive.google.com/drive/folders/1nv2xMZnNctl5MaZs-XHenAfEHmQughIx?usp=sharing)**
    *   **Hướng dẫn:** Tại trang Drive, nhấn vào tên thư mục `ViCaptionTalk` và chọn **"Tải xuống"**. Sau đó, giải nén file `.zip` vừa tải về.

2.  **Cài đặt Môi trường:**
    Mở terminal trong thư mục dự án vừa giải nén và chạy các lệnh sau:
    ```bash
    # (Khuyến nghị) Tạo và kích hoạt môi trường ảo
    python -m venv venv
    source venv/bin/activate  # Trên Linux/macOS. Dùng venv\Scripts\activate trên Windows

    # Cài đặt các thư viện
    pip install -r requirements.txt
    ```

3.  **Chạy Web Demo:**
    Đây là cách tốt nhất để trải nghiệm mô hình. Lệnh này sẽ khởi động một web server và tự động mở trình duyệt.
    ```bash
