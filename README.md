# SỔ TAY HƯỚNG DẪN CÀI ĐẶT & LỆNH THỰC THI (GUIDE.MD)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, danh mục lệnh chạy và các kịch bản thực thi cho dự án số hóa biểu mẫu khảo sát / đánh giá thực tập (`thesis`).

---

## 1. Yêu Cầu & Thiết Lập Môi Trường

### 1.1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 (khuyến nghị chạy lệnh qua terminal **Git Bash** của VS Code/IDE hoặc PowerShell).
- **Python**: Phiên bản 3.12+ (x64).
- **Tesseract OCR**: Cài đặt tại `C:\Program Files\Tesseract-OCR\tesseract.exe` (khuyến nghị cài thêm gói ngôn ngữ tiếng Việt `vie`).

### 1.2. Khởi tạo môi trường ảo & Cài đặt gói
```bash
# Trong Git Bash tại thư mục gốc c:/dev/thesis:
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

> **Lưu ý về PyTorch (CPU)**: Trong `requirements.txt` đã cấu hình sẵn bản CPU `torch==2.15.0+cpu`. Nếu cần cài lại thủ công:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 1.3. Kiểm tra môi trường
```bash
python src/check_env.py
```
*Đầu ra in `OpenCV: ...` và `OK` là môi trường đã sẵn sàng.*

### 1.4. Lưu ý về bảng mã UTF-8 trên Windows
- Khi chạy trên **PowerShell**, bắt buộc khai báo biến môi trường UTF-8 để không bị lỗi ký tự tiếng Việt:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'
  ```
- Khi chạy trên **Git Bash**:
  ```bash
  export PYTHONIOENCODING=utf-8
  ```

---

## 2. Bảng Tra Cứu Lệnh Nhanh (Cheat Sheet)

| Tác vụ | Lệnh Git Bash | Ghi chú |
|---|---|---|
| **Kiểm tra môi trường** | `python src/check_env.py` | Kiểm tra Tesseract, PyMuPDF, OpenCV |
| **Quét form thật** | `python src/scan_form.py data/samples/Data1.1.pdf` | Xuất ảnh scan và file JSON vào `output/` |
| **Quét + Mô phỏng** | `python src/scan_form.py data/samples/M02_...pdf --simulate` | Tự vẽ dấu ngẫu nhiên để kiểm tra pipeline |
| **Kiểm thử hồi quy** | `python src/regress.py` | Chạy 22 kịch bản song song, so với baseline |
| **Cập nhật baseline** | `python src/regress.py --update-baseline` | Chỉ dùng khi kết quả tốt hơn và đã xác minh |
| **Chấm 1 file thật** | `python src/evaluate.py output/Data1.1_scan.json` | So kết quả quét với `data/truth/` |
| **Tạo nháp đáp án** | `python src/make_truth.py output/Data1.1_scan.json` | Sinh nháp vào `data/truth/` để người sửa tay |
| **Cắt ảnh train (thử)** | `python src/make_dataset.py --n 5 --only sim` | Sinh dữ liệu mô phỏng nhanh |
| **Cắt toàn bộ ảnh train**| `python src/make_dataset.py` | Chạy 40 seed × 2 mẫu + toàn bộ file thật |
| **Cắt lại file thật** | `python src/make_dataset.py --only real` | Chạy sau khi có đáp án thật mới |
| **Train CNN (nhanh)** | `python src/train_marker.py --epochs 3` | Train thử 3 epoch |
| **Train CNN (chuẩn)** | `python src/train_marker.py` | Train 15 epoch, lưu `models/marker_cnn.pt` |

Thêm 2026-11-15:

| Mục đích | Lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Quét chỉ bằng luật** | `python src/scan_form.py data/samples/Data9.1.pdf --no-cnn` | Mặc định: luật + CNN cứu dấu (`CNN_RESCUE` = 0.90) |
| **Đánh giá chéo CNN** | `python src/train_marker.py --fold 0 --out <thư mục tạm>` rồi `--fold 1` | Học mô phỏng + 1/2 số sinh viên, test nửa còn lại. Dùng `--out` để không ghi đè `models/marker_cnn.pt` |
| **Train CNN dùng thật** | `python src/train_marker.py --fold -1` | Học mô phỏng + mọi file thật, lưu `models/marker_cnn.pt` |
| **Quét lại toàn bộ để xem ảnh** | `ls data/samples/*.pdf \| xargs -P 4 -I{} python src/scan_form.py "{}"` | Sau mỗi thay đổi; `regress.py` không cập nhật `output/` |
| **Trang cần xem lại** | xem `needs_review` trong `output/<file>_scan.json` | Ảnh trang đó có băng cam "NEEDS REVIEW" |

---

## 3. Hướng Dẫn Chi Tiết Từng Luồng Thực Thi

### 3.1. Quét biểu mẫu đơn lẻ (`scan_form.py`)
```bash
# Quét bản scan thực tế 6 trang (CamScanner)
python src/scan_form.py data/samples/Data1.1.pdf

# Quét bản scan thực tế 2 trang (CamScanner)
python src/scan_form.py data/samples/Data1.2.pdf

# Quét mẫu chuẩn M02 (5 trang điện tử)
python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf

# Tùy chọn đường dẫn xuất kết quả khác output/
python src/scan_form.py data/samples/Data1.1.pdf --out my_output
```
*Kết quả xuất ra gồm ảnh trực quan hóa `output/<stem>_p<trang>_scan.png` và dữ liệu nhận diện `output/<stem>_scan.json`.*

### 3.2. Chế độ mô phỏng (`--simulate`)
Dùng để kiểm thử sức chịu đựng của thuật toán với các kịch bản khó (dấu to tràn viền, ảnh scan bị nghiêng, ô nhỏ viền mảnh):
```bash
# Vẽ dấu giả cơ bản
python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf --simulate

# Mô phỏng dấu tràn ra ngoài viền ô (k=1.7) và xoay nghiêng 1.5 độ
python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf --simulate --seed 7 --overshoot 0.3 --rotate 1.5

# Mô phỏng trộn lẫn ô nhỏ viền mảnh 1px (như Word in)
python src/scan_form.py data/samples/M02_Intern-evaluation-template.pdf --simulate --seed 3 --mixed
```

### 3.3. Kiểm thử hồi quy tự động (`regress.py`)
Mỗi lần sửa đổi thuật toán hoặc tham số, **bắt buộc chạy script này** để đảm bảo không làm giảm độ chính xác:
```bash
# Chạy toàn bộ (4 file thật + 18 kịch bản mô phỏng song song)
python src/regress.py

# Chỉ chạy các file thật
python src/regress.py --only real

# Chỉ chạy mô phỏng với 2 tiến trình
python src/regress.py --only sim -j 2

# Giữ lại thư mục tạm để xem ảnh khi có lỗi
python src/regress.py --keep
```

### 3.4. Sinh dữ liệu & Huấn luyện bộ phân loại dấu AI (CNN)
Quy trình huấn luyện mạng CNN nhỏ (80k tham số, chạy hoàn toàn trên CPU):

1. **Bước 1: Sinh bộ ảnh cắt** (hình vuông 64×64 pixel quanh từng ô):
   ```bash
   # Cắt lại ảnh cho các tài liệu thật (sau khi có sửa đổi hàm cắt hoặc thêm đáp án)
   python src/make_dataset.py --only real

   # Sinh đủ bộ dữ liệu mô phỏng (40 seed × 2 mẫu + file thật, mất ~10-12 phút)
   python src/make_dataset.py
   ```
2. **Bước 2: Huấn luyện mô hình**:
   ```bash
   # Huấn luyện 15 epoch (mất ~2 phút trên CPU laptop)
   python src/train_marker.py

   # Huấn luyện với tham số tùy chỉnh
   python src/train_marker.py --epochs 20 --lr 0.0005 --batch 64
   ```
   *Mô hình tốt nhất được lưu tại `models/marker_cnn.pt`, báo cáo kết quả ghi tại `models/marker_metrics.json`.*

### 3.5. Quy trình ghi nhãn bán tự động cho file scan mới
Khi có một file scan PDF mới từ sinh viên/trường:
1. Quét form:
   ```bash
   python src/scan_form.py data/samples/TenFileMoi.pdf
   ```
2. Sinh bản nháp đáp án:
   ```bash
   python src/make_truth.py output/TenFileMoi_scan.json
   ```
3. Mở file ảnh `output/TenFileMoi_p*_scan.png` đối chiếu bằng mắt, mở `data/truth/TenFileMoi.json` sửa các chỗ máy đoán sai, sau đó đổi `"verified": false` thành `"verified": true`.
4. Đánh giá lại:
   ```bash
   python src/evaluate.py output/TenFileMoi_scan.json
   ```

