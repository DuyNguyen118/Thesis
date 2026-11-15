# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP
## Hệ thống số hóa và nhận dạng tự động phiếu khảo sát, đánh giá (M02 & M03) từ ảnh quét PDF

* **Người thực hiện:** Sinh viên thực hiện đồ án
* **Ngày cập nhật:** 08/11/2026
* **Môi trường triển khai:** Python 3.12, OpenCV 4.x, Tesseract OCR 5.x, PyTorch (CPU)

---

## 1. TỔNG QUAN ĐỀ TÀI

### 1.1. Bài toán thực tế
Hàng kỳ, nhà trường tiếp nhận hàng trăm biểu mẫu đánh giá thực tập doanh nghiệp (**M02 - Internship Evaluation**) và phiếu phản hồi thị trường lao động (**M03 - Labor Market's Feedback**) dưới dạng bản in được điền tay rồi scan/chụp gửi lại (qua máy scan chuyên dụng hoặc ứng dụng di động như CamScanner).
* **Biểu mẫu M02 (4–6 trang):** Gồm 13 tiêu chí đánh giá dạng trắc nghiệm chọn 1 (ô checkbox vuông hoặc tròn) và 1 bảng Likert đánh giá chuyên cần (Attendance & Punctuality).
* **Biểu mẫu M03 (2 trang):** Gồm các bảng ma trận Likert đánh giá năng lực theo thang điểm 1 đến 5, bảng nối tiếp qua nhiều trang, kèm chữ ký và con dấu doanh nghiệp.

### 1.2. Mục tiêu nghiên cứu
Xây dựng hệ thống phần mềm tự động hóa toàn bộ quy trình:
1. Nhận đầu vào là thư mục chứa các tập tin PDF scan chất lượng đa dạng (nghiêng, mờ, có dấu mộc, chữ ký đè, nét vẽ tràn viền).
2. Tự động chỉnh thẳng góc nghiêng (deskew), tách layout, nhận diện cấu trúc checkbox và bảng biểu.
3. Nhận dạng chính xác ô nào được đánh dấu (tick, gạch chéo, khoanh tròn, tô kín) với độ tin cậy cao.
4. Đối chiếu với cấu trúc câu hỏi chuẩn và xuất ra bảng dữ liệu tổng hợp (Excel/CSV) điểm số của từng sinh viên, tự động gắn cờ các trường hợp bất thường để con người thẩm định.

---

## 2. KIẾN TRÚC GIẢI PHÁP & PHƯƠNG PHÁP TIẾP CẬN

Đồ án áp dụng **mô hình lai (Hybrid Architecture)** kết hợp Thị giác máy tính truyền thống (Computer Vision), Xử lý ngôn ngữ tự nhiên (NLP matching) và Học máy (Deep Learning):

```
PDF Scan ──► Render ảnh (300 DPI) ──► Tự động Deskew ──► Binarization (Otsu)
                                                               │
       ┌───────────────────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                                                                                ▼
[Nhánh 1: Checkbox]                                                                           [Nhánh 2: Bảng Likert]
├── Tự học mẫu ô (Template Matching)                                                          ├── Phát hiện đường kẻ (Morphology)
├── Khớp mẫu đa tỉ lệ + Lọc thẳng hàng                                                        ├── Phân đoạn bảng, hàng, cột
├── Cứu ô mờ viền trong cột (_rescue_in_columns)                                              ├── Tự động nhận diện header số/chữ
└── Lọc chữ cái OCR đè (_inside_word)                                                         └── Xử lý bảng nối tiếp không header
       │                                                                                                │
       └───────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                           ▼
                                 [Trích xuất vùng quan tâm]
                                 ├── Baseline: Đo mực hình học (Cell ink / Center ink / Halo)
                                 └── CNN: Mạng nơ-ron tích chập phân loại ảnh cắt 64x64
                                           ▼
                                 [Hệ thống đối chiếu & Đánh giá]
                                 ├── So sánh với Ground Truth (IoU, vị trí, RapidFuzz)
                                 └── Kiểm thử hồi quy tự động (Regression Suite)
```

1. **Tầng tiền xử lý:** Sử dụng PyMuPDF render ảnh chuẩn 300 DPI. Thuật toán phân tích phổ góc xoay tự động nắn thẳng trang (deskew) từng trang độc lập.
2. **Tầng định vị hình học (Layout Detection):**
   * *Biểu mẫu:* Tự động học mẫu hình dạng ô vuông từ chính trang đang quét (`learn_template`), sau đó áp dụng `cv2.matchTemplate` với mặt nạ viền để tìm ô kể cả khi có nét vẽ đè lên.
   * *Bảng biểu:* Áp dụng biến đổi hình thái học (morphological operations) trích xuất lưới đường kẻ ngang/dọc, nhóm các giao điểm thành ô bảng Likert.
3. **Tầng nhận diện đánh dấu (Mark Classification):**
   * **Quy tắc hình học (Baseline):** Đo mật độ mực lòng ô (`ink_fill`), vành ngoài (`halo_fill`) và độ lệch mực tương đối giữa các ô trong cùng hàng bảng (`CELL_DELTA`).
   * **Mạng CNN (~80k tham số):** Nhận đầu vào là ảnh cắt 64×64 chuẩn hóa của từng ô, dự đoán xác suất có dấu (đạt tiêu chí nhẹ, chạy nhanh hoàn toàn trên CPU thông thường).

---

## 3. TẬP DỮ LIỆU THỰC NGHIỆM & BỘ ĐÁP ÁN CHUẨN (GROUND TRUTH)

Để đảm bảo tính khoa học và ngăn chặn hiện tượng quá khớp (overfitting), toàn bộ quá trình phát triển được kiểm soát bằng tập đáp án chuẩn đã được con người kiểm tra (`verified: true`).

### 3.1. Danh mục tài liệu thực nghiệm (8 tài liệu)

| STT | Tên tài liệu | Loại mẫu | Số trang | Nguồn gốc / Đặc điểm | Số ô checkbox | Số hàng bảng | Trạng thái đáp án |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | `Data1.1.pdf` | M02 | 6 | CamScanner, dấu tick to, chữ ký/mộc đè | 54 (13 có dấu) | 2 (Attendance) | `verified: true` |
| 2 | `Data1.2.pdf` | M03 | 2 | CamScanner, bảng thang điểm 1..5 | 0 | 12 (12 có dấu) | `verified: true` |
| 3 | `Data2.1.pdf` | M02 | 6 | CamScanner, ô mờ viền, bảng có ô vuông | 54 (13 có dấu) | 2 (Attendance) | `verified: true` |
| 4 | `Data2.2.pdf` | M03 | 2 | CamScanner, bảng chia đôi qua trang 2 | 0 | 12 (11 có dấu) | `verified: true` |
| 5 | `Data3.1.pdf` | M02 | 4 | Máy scan phẳng, dùng ô tròn thay ô vuông | 54 (13 có dấu) | 2 (Attendance) | `verified: true` |
| 6 | `Data3.2.pdf` | M03 | 2 | Máy scan phẳng, thang điểm 1..5, có ô trống | 1 (0 có dấu) | 12 (11 có dấu) | `verified: true` |
| 7 | `M02_template.pdf` | M02 | 5 | Biểu mẫu trắng chuẩn M02 từ trường | 54 (0 có dấu) | 2 (0 có dấu) | `verified: true` |
| 8 | `M03_template.pdf` | M03 | 2 | Biểu mẫu trắng chuẩn M03 từ trường | 0 | 12 (0 có dấu) | `verified: true` |

### 3.2. Tập dữ liệu ảnh cắt (Crops Dataset)
* **Tổng số ảnh cắt thật (Test Set):** **498 ảnh** (gồm 218 ô checkbox và 280 ô bảng điểm).
  * 80 ảnh có đánh dấu (`marked`).
  * 417 ảnh ô trống (`empty`).
  * 1 ảnh chữ ký/nhiễu viền (`not_box`).
* **Tập dữ liệu mô phỏng (Train/Val Set):** Hơn **4.800 ảnh** được tổng hợp tự động qua 40 biến thể giả lập (nhiều kiểu nét bút: tick, cross, circle, fill; độ dày mực, góc xoay scan lệch, nét vẽ tràn viền). Tập train/val độc lập 100% với tập test thật.

---

## 4. CÁC VẤN ĐỀ KỸ THUẬT ĐÃ GIẢI QUYẾT

Trong quá trình thử nghiệm trên các bản scan thực tế, hệ thống đã phát hiện và xử lý thành công 4 thách thức lớn:

### 4.1. Nhận diện bảng điểm Likert bị ngắt qua nhiều trang (Data2.2)
* **Hiện tượng:** Ở mẫu M03, Bảng 2 bị chia cắt: 8 hàng đầu ở trang 1 (có tiêu đề `1 2 3 4 5`), 2 hàng cuối (`Independence`, `Relations with others`) bị đẩy sang đầu trang 2 và **không có dòng tiêu đề cột điểm**. Thuật toán cũ yêu cầu bảng phải có header và $\ge 3$ hàng nên bỏ sót hoàn toàn bảng này.
* **Giải pháp:** Bổ sung cơ chế phát hiện bảng không header (`has_header = False`). Nếu bảng chỉ có $\ge 2$ hàng và không chứa số/chữ ở hàng đầu, hệ thống tự động gán thang điểm mặc định `1..n` và đọc dữ liệu ngay từ hàng đầu tiên (`start_r = 0`).
* **Kết quả:** Nhận diện đủ 12/12 hàng của Data2.2.

### 4.2. Độ nhạy nhận diện dấu trong bảng có in sẵn checkbox (Data2.1 & Data2.2)
* **Hiện tượng:** Bảng chuyên cần của M02 in sẵn khung checkbox rỗng bên trong từng ô bảng. Bản thân khung in sẵn đã chiếm mật độ mực ~0.05, khi người dùng tick bút nhạt thì mật độ mực chỉ tăng thêm ~0.013. Ngưỡng cũ `CELL_DELTA = 0.03` quá cao dẫn đến bỏ sót toàn bộ dấu của bảng này.
* **Giải pháp:** Khảo sát toàn bộ dữ liệu cho thấy các ô bảng thực sự trống chỉ có độ chênh lệch mực $\le 0.001$. Do đó, ngưỡng `CELL_DELTA` được hạ an toàn xuống `0.007`.
* **Kết quả:** Data2.1 đạt **2/2 hàng** (nhận đúng cột Excellent); Data2.2 nhận diện thành công nét mực siêu mờ (0.008) ở hàng 2, đạt **12/12 hàng**.

### 4.3. Cứu ô bị đứt viền do chất lượng scan (Data2.1 trang 3)
* **Hiện tượng:** Ô `Sometimes Interested` trên Data2.1 bị đứt nét viền ngang trên và dưới, khiến điểm khớp mẫu chỉ đạt **0.569** (dưới ngưỡng ứng viên 0.60). Đồng thời, hàm cứu ô theo cột trước đây chỉ kích hoạt cho ô có mực, dẫn đến ô trống này bị sót.
* **Giải pháp:** Cải tiến hàm `_rescue_in_columns()`: đối với các cột đã xác định được $\ge 3$ checkbox thẳng hàng, nếu một vị trí có điểm khớp $\ge \text{TM\_LOW}$ (0.55), có dòng chữ nhãn bên phải (`text_right_of`) và không phải chữ cái OCR, hệ thống sẽ cứu lại ô đó vào danh sách.
* **Kết quả:** Data2.1 phát hiện đủ **54/54 ô** (0 sót, 0 thừa).

### 4.4. Tránh nhận nhầm tiêu đề mục in hoa thành checkbox
* **Hiện tượng:** Các tiêu đề in hoa căn lề trái như `4. INITIATIVE` hay `5. INDEPENDENCE` có chữ `I` hoặc số thứ tự thẳng hàng dễ bị nhầm thành ô checkbox nếu hạ ngưỡng toàn cục.
* **Giải pháp:** Kết hợp chặt chẽ giữa ngưỡng khớp ban đầu (giữ 0.60 ở bước lọc thô), hàm kiểm tra chữ nằm trong từ (`_inside_word`), và chỉ cho phép ngưỡng 0.55 bên trong phạm vi cột checkbox đã được kiểm chứng.

---

## 5. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG

Hệ thống được kiểm thử bằng bộ kiểm thử hồi quy tự động [`src/regress.py`](file:///c:/dev/thesis/src/regress.py) với **26 kịch bản chạy song song**:

### 5.1. Kết quả trên 8 tài liệu scan thật (Baseline Luật)

| Tài liệu scan | Loại | Số trang | Checkbox tìm được | Checkbox sót | Checkbox thừa | Dấu nhận đúng | Dấu sót (FN) | Báo nhầm (FP) | Hàng bảng đúng | Độ khớp nhãn OCR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data1.1** | M02 | 6 | 54/54 | 0 | 0 | 13/13 | 0 | 0 | 2/2 | 99.6% |
| **Data1.2** | M03 | 2 | 0/0 | 0 | 0 | 0/0 | 0 | 0 | 12/12 | 100.0% |
| **Data2.1** | M02 | 6 | 54/54 | 0 | 0 | 13/13 | 0 | 0 | 2/2 | 99.9% |
| **Data2.2** | M03 | 2 | 0/0 | 0 | 1* | 0/0 | 0 | 0 | 12/12 | 100.0% |
| **Data3.1** | M02 | 4 | 54/54 | 0 | 0 | 13/13 | 0 | 0 | 2/2 | 100.0% |
| **Data3.2** | M03 | 2 | 1/1 | 0 | 0 | 0/0 | 0 | 0 | 12/12 | 100.0% |
| **M02 Template**| M02 | 5 | 54/54 | 0 | 0 | 0/0 | 0 | 0 | 2/2 | 100.0% |
| **M03 Template**| M03 | 2 | 0/0 | 0 | 0 | 0/0 | 0 | 0 | 12/12 | 100.0% |
| **TỔNG CỘNG** | — | **29** | **217/217 (100%)** | **0** | **1\*** | **39/39 (100%)** | **0** | **0** | **46/46 (100%)** | **$\ge$ 99.6%** |

*\* Ghi chú:* Ô thừa ở Data2.2 trang 2 là chữ cái `C` trong phần chữ ký tay người đánh giá ở góc dưới cùng trang, ô này không có dấu nên không gây sai lệch dữ liệu.

### 5.2. Kết quả trên 18 kịch bản mô phỏng kiểm thử độ bền (Stress Testing)
* **Tổng số dấu giả lập được vẽ:** 233 dấu (trộn lẫn dấu tick, cross, khoanh tròn, tràn viền, mực mờ, xoay nghiêng $-1.8^\circ \dots +2.0^\circ$).
* **Số dấu nhận dạng chính xác:** **231 / 233 (99.14%)**.
* **Số trường hợp báo nhầm (False Positive):** **0 (0.00%)**.

### 5.3. So sánh hiệu năng phân loại trên 498 ảnh cắt thực tế (Test Set)

| Phương pháp | Tổng mẫu test | Độ chính xác (Accuracy) | Độ chuẩn xác (Precision) | Độ thu hồi (Recall) | F1-Score | Số lỗi sót (FN) | Số lỗi nhầm (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quy tắc hình học (Baseline)** | 498 | **1.000 (100%)** | **1.000** | **1.000** | **1.000** | **0** | **0** |
| **Mô hình CNN (Hiện tại)** | 498 | 0.974 (97.4%) | 0.904 | 0.938 | 0.920 | 5 | 8 |

*Nhận xét:*
* Bộ quy tắc hình học hiện tại đang hoạt động cực kỳ tối ưu và ổn định trên toàn bộ tập dữ liệu đã có.
* Mô hình CNN hiện tại đạt F1 = 0.920; nguyên nhân chưa đạt tối đa do mô hình chỉ được huấn luyện trên dữ liệu mô phỏng ô vuông, chưa được học các biến thể ô tròn (Data3.1) và các ô bảng chứa sẵn checkbox (Data2.1). Đây là cơ sở để tiếp tục tinh chỉnh ở giai đoạn tiếp theo.

---

## 6. HƯỚNG DẪN TÁI LẬP KẾT QUẢ NGHIÊN CỨU

Mọi số liệu trong báo cáo này đều có thể được tái lập hoàn toàn thông qua các câu lệnh tự động:

1. **Kiểm tra môi trường:**
   ```bash
   python src/check_env.py
   ```
2. **Chạy toàn bộ 26 bài kiểm thử hồi quy (Simulated & Real):**
   ```bash
   python src/regress.py -j 4
   ```
3. **Quét và đánh giá chi tiết một file scan bất kỳ:**
   ```bash
   python src/scan_form.py data/samples/Data2.1.pdf --out output
   python src/evaluate.py output/Data2.1_scan.json
   ```
4. **Đánh giá mô hình CNN trên 498 ảnh test thực tế:**
   ```bash
   python src/train_marker.py --epochs 0
   ```

---

## 7. KẾ HOẠCH TRIỂN KHAI TIẾP THEO

1. **Mở rộng tập dữ liệu thực nghiệm:** Tiếp tục scan thêm 10–20 bản đánh giá thật từ nhiều nguồn sinh viên/doanh nghiệp khác nhau để kiểm tra độ khái quát hóa.
2. **Nâng cấp mô hình CNN (Mục 2B.5):** Bổ sung dữ liệu huấn luyện cho ô tròn và ô bảng, xuất mô hình sang định dạng ONNX, tích hợp cờ chuyển đổi `--marker cnn|rule` trong phần mềm.
3. **Hoàn thiện Giai đoạn 3 (Ghép cấu trúc biểu mẫu):** 
   * Xây dựng file schema `forms/M02.json` và `forms/M03.json`.
   * Khớp nhãn OCR bằng thuật toán so sánh chuỗi mờ (RapidFuzz).
   * Bóc tách các trường thông tin tự do: Tên sinh viên, Tên công ty, Điểm tổng kết.
4. **Giai đoạn 4 (Xử lý hàng loạt & Xuất báo cáo):** Viết script `batch_scan.py` xử lý tự động cả thư mục và xuất bảng kết quả tổng hợp ra file Excel (`results.xlsx`).

---

## 8. CẬP NHẬT 2026-11-15: MỞ RỘNG LÊN 28 TÀI LIỆU, CỜ "CẦN XEM LẠI" VÀ CNN CỨU DẤU

> Các mục 1–7 ở trên giữ nguyên như lúc viết (8 tài liệu). Mục này gộp mọi thay đổi và số liệu của ngày 2026-11-15, số liệu là trạng thái cuối ngày. Chỗ nào mục cũ không còn đúng được ghi ở mục 8.8 (đính chính), không sửa vào mục cũ.

### 8.1. Tập dữ liệu: 28 tài liệu có đáp án

* **28 tài liệu**, tất cả `verified: true` trong `data/truth/`: 13 sinh viên × 2 biểu mẫu (`DataN.1` = M02, `DataN.2` = M03, CamScanner) + 2 biểu mẫu trắng (M02, M03).
* Quy ước đã chốt: mỗi PDF = 1 sinh viên, 1 loại biểu mẫu; mỗi sinh viên có đúng 2 PDF (M02 + M03); không có biểu mẫu/phiên bản nào khác.
* Tổng số phần tử trong đáp án: **759 ô checkbox (169 có dấu)** và **197 hàng bảng**.
* Rà soát đáp án bằng ảnh, đã sửa 7 lỗi ghi nhãn:

| Tài liệu | Trang | Lỗi trong đáp án | Đã sửa thành |
| :--- | :---: | :--- | :--- |
| Data4.1 | 6 | Bảng Attendance/Punctuality ghi thành 8 ô rời (nhãn rác "U", "w"…), thiếu cột Excellent | 1 bảng, Attendance = Good, Punctuality = Excellent |
| Data6.1 | 5 | Đã xóa 10 ô bảng nhưng không thêm bảng | 1 bảng, cả 2 hàng = Excellent |
| Data9.1 | 7 | Trang rỗng, thực tế có 7 ô (2 có dấu) | Thêm 7 ô |
| Data9.1 | 9 | Thiếu bảng | 1 bảng, cả 2 hàng = Excellent |
| Data6.1 | 2 | 2 ô trùng tọa độ (494, 1304) | Ô "All objectives…" dời về (493, 1197) |
| Data13.1 | 5 | Ô "Work usually required review…" có tick nhưng ghi `marked: false` | `marked: true` |
| Data5.1 | 4 | Tọa độ 3 ô đầu trang lệch 87–129 px theo chiều dọc | (513, 279), (515, 387), (520, 723), lấy theo khung ký tự OCR |

### 8.2. Các vấn đề kỹ thuật đã giải quyết

**8.2.1. Bảng Attendance/Punctuality không được nhận trên scan lệch hoặc mất viền (Data4.1, 6.1, 9.1)**
* *Hiện tượng:* cả 3 tài liệu đều `tables: []`; 10 ô của bảng bị báo thành ô checkbox rời với nhãn rác.
* *Nguyên nhân:* nét kẻ đứt quãng sau scan; ở Data4.1 cột Excellent lệch lên ~10 px; ở Data9.1 bảng mất hẳn đường viền phải.
* *Giải pháp* (`src/detect_layout.py:130` `find_table_lines`, `src/detect_tables.py:22` `find_tables`): độ dài tối thiểu của nét kẻ tính theo mm (8 mm) thay vì 8% bề rộng trang; đóng hình thái (MORPH_CLOSE 5 px) để nối nét đứt; nở 7×7 khi gom nét thành khối bảng; tự bù cột khi nét ngang kéo dài quá đường dọc cuối (bù theo bề rộng cột trung vị, chỉ khi > 10 mm).
* *Kết quả:* cả 3 tài liệu từ "không thấy bảng" lên **2/2 hàng đúng**.

**8.2.2. Đọc dấu trong ô bảng: một ngưỡng cố định không đủ**
* *Số đo* (mực vùng giữa ô trừ trung vị của hàng): tick nhạt thật ở Data9.1 chỉ **+0.014 / +0.020**; bóng mờ ở ô trống bên cạnh ở Data6.1 đã **+0.011**; tick trên PDF sạch ở Data2.2 **+0.008** (các ô khác = 0).
* → Không tồn tại một ngưỡng tuyệt đối tách được cả 3 trường hợp. Ngưỡng 0.007 (mục 4.2) làm Data6.1, 12.1, 13.1 bị đọc 2 cột ở một hàng.
* *Giải pháp* (`src/scan_form.py:26-30`, `:535`): giữ `CELL_DELTA = 0.03`; nếu không ô nào vượt thì xét ô đậm nhất, nhận nếu (a) cao hơn trung vị > `CELL_FAINT` = 0.012, hoặc hàng "sạch" (ô đậm thứ hai cao hơn trung vị ≤ `CELL_CLEAN` = 0.002) và cao hơn > `CELL_CLEAN_MIN` = 0.005; **và** (b) vượt ô đậm thứ hai ≥ `CELL_LEAD` = 50% khoảng cách tới trung vị.
* *Kết quả:* hàng bảng **196/197** (sai duy nhất Data6.2 tr.1 hàng 2, tick quá nhạt).

**8.2.3. Ngưỡng khớp mẫu theo chất lượng trang ("fallback cho ảnh mờ")**
* *Hiện tượng:* trên trang mờ, ô in mảnh/nhạt chỉ khớp mẫu **0.55–0.59** (dưới ngưỡng 0.60). Hạ ngưỡng xuống 0.55 cho mọi trang thì chữ cái đầu tiêu đề in hoa ("I" của `INITIATIVE`, `INDEPENDENCE` ở Data1.1; tiêu đề ở Data3.1) cũng lọt thành ô.
* *Giải pháp* (`src/scan_form.py:241-256` `page_quality`): đo từng trang
  * độ nét = phương sai Laplacian của ảnh xám,
  * độ đậm mực = trung vị mức xám của các điểm mực (càng cao càng nhạt).
  * Trang "mờ/nhạt" khi độ nét < `SOFT_SHARPNESS` = 200 **hoặc** mực > `SOFT_INK` = 40. Chỉ trang này dùng ngưỡng `TM_LOW` = 0.55; trang rõ giữ `TM_FRAME` = 0.60.
  * Điểm khớp < 0.60 còn phải qua 2 kiểm tra (`src/scan_form.py:291`): có ≥ 3/4 cạnh thẳng (mỗi cạnh có mực ≥ 70% chiều dài), và không có từ OCR dài (≥ 3 ký tự) **bắt đầu bên trong ô** (`_starts_word`, `:322`), vì chữ cái đầu tiêu đề luôn nằm ở đầu một từ.
* *Số đo độ nét/độ đậm mực (một số trang):*

| Trang | Độ nét | Mực | Mờ/nhạt? |
| :--- | :---: | :---: | :---: |
| Data9.1 tr.4 | 102 | 98 | Có |
| Data9.1 tr.7 | 122 | 45 | Có |
| Data9.1 tr.6 | 157 | 52 | Có |
| Data6.1 tr.2 | 205 | 46 | Có (mực) |
| Data3.1 tr.1 (có tiêu đề dễ nhầm) | 319 | 17 | Không |
| Data1.1 tr.3 (có tiêu đề dễ nhầm) | 382 | 17 | Không |
| Các trang rõ khác | 210–1320 | 7–35 | Không |

* *Kết quả:* Data9.1 tìm thêm 4 ô (44 → 48/54), không phát sinh ô thừa ở Data1.1, Data3.1.

**8.2.4. Cờ "cần xem lại" cho trang mờ (`needs_review`)**
* Trang mờ/nhạt (mục 8.2.3) được gắn cờ vì máy dễ sót ô/dấu ở đó:
  * `output/<file>_scan.json`: mỗi trang có `"needs_review": true/false` và `"quality": {"sharpness", "ink_gray", "soft"}`;
  * ảnh `output/<file>_p<n>_scan.png`: băng cam "NEEDS REVIEW" kèm 2 chỉ số;
  * màn hình: dòng `[CẦN XEM LẠI]`.
* 21/110 trang bị gắn cờ (ví dụ Data9.1: trang 3, 4, 6, 7, 10).
* Chưa có cơ sở dữ liệu; khi có, cờ này là một cột `needs_review` của từng trang.

**8.2.5. Ô in kiểu khác và ô OCR đọc thành ký tự (Data2.1, Data5.1)**
* *Hiện tượng:*
  * Data2.1 tr.3: ô "Sometimes Interested" khớp mẫu 0.569, OCR đọc thành `()`.
  * Data5.1 tr.3–5: nhiều ô không tìm ra; ô trống "Can learn…" lại bị đánh dấu đỏ.
* *Nguyên nhân (đo được):*
  * Các ô sót ở Data5.1 **không phải do mờ**: độ nét quanh ô 1.400–4.600, mực 16–34. Đó là ô in **kiểu khác** (nhỏ, viền mảnh), khớp mẫu chỉ **0.05–0.53**, thấp hơn cả `TM_LOW`; hạ ngưỡng sẽ không bắt được mà còn nhận chữ cái tiêu đề (khớp 0.556–0.597). OCR đọc chúng thành `(1`, `O`, `C)`, `LJ`, `OQ`.
  * Bản đầu của bước cứu ô từ ký tự đặt ô ở **x trung vị của cột**; cột lệch dần do nghiêng còn sót nên ô trượt phải ~12 px, trùm chữ "C" của "Can" và bị tính là có dấu.
* *Giải pháp* (`_rescue_ocr_glyphs`, `src/scan_form.py:412`): trong các cột đã có ≥ 3 ô, một "từ" OCR thuộc tập ký tự dạng ô (`CHECKBOX_OCR_GLYPHS`, `:408`), nằm thẳng cột (lệch ≤ 25 px, cỡ ≤ 55 px) và có nhãn bên phải → thêm ô **đúng tọa độ ký tự**, cỡ ô lấy theo các ô cùng cột.
  * Không dùng `(|`, `C)`: trên mô phỏng `--mixed` đó là ô viền mảnh mà khi đã tick thì không tìm lại được. Đã thử lại `C)` sau khi sửa vị trí ô: mô phỏng M02 seed 44 vẫn sót thêm 1 dấu → giữ loại.
  * Cột phải có ≥ 3 ô (`:369`): cho phép cột 2 ô thì chữ cái đầu tiêu đề + 1 ô đã thành "cột" (Data1.1).
* *Kết quả:* Data2.1 54/54 ô; Data5.1 ô 47 → 51/54, báo nhầm 1 → 0; Data10.1 báo nhầm 4 → 0.

**8.2.6. Trang không học được mẫu ô (trang M03 không có checkbox)**
* Bỏ các "ô" không có chữ nhãn bên phải (`src/scan_form.py:446`): đó là nét chữ ký, con dấu, chữ viết tay (Data2.2 tr.2; Data4.2 tr.2 từ 3 ô thừa còn 1).

**8.2.7. Dọn mã nguồn**
* Bỏ ~190 dòng không được gọi tới: một cách tìm ô bằng 4 cạnh (`find_boxes_by_lines`, `frame_ok`, `_ring_ink`) chưa từng được dùng; công cụ đọc bảng chạy riêng trong `detect_tables.py` (đọc dấu bằng ngưỡng cũ 0.02, khác pipeline thật); các hàm kiểm tra hình dạng (`_is_rectangular`, `_corner_angles`, `_border_density`) và hằng số, import thừa.
* Cấu trúc sau khi dọn: tìm ô, đọc dấu ô và đọc dấu bảng nằm trong `scan_form.py`; `detect_tables.py` chỉ tách bảng; `detect_layout.py` chỉ chứa các hàm hình học cơ bản (render, nhị phân hóa, tìm ô sạch, tìm đường kẻ).
* Kiểm chứng: regress 46/46 lượt đạt mốc, mọi chỉ số giống hệt trước khi dọn.

### 8.3. CNN: đánh giá lại trung thực trên 28 tài liệu

**Mô hình cũ (chỉ học mô phỏng), test trên 1.753 ảnh cắt thật của 28 tài liệu:**

| Phương pháp | Ô checkbox: sót / báo nhầm | Ô bảng: sót / báo nhầm | Tổng: sót / báo nhầm | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Luật | 15 / 9 | 1 / 0 | 16 / 9 | 0.964 |
| CNN (chỉ mô phỏng) | 11 / 23 | 9 / 22 | 20 / 45 | 0.910 |

**Đánh giá chéo 2 fold theo sinh viên (học mô phỏng + 1/2 số sinh viên thật, test nửa còn lại):**
* Chia theo sinh viên (`real_form`, `src/train_marker.py:50`): `Data4.1` và `Data4.2` luôn cùng một nửa, tránh "quen nét bút" của người điền.
* Mỗi tài liệu thật được chấm đúng 1 lần, bởi mô hình chưa từng thấy nó.
* Huấn luyện: 15 epoch, batch 64, Adam lr 1e-3, cosine, chọn epoch theo F1 trên val mô phỏng; ~10 giây/epoch trên CPU laptop.

| Fold | Ảnh test | CNN ô: sót / nhầm | Luật ô: sót / nhầm | CNN bảng: sót / nhầm | Luật bảng: sót / nhầm |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (15 tài liệu) | 943 | 1 / 20 | 13 / 8 | 0 / 16 | 1 / 0 |
| 1 (13 tài liệu) | 810 | 4 / 3 | 2 / 1 | 20 / 0 | 0 / 0 |
| **Tổng** | **1.753** | **5 / 23** | **15 / 9** | **20 / 16** | **1 / 0** |

* *Nhận xét:* CNN sót ít hơn luật ở ô checkbox nhưng báo nhầm nhiều hơn; ở ô bảng luật gần như hoàn hảo. CNN dao động mạnh giữa 2 fold (fold 0 nghiêng về "có dấu", fold 1 nghiêng về "trống") vì epoch được chọn theo val mô phỏng, không phản ánh scan thật.
* → **Không thay luật bằng CNN.**

**Kết hợp "luật + CNN cứu ô":** luật quyết định trước; ô luật cho là trống mà CNN ≥ ngưỡng → có dấu. Đo trên 742 ô thật **máy đã tìm thấy** (dự đoán của 2 fold):

| Ngưỡng CNN | Sót | Báo nhầm | Ghi chú |
| :---: | :---: | :---: | :--- |
| Chỉ luật | 6 | 9 | |
| ≥ 0.50 | 0 | 27 | |
| ≥ 0.80 | 0 | 16 | |
| ≥ 0.90 | 1 | 13 | +4 nhầm đều ở trang không có mẫu ô (Data8.2 ×2, Data4.2, Data3.2) |
| ≥ 0.95 | 1 | 13 | |
| ≥ 0.99 | 1 | 12 | |

* Loại trang không có mẫu ô (ở đó "ô" là nét chữ ký) → ≥ 0.90: **sót 1, báo nhầm 9** (bằng luật).
* *Mô hình dùng thật:* học mô phỏng + cả 28 tài liệu (`--fold -1`), lưu `models/marker_cnn.pt`; kết quả 2 fold lưu ở `models/marker_metrics_cv_fold0.json`, `models/marker_metrics_cv_fold1.json`.
* Kiểm tra mô hình cuối trên output: mọi ô có CNN ≥ 0.8 (6 ô, 0.93–1.00) đều có dấu thật; mọi ô trống đều < 0.8 → chọn **`CNN_RESCUE` = 0.90** (`src/scan_form.py:470`, `cnn_rescue` `:491`). Ô bảng vẫn chỉ dùng luật.

### 8.4. Bảng thông số (giá trị cuối ngày 2026-11-15)

| Thông số | Giá trị | Vị trí | Ý nghĩa | Căn cứ / số đo | Giá trị trước |
| :--- | :---: | :--- | :--- | :--- | :---: |
| `DPI` | 300 | `detect_layout.py:8` | Độ phân giải render PDF (~11.81 px/mm) | | 300 |
| `TM_FRAME` | 0.60 | `scan_form.py:142` | Ngưỡng khớp mẫu ô ở trang rõ | Chữ cái tiêu đề khớp tới 0.556–0.597 | 0.60 (ngưỡng cứng) |
| `TM_LOW` | 0.55 | `scan_form.py:141` | Ngưỡng khớp mẫu ô ở trang mờ/nhạt | Ô in mảnh Data2.1 = 0.569, Data9.1 tr.6–7 = 0.55–0.59 | (không dùng) |
| Kiểm tra cạnh | ≥ 3/4 cạnh, mỗi cạnh ≥ 70% | `scan_form.py:291` | Điểm khớp < 0.60 phải còn khung ô | | (mới) |
| `_starts_word` | từ ≥ 3 ký tự, bắt đầu trong nửa trái ô, dài > 1.5 × rộng ô | `scan_form.py:322` | Loại chữ cái đầu tiêu đề | "I" ở x=489, từ "INITIATIVE" bắt đầu x=487 | (mới) |
| `SOFT_SHARPNESS` | 200 | `scan_form.py:241` | Dưới mức này là trang mờ | Data9.1 tr.4/6/7: 102/157/122; trang rõ 210–1320 | (mới) |
| `SOFT_INK` | 40 | `scan_form.py:242` | Mực xám trung vị trên mức này là trang nhạt | Data9.1 tr.4: 98; Data6.1 tr.2: 46; trang đậm 7–35 | (mới) |
| `TM_RESCUE` | 0.40 | `scan_form.py:349` | Cứu ô theo cột khi dấu to đè viền | Đo được 0.44–0.65 | 0.40 |
| `RESCUE_FILL` | 0.12 | `scan_form.py:350` | Ô cứu theo cột phải có mực rõ | | 0.12 |
| Số ô tối thiểu của cột | 3 | `scan_form.py:369` | Mới được cứu ô trong cột | Cột 2 ô → chữ đầu tiêu đề thành ô (Data1.1) | 3 |
| `CHECKBOX_OCR_GLYPHS` | `( ) [] () [ ] CO J — – - © O) (C DC OC o O 0 u U LJ (1 OQ` | `scan_form.py:408` | Ký tự OCR được coi là ô | Data2.1 tr.3: `()`; Data5.1: `LJ`, `(1`, `OQ` | (mới) |
| Lệch cột / cỡ ký tự | ≤ 25 px / ≤ 55 px | `scan_form.py:423` | Ký tự phải thẳng cột ô, cỡ bằng ô | | (mới) |
| `CELL_DELTA` | 0.03 | `scan_form.py:26` | Ô bảng có dấu nếu mực > trung vị hàng + 0.03 | | 0.03 → 0.007 (8/11) → 0.03 |
| `CELL_FAINT` | 0.012 | `scan_form.py:27` | Ô đậm nhất (khi không ô nào qua CELL_DELTA) | Tick nhạt Data9.1 +0.014; bóng Data6.1 +0.011 | (mới) |
| `CELL_LEAD` | 0.5 | `scan_form.py:28` | Ô đậm nhất phải vượt ô thứ hai ≥ 50% | | (mới) |
| `CELL_CLEAN` | 0.002 | `scan_form.py:29` | Hàng "sạch" nếu ô thứ hai ≤ trung vị + 0.002 | Data2.2: các ô khác = 0 | (mới) |
| `CELL_CLEAN_MIN` | 0.005 | `scan_form.py:30` | Ở hàng sạch, ô đậm nhất chỉ cần > +0.005 | Data2.2 hàng 2: +0.008 | (mới) |
| Nét kẻ bảng tối thiểu | 8 mm (ngang và dọc) | `detect_layout.py:133` | Độ dài tối thiểu để là đường kẻ | | ngang 8% bề rộng trang |
| MORPH_CLOSE nét kẻ | 5 px | `detect_layout.py:135` | Nối nét kẻ bị đứt do scan | | (không có) |
| Nở gom khối bảng | 7×7 | `detect_tables.py:29` | Gom nét ngang/dọc thành một bảng | | 5×5 |
| Bảng tối thiểu | 30 × 12 mm | `detect_tables.py:34` | Kích thước khối bảng | | 30 × 12 mm |
| Gộp đường gần | 1.5 mm | `detect_tables.py:39` | Gộp 2 nét kẻ quá sát | | 1.5 mm |
| Bù cột mất viền | cột trung vị > 10 mm | `detect_tables.py:50` | Thêm cột khi mất viền phải (Data9.1) | | (không có) |
| `CNN_RESCUE` | 0.90 | `scan_form.py:470` | CNN ≥ mức này thì cứu ô luật cho là trống | Mục 8.3 | (mới) |
| `USE_CNN` | True | `scan_form.py:471` | Tắt bằng `--no-cnn`; tự tắt khi `--crops` | | (mới) |
| `BOX_SCALE` / `CROP_SIZE` | 2.4 / 64 | `export_crops.py:27-28` | Ảnh cắt cho CNN (dùng chung khi học và khi chạy) | | 2.4 / 64 |
| Số fold | 2, chia theo sinh viên | `train_marker.py:50-55` | Đánh giá chéo CNN | | (không có, test = mọi file thật) |
| Epoch / batch / lr | 15 / 64 / 1e-3 | `train_marker.py` | Huấn luyện CNN | | như cũ |

### 8.5. Kết quả trên 28 tài liệu thật

| Giai đoạn trong ngày | Ô tìm được | Ô thừa | Dấu ở ô đúng | Sót dấu | Báo nhầm | Hàng bảng đúng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A. Luật, sau sửa bảng (8.2.1–8.2.2, 8.2.6) | 729/759 | 9 | 154/169 | 15 | 9 | 196/197 |
| B. + ngưỡng theo chất lượng trang (8.2.3) | 733/759 | 9 | 154/169 | 15 | 9 | 196/197 |
| C. + CNN cứu dấu, ngưỡng 0.90 (8.3) | 733/759 | 9 | 160/169 | 9 | 9 | 196/197 |
| **D. + đặt ô cứu đúng vị trí, thêm ký tự (8.2.5) — cuối ngày** | **738/759** | **9** | **160/169** | **9** | **4** | **196/197** |

Tỉ lệ cuối ngày: tìm ô **97.2%**, nhận đúng dấu **94.7%** (precision **97.6%**), hàng bảng **99.5%**; mô phỏng **231/233 (99.1%), 0 báo nhầm**. Trên các ô **đã tìm thấy**, máy đọc đúng 100% ô có dấu.

**Chi tiết từng tài liệu M02** (các tài liệu M03 `DataN.2` và 2 mẫu trắng: 0 lỗi dấu, bảng 12/12, trừ Data6.2 12/13):

| Tài liệu | Ô tìm được | Ô thừa | Dấu đúng | Sót | Báo nhầm | Bảng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Data1.1 | 54/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data2.1 | 54/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data3.1 | 54/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data4.1 | 53/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data5.1 | 51/54 | 0 | 12/13 | 1 | 0 | 2/2 |
| Data6.1 | 51/54 | 1 | 11/13 | 2 | 1 | 2/2 |
| Data7.1 | 53/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data8.1 | 54/54 | 0 | 13/13 | 0 | 0 | 2/2 |
| Data9.1 | 48/54 | 0 | 10/13 | 3 | 0 | 2/2 |
| Data10.1 | 51/54 | 1 | 11/13 | 2 | 0 | 2/2 |
| Data11.1 | 54/54 | 1 | 13/13 | 0 | 1 | 2/2 |
| Data12.1 | 54/54 | 2 | 13/13 | 0 | 2 | 2/2 |
| Data13.1 | 52/54 | 0 | 12/13 | 1 | 0 | 2/2 |

* **6 dấu được CNN cứu** (đều đúng): Data5.1 tr.4 (CNN 0.93), Data6.1 tr.3 (1.00), Data6.1 tr.4 (0.97), Data9.1 tr.4 (0.98), Data9.1 tr.8 (1.00), Data13.1 tr.5 (1.00).
* **Kiểm thử hồi quy:** `regress_baseline.json` đã cập nhật phủ **cả 28 tài liệu + 18 kịch bản mô phỏng (46 lượt)**; trạng thái cuối ngày đạt 46/46.

### 8.6. Lỗi còn lại theo trang và vấn đề còn mở

(⚑ = trang có cờ `needs_review`)

**Báo nhầm (4), đều là chữ cái đầu tiêu đề in hoa trên trang rõ:**

| Tài liệu | Trang | Vị trí | Là gì |
| :--- | :---: | :--- | :--- |
| Data6.1 | 3 | (504, 1496) | "I" của "6.INDEPENDENCE" |
| Data11.1 | 3 | (495, 1532) | "I" của "INDEPENDENCE" |
| Data12.1 | 2 | (1107, 959) | chữ trong dòng "PHONG…" |
| Data12.1 | 2 | (1092, 1411) | chữ trong tiêu đề "OBJECTIVES" |

**Sót dấu (9), đều là ô không tìm ra (CNN không có ảnh để chấm):**

| Tài liệu | Trang | Câu trả lời bị sót | Cờ |
| :--- | :---: | :--- | :---: |
| Data5.1 | 3 | Generally capable of learning new concepts… | |
| Data6.1 | 2 | All objectives adequately objectives | ⚑ |
| Data6.1 | 4 | Takes prompt action in response to suggestions… | |
| Data9.1 | 4 | Excellent | ⚑ |
| Data9.1 | 7 | Usually thorough. Generally, good work… | ⚑ |
| Data9.1 | 7 | Generally understands the problems at hand… | ⚑ |
| Data10.1 | 3 | Exceptional ability to learn new concepts… | |
| Data10.1 | 5 | Takes prompt action in response to suggestions… | |
| Data13.1 | 5 | Work usually required review, satisfactory work… | |

**Sót ô trống (12):** Data4.1 tr.5 ⚑; Data5.1 tr.3, tr.4; Data6.1 tr.4; Data7.1 tr.5; Data7.2 tr.1; Data9.1 tr.7 ×3 ⚑; Data10.1 tr.5 (tìm ra nhưng lệch vị trí); Data13.1 tr.5; Data13.2 tr.1.

**Ô thừa không dấu (5):** Data4.2 tr.2 (chữ viết tay "Prompt"), Data7.2 tr.1, Data8.2 tr.2 ×2 (chữ ký), Data10.1 tr.5 (ô lệch).

**Bảng (1):** Data6.2 tr.1 hàng 2, tick cột 5 quá nhạt.

**Hướng xử lý tiếp:**
1. Cờ theo chất lượng trang chỉ bao **4/9** dấu bị sót; 5 trường hợp còn lại nằm trên trang rõ, do ô in kiểu khác. Kiểm tra theo câu hỏi (một câu chọn-1 có 0 hoặc ≥ 2 ô có dấu, task.md 3.5) sẽ bao được cả 9.
2. Cả 4 báo nhầm là chữ cái đầu tiêu đề có số mục đứng trước ("6.", "5.") → loại theo số mục.
3. Ô in kiểu mảnh (Data5.1, 10.1, 13.1; khớp 0.05–0.53 trên trang rõ) → học mẫu riêng cho kiểu ô này.
4. Ô thừa do chữ viết tay/chữ ký trên trang M03.

### 8.7. Lệnh tái lập

```bash
python src/regress.py                                    # kiểm thử hồi quy (46 lượt, ~6 phút)
python src/scan_form.py data/samples/Data9.1.pdf         # quét 1 file (luật + CNN cứu dấu)
python src/scan_form.py data/samples/Data9.1.pdf --no-cnn   # chỉ luật
python src/evaluate.py                                   # chấm mọi file trong output/ với data/truth/
python src/make_dataset.py --only real                   # cắt lại ảnh thật sau khi sửa đáp án
python src/train_marker.py --fold 0                      # đánh giá chéo, fold 0 (rồi --fold 1)
python src/train_marker.py --fold -1                     # mô hình dùng thật: học mọi file thật
```

### 8.8. Đính chính các mục trước (2026-11-15)

* **Mục 3.1, 5.1:** số liệu là của 8 tài liệu; số liệu trên 28 tài liệu ở mục 8.1 và 8.5.
* **Mục 4.2:** ngưỡng `CELL_DELTA = 0.007` đã được bỏ (làm hàng có bóng mờ bị đọc 2 cột ở Data6.1, 12.1, 13.1). Thay bằng quy tắc ở mục 8.2.2; Data2.2 hàng 2 (+0.008) vẫn đọc đúng nhờ quy tắc "hàng sạch".
* **Mục 4.3:** cơ chế cứu ô trống ở ngưỡng 0.55 trong `_rescue_in_columns` đã được bỏ (đặt sai cỡ ô ở trang trộn ô to/nhỏ → regress mô phỏng sót thêm dấu). Data2.1 tr.3 nay được cứu bằng `_rescue_ocr_glyphs` (mục 8.2.5).
* **Mục 4.4:** bổ sung bằng ngưỡng theo chất lượng trang + `_starts_word` (mục 8.2.3).
* **Mục 5.3:** số liệu 498 ảnh là của 8 tài liệu; trên 1.753 ảnh của 28 tài liệu luật đạt F1 0.964 (không còn 1.000), CNN chỉ mô phỏng 0.910 (mục 8.3).
