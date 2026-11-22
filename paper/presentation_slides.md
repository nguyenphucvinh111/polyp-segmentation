# Kịch bản Slide Báo cáo Khoa học: PID-UNet Hardgate

> **Dựa trên bản thảo làm việc ngày 18/11/2026:**  
> *"Hợp nhất đặc trưng PID có dẫn hướng saliency cho phân đoạn polyp với mạng dưới một triệu tham số"*  
> **Tên tiếng Anh:** *Saliency-Guided PID Feature Fusion for Sub-Million-Parameter Polyp Segmentation*  
> **Triết lý trình bày:** Khách quan, khoa học, lập luận chặt chẽ dựa trên bằng chứng thực nghiệm, không nói quá (no overclaiming).

---

## CẤU TRÚC BÁO CÁO (14 SLIDES)

```
[1. Tiêu đề] ──► [2. Đặt vấn đề] ──► [3. Câu hỏi nghiên cứu & Mục tiêu]
     │
     ▼
[4. Kiến trúc tổng thể PID-UNet] ──► [5. Khối hạ mẫu IPD & Bottleneck HARF-GC]
     │
     ▼
[6. Module SG-PID Fusion] ──► [7. Bản chất Hardgate & Dynamic Controller]
     │
     ▼
[8. Nhánh biên SBS & Hàm mục tiêu] ──► [9. Thiết lập thực nghiệm & Protocol]
     │
     ▼
[10. Kết quả trên 5 tập dữ liệu] ──► [11. Đối chiếu lịch sử với v2]
     │
     ▼
[12. Phân tích Ablation Study] ──► [13. Thảo luận & Hạn chế] ──► [14. Kết luận & Hướng đi tiếp]
```

---

### SLIDE 1: TRANG TIÊU ĐỀ
* **Tiêu đề báo cáo:** Hợp nhất đặc trưng PID có dẫn hướng Saliency cho phân đoạn polyp với mạng dưới một triệu tham số
* **Tên tiếng Anh:** *Saliency-Guided PID Feature Fusion for Sub-Million-Parameter Polyp Segmentation*
* **Tên triển khai mã nguồn:** `pid_unet_v2_hardgate` (814.573 tham số)
* **Báo cáo viên:** [Tên tác giả]
* **Đơn vị công tác:** [Tên trường / Viện nghiên cứu / Lab]
* **Speaker Notes:**  
  *"Kính thưa Hội đồng và các thầy cô, hôm nay tôi xin trình bày báo cáo về thiết kế và đánh giá thực nghiệm cơ chế hợp nhất đặc trưng lấy cảm hứng từ PID có dẫn hướng saliency cho bài toán phân đoạn polyp nội soi trong ngân sách dưới 1 triệu tham số."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & BỐI CẢNH LÂM SÀNG
* **Bài toán:** Phân đoạn tổn thương polyp trên ảnh nội soi đại tràng ở mức điểm ảnh (pixel-level).
* **Đặc thù quang học & hình thái:**
  * Kích thước, màu sắc và kết cấu tổn thương biến thiên rất lớn.
  * Ranh giới giữa mô bệnh (polyp) và niêm mạc ruột lành thường mờ nhạt, khó phân tách.
* **Mâu thuẫn trong mạng Encoder–Decoder nhẹ:**
  * *Đặc trưng nông:* Giữ chi tiết không gian nhưng dễ kéo theo nhiễu nền (nếp gấp niêm mạc, đốm chói sáng).
  * *Đặc trưng sâu:* Cung cấp ngữ cảnh nhận diện nhưng độ phân giải thấp, làm mờ biên.
  * *Ràng buộc phần cứng:* Ngân sách dưới 1M tham số không cho phép mở rộng mạng tùy tiện, đòi hỏi phải tối ưu **cơ chế hợp nhất đặc trưng (feature fusion)**.

---

### SLIDE 3: CÔNG TRÌNH LIÊN QUAN & ĐỊNH VỊ ĐÓNG GÓP
* **Nền tảng U-Net & Hạn chế của Plain Skip Connection:**
  * *U-Net kinh điển (MICCAI 2015):* Là xương sống phân đoạn y tế với các đường skip connection truyền thẳng đặc trưng từ Encoder sang Decoder.
  * *Hạn chế trong nội soi polyp:*
    1. *Truyền thụ động (Unselective):* Đưa toàn bộ nhiễu cao tần (nếp gấp niêm mạc, bóng phản xạ ánh sáng) sang Decoder $\to$ Gây nhiều dương tính giả (False Positives).
    2. *Quá nặng:* U-Net chuẩn có **31.04 triệu tham số**, không thể chạy trên thiết bị biên nội soi cầm tay.
* **Kế thừa & Cải tiến từ PraNet & PIDNet:**
  * *PraNet (MICCAI 2020):* Reverse Attention để hiệu chỉnh vùng và biên polyp.
  * *PIDNet (CVPR 2023):* 3 nhánh chi tiết ($P$), ngữ cảnh ($I$) và biên ($D$) trong phân đoạn thời gian thực.
* **Câu hỏi nghiên cứu & Đóng góp của PID-UNet:**
  > *“Liệu việc thay thế các đường plain skip connection thụ động của U-Net bằng khối SG-PID Fusion (điều tiết chi tiết P và cao tần D bằng saliency và biên) có giúp đạt hiệu năng vượt trội trong ngân sách dưới 1 triệu tham số hay không?”*
  * **Bước tiến:** Giảm tham số từ 31.04M xuống **0.815M (giảm 38 lần)**, đồng thời vượt U-Net chuẩn về độ chính xác phân đoạn (**0.7972 vs. 0.7755 Mean Dice**).
  * Định vị khoa học: Phép tương tự trong không gian đặc trưng (Feature Analogy), kiểm chứng qua thực nghiệm chặt chẽ.

---

### SLIDE 4: KIẾN TRÚC TỔNG THỂ MODEL (PID-UNET HARDGATE)
* **Chèn hình:** Figure 1 ([figure1_pid_unet_architecture.svg](file:///d:/TÀI%20LIỆU%20HỌC%20TẬP/LIDP/Polyp_Segmentation_Research-main/paper/figure1_pid_unet_architecture.svg))
* **Quy mô:** **814.573 tham số** (~0.815M, huấn luyện từ đầu).
* **Các mức phân giải và kênh số:**
  * $E1$: 32 kênh, $352 \times 352$ (bản đồ chi tiết cao nhất).
  * $E2$: 64 kênh, $176 \times 176$.
  * $E3$: 96 kênh, $88 \times 88$.
  * $E4$: 128 kênh, $44 \times 44$.
  * $\text{Bottleneck}$: 192 kênh, $22 \times 22$ (đáy mạng nén sâu nhất).
* **Quy trình luồng dữ liệu:**
  * Encoder đi xuống qua 4 lần hạ mẫu.
  * Nhánh biên SBS gom 5 mức dự đoán logits biên $B$ ($1 \times 176 \times 176$).
  * Decoder chạy ngược từ dưới lên qua 4 khối **SG-PID Fusion** (Fusion 4 $\to$ 1).
  * 4 Head giám sát sâu khi train; **Head 1 là đầu ra dự đoán cuối cùng**.

---

### SLIDE 5: TRÍCH XUẤT ĐẶC TRƯNG & HẠ MẪU BẢO TOÀN THÔNG TIN
* **Khối tích chập bất đối xứng (AsymmetricRFB):**
  * Dùng tích chập phân rã $3 \times 1$ và $1 \times 3$ với dilation khác nhau $\to$ Giảm tham số và tăng độ sâu phi tuyến.
* **Hạ mẫu IPD (Info-Preserving Downsampler):**
  * Hai đường song song: $\text{Conv}_{3\times3, s=2}$ ghép với $\text{MaxPool}_{2\times2} + \text{Conv}_{1\times1}$.
  * Tạo hai cách tổng hợp thông tin khi giảm độ phân giải: một đường học bộ lọc không gian, một đường giữ lại phản ứng cực đại.
* **Đáy mạng HARF-GC Bottleneck ($22 \times 22$):**
  * 4 nhánh dãn nở $d \in \{1, 2, 4, 8\}$ + 1 nhánh Global Average Pooling.
  * Thêm khối cộng ngữ cảnh toàn cục:
    $$Z' = Z + W_2 \, \text{ReLU}\big(W_1 \, \text{GAP}(Z)\big)$$
    (với $W_1$ nén từ 192 xuống 24, $W_2$ mở rộng về 192, broadcast không gian).

---

### SLIDE 6: MODULE CỐT LÕI - SG-PID FUSION
* **Chèn hình:** Figure 2 ([figure2_sgpid_module.svg](file:///d:/TÀI%20LIỆU%20HỌC%20TẬP/LIDP/Polyp_Segmentation_Research-main/paper/figure2_sgpid_module.svg))
* **Bốn nguồn thông tin đầu vào:**
  * $X_P$ (từ Encoder) $\to P = \phi_P(X_P)$: đại diện chi tiết nông.
  * $X_I$ (từ Decoder sâu hơn) $\to I = \phi_I(X_I)$: đại diện ngữ nghĩa sâu.
  * $S = \sigma(\psi(X_I))$: Saliency ngữ nghĩa liên tục trong $[0, 1]$.
  * $\bar{b} = \sigma(\text{Resize}(B))$: Bản đồ hướng dẫn biên.
* **Ba thành phần song song:**
  * **Thành phần P:** $P_S = P \odot S$ (chi tiết được điều tiết bởi saliency).
  * **Thành phần I:** $I$ (giữ nguyên ngữ nghĩa tầng sâu).
  * **Thành phần cao tần D:**
    $$D = |P - \text{AvgPool}_{3\times3}(P)| \odot \sigma(\bar{B}) \odot S$$
    (Sai khác với trung bình lân cận nhấn mạnh biến thiên không gian, được nhân kép với biên và saliency).

---

### SLIDE 7: BẢN CHẤT HARDGATE & DYNAMIC CONTROLLER
* **Thuật ngữ "Hardgate" thực chất là gì?**
  * Không phải mask nhị phân $\{0, 1\}$, không phải threshold cứng.
  * Là **loại bỏ mức sàn dương $f$** của gate: $g = S$ (thay vì $g = f + (1-f)S$).
  * Giá trị $S$ vẫn liên tục trong khoảng $[0, 1]$ từ hàm sigmoid.
  * *Động cơ thiết kế:* Khi saliency thấp (vùng niêm mạc bình thường), đường truyền chi tiết được phép giảm sát về 0 để triệt tiêu dương tính giả.
* **Bộ điều khiển thích ứng (Adaptive PID Controller):**
  $$[k_P, k_I, k_D] = \text{Softmax}\big(\text{MLP}(\text{GAP}(X_I))\big)$$
  * Là **3 hệ số vô hướng cho mỗi ảnh** (không phải bản đồ pixel), tổng bằng 1.
* **Biểu thức tổng hợp & Đường Residual $I$:**
  $$Y = \text{Fuse}\big(\text{SE}(k_P P_S + k_I I + k_D D)\big) + I$$
  * Đường $+ I$ nằm **ngoài tổng có trọng số**, đảm bảo ngữ nghĩa sâu luôn được truyền qua ngay cả khi $k_I$ nhỏ.

---

### SLIDE 8: NHÁNH BIÊN SBS & HÀM MỤC TIÊU
* **Nhánh biên đa mức (SBS):**
  * Chiếu $E_1, E_2, E_3, E_4, Z'$ về 24 kênh, căn chỉnh về độ phân giải của $E_2$ ($176 \times 176$).
  * Nối kênh ($120$ch) $\to$ Tích chập bất đối xứng $\to$ Logits biên $B$.
  * Trong suy luận, $B$ **vẫn tham gia fusion**, không được thay bằng ground truth.
* **Hàm mục tiêu hỗn hợp:**
  $$\mathcal{L} = \sum_{j=1}^4 \mathcal{L}_{\text{str}}^{(j)} + 0.2 \, \mathcal{L}_{\text{boundary}}$$
  * $\mathcal{L}_{\text{str}} = \mathcal{L}_{\text{WBCE}} + \mathcal{L}_{\text{WIoU}}$ với trọng số không gian:
    $$w = 1 + 5 \, |\text{AvgPool}_{31\times31}(G) - G|$$
  * Nhãn biên tạo từ gradient hình thái $3 \times 3$ của ground truth nhị phân; giám sát bằng weighted BCE + Dice loss.

---

### SLIDE 9: THIẾT LẬP THỰC NGHIỆM & PROTOCOL
* **Tập dữ liệu huấn luyện:**
  * 1.450 cặp ảnh/mask chuẩn (TrainDataset).
  * Split cố định seed 42: 1.305 train / 145 validation.
* **5 Tập kiểm tra độc lập (Zero-shot test):**
  * `CVC-300` (60 ảnh), `CVC-ClinicDB` (62 ảnh), `CVC-ColonDB` (380 ảnh), `ETIS-LaribPolypDB` (196 ảnh), `Kvasir-SEG` (100 ảnh).
* **Cấu hình huấn luyện tham chiếu:**
  * AdamW, $\text{lr}=3\times10^{-4}$, weight decay $10^{-4}$, cosine decay, 5 epochs warmup, 100 epochs, batch size 8.
  * Kích thước cơ sở $352 \times 352$, stochastic multi-scale ngẫu nhiên $\{256, 352, 448\}$.
  * AMP bật trên CUDA.
* **Pipeline đánh giá:**
  * Logits cuối nội suy về kích thước ảnh gốc $\to$ Sigmoid $\to$ Chuẩn hóa min–max per-image $\to$ Ngưỡng 0.5.

---

### SLIDE 10: KẾT QUẢ THỰC NGHIỆM SƠ BỘ TRÊN 5 TẬP
* **Bảng 1: Kết quả một lần chạy của mô hình hardgate tham chiếu:**

| Tập kiểm tra | Dice ↑ | IoU ↑ | Recall ↑ | Precision ↑ | Specificity ↑ | HD95 ↓ | MAE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CVC-300** | 0.8366 | 0.7502 | 0.9609 | 0.7762 | 0.9888 | 75.5060 | 0.0142 |
| **CVC-ClinicDB** | 0.8920 | 0.8311 | 0.9348 | 0.8815 | 0.9914 | 32.6096 | 0.0193 |
| **CVC-ColonDB** | 0.7286 | 0.6419 | 0.8111 | 0.7328 | 0.9768 | 103.7807 | 0.0470 |
| **ETIS-LaribPolypDB** | 0.6380 | 0.5611 | 0.7307 | 0.6053 | 0.9796 | 281.2059 | 0.0310 |
| **Kvasir** | 0.8906 | 0.8271 | 0.8900 | 0.9260 | 0.9854 | 74.6279 | 0.0400 |
| **Macro-average** | **0.7972** | **0.7223** | **0.8655** | **0.7843** | **0.9844** | **113.5460** | **0.0303** |

* **Nhận xét khoa học:**
  * Dice đạt xấp xỉ 0.89 trên ClinicDB và Kvasir, nhưng thấp hơn trên ColonDB (0.7286) và ETIS (0.6380).
  * Trên CVC-300: Recall rất cao (0.9609) nhưng Precision đạt 0.7762 $\to$ cần tiếp tục khảo sát việc hạn chế vùng dự đoán dư.

---

### SLIDE 11: ĐỐI CHIẾU LỊCH SỬ VỚI BẢN V2 CŨ
* **Bảng 2: Đối chiếu lịch sử (không phải ablation kiểm soát):**

| Dataset | Dice v2 cũ | Dice Hardgate | Chênh lệch |
|:---|:---:|:---:|:---:|
| CVC-300 | 0.8294 | 0.8366 | +0.0072 |
| CVC-ClinicDB | 0.8760 | 0.8920 | +0.0160 |
| CVC-ColonDB | 0.7342 | 0.7286 | −0.0056 |
| ETIS-LaribPolypDB | 0.6201 | 0.6380 | +0.0179 |
| Kvasir | 0.8871 | 0.8906 | +0.0035 |
| **Macro-average** | **0.7894** | **0.7972** | **+0.0078** |

* **Nhận định thận trọng:**
  * Hardgate có Dice cao hơn ở 4/5 tập, nhưng ColonDB giảm nhẹ.
  * Trên CVC-300: Dice tăng +0.72% nhưng HD95 tăng từ 70.12 lên 75.50 $\to$ *chưa hỗ trợ khẳng định hardgate cải thiện đồng thời cả vùng và biên trên mọi tập*.
  * Do v2 cũ và bản hardgate có khác biệt ở bottleneck, chênh lệch này chỉ là đối chiếu lịch sử; phép so sánh trực tiếp phải là **Full vs. Soft_floor** trong cùng study.

---

### SLIDE 12: PHÂN TÍCH ABLATION STUDY (7 BIẾN THỂ CÓ KIỂM SOÁT)
* **Bảng 3: Kết quả Ablation với seed 42:**

| Cấu hình | CVC-300 | ClinicDB | ColonDB | ETIS | Kvasir | **Macro Dice** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full (Tham chiếu)** | **0.8366** | 0.8920 | 0.7286 | **0.6380** | **0.8906** | **0.7972** |
| `fixed_gains` | 0.8138 | 0.8595 | 0.7169 | 0.6100 | 0.8664 | 0.7733 (-2.39%) |
| `no_integral` | 0.8059 | 0.8817 | 0.7216 | 0.5984 | 0.8793 | 0.7774 (-1.98%) |
| `no_boundary_loss` | 0.8082 | 0.8833 | 0.7288 | 0.6183 | 0.8848 | 0.7847 (-1.25%) |
| `no_global` | 0.8279 | 0.9008 | 0.7213 | 0.5911 | 0.8859 | 0.7854 (-1.18%) |
| `no_boundary_gate` | 0.8361 | 0.8932 | 0.7145 | 0.6037 | 0.8868 | 0.7868 (-1.04%) |
| `soft_floor` | 0.8184 | 0.8944 | 0.7192 | 0.6246 | 0.8819 | 0.7877 (-0.95%) |
| `no_derivative` | 0.8317 | 0.8792 | 0.7254 | 0.6188 | 0.8849 | 0.7880 (-0.92%) |

* **Các quan sát quan trọng:**
  1. `fixed_gains` giảm mạnh nhất (-2.39%) $\to$ Xác nhận vai trò tối quan trọng của bộ điều khiển thích ứng.
  2. `no_integral` giảm 1.98% $\to$ Xác nhận vai trò của đường truyền ngữ nghĩa sâu.
  3. `soft_floor` có Dice CVC-300 (0.8184) thấp hơn Full (0.8366), Precision giảm từ 0.7762 xuống 0.7583 $\to$ Phù hợp với giả thuyết mức sàn có thể cho vùng dự đoán dư đi qua.
  4. `no_global` làm ETIS rớt mạnh từ 0.6380 xuống 0.5911 $\to$ Ngữ cảnh toàn cục ở đáy mạng đặc biệt quan trọng với polyp nhỏ.

---

### SLIDE 13: THẢO LUẬN & GIỚI HẠN CỦA NGHIÊN CỨU (LIMITATIONS)
* **Bản chất hai mặt của Saliency Gate chung:**
  * Việc đặt một nguồn saliency chung lên cả P và D giúp tinh gọn mô hình, nhưng nếu saliency dự đoán sai, cả chi tiết lẫn cao tần đều có thể bị triệt tiêu sai lệch.
* **Các giới hạn thực nghiệm cần hoàn thiện:**
  1. *Số liệu hiện tại phản ánh 1 lần chạy:* Cần chạy lặp trên nhiều seeds (42, 3407, 1234) để báo cáo dạng $\text{Mean} \pm \text{SD}$.
  2. *Phân chia dữ liệu theo ảnh:* Chưa chứng minh được tính độc lập hoàn toàn theo bệnh nhân/video.
  3. *Quy trình Min–Max threshold:* Cần đối chiếu đồng nhất với các bài báo khác khi so sánh.
  4. *Đo đạc phần cứng:* Cần profile latency/FPS chính thức trên cùng giao thức để công bố.

---

### SLIDE 14: KẾT LUẬN & HƯỚNG PHÁT TRIỂN TIẾP THEO
* **Kết luận chính:**
  * Xây dựng thành công kiến trúc encoder–decoder nhẹ (**814.573 tham số**) sử dụng cơ chế hợp nhất đặc trưng lấy cảm hứng từ PID.
  * Cơ chế trung tâm SG-PID kết hợp saliency không mức sàn, dẫn hướng biên và hệ số thích nghi theo đầu vào.
  * Kết quả sơ bộ đạt **Macro-average Dice 0.7972** và **CVC-300 Dice 0.8366**, cho thấy tiềm năng phân đoạn tốt trong ngân sách dưới 1 triệu tham số.
* **Kế hoạch tiếp theo trước khi nộp bài:**
  * Chạy kiểm chứng trên nhiều seed để lấy sai số chuẩn.
  * Bổ sung bảng đối chiếu chi tiết với các phương pháp nhẹ chính thức (EDANet, NanoNet).
  * Xuất bản đồ sai số (Error Maps) và trực quan hóa các ca thành công/thất bại.

---

### LỜI KẾT & CẢM ƠN
*"Xin chân thành cảm ơn Quý Thầy Cô và Hội đồng đã lắng nghe. Tôi rất mong nhận được các câu hỏi phản biện và ý kiến đóng góp quý báu!"*
