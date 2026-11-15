# PID-UNet Ablation Study Protocol & Guide

Bộ tài liệu hướng dẫn và ma trận thực nghiệm **Ablation Study** cho mô hình **PID-UNet** (kiến trúc đề xuất dựa trên họ `pid_unet_v2`), thiết kế theo chuẩn các hội nghị/tạp chí quốc tế hàng đầu (MICCAI, IEEE TMI, CVPR).

---

## 1. Mục tiêu khoa học & Các giả thuyết cần chứng minh (Hypotheses)

Một bài báo khoa học chất lượng cao đòi hỏi Ablation Study phải trả lời được 4 câu hỏi lâm sàng và kỹ thuật cốt lõi:

1. **Giả thuyết 1 (PID Component Synergism):** 
   - Liệu việc kết hợp cả 3 nhánh **Tỉ lệ (Proportional - P)**, **Tích phân (Integral - I)**, và **Đạo hàm (Derivative - D)** có thực sự vượt trội hơn các mô hình chỉ có $P+I$ (thiếu biên) hoặc $P+D$ (thiếu ngữ cảnh)?
   - Bộ điều khiển thích nghi động **Dynamic Adaptive Gating** ($K_p, K_i, K_d$ tự tinh chỉnh theo từng khung hình) có đóng góp gì so với việc gán cứng trọng số tĩnh ($1/3, 1/3, 1/3$)?
2. **Giả thuyết 2 (Boundary Guidance & Supervision):**
   - Giám sát tường minh nhánh viền bằng hàm loss phụ trợ ($\mathcal{L}_{boundary}$ với $\lambda_b = 0.2$) giúp mô hình định hình sắc nét ranh giới tổn thương như thế nào so với việc tắt loss biên ($\lambda_b = 0.0$)?
3. **Giả thuyết 3 (Soft/Bounded Gating vs Hard Gating):**
   - Tại sao cơ chế **Bounded Soft Gate** ($0.6 \cdot \sigma(\alpha) + (1-\text{gate})\cdot S_{sal}$) lại quan trọng? Cổng cứng (Hard Saliency Gate) có làm triệt tiêu đặc trưng ở các polyp nhỏ/phẳng (diminutive/flat lesions) hay không?
4. **Giả thuyết 4 (Info-Preserving Downsampler & Global Context):**
   - Khối **Info-Preserving Downsampler** (Dual-path Conv + MaxPool Concat) bảo toàn tổn thương nhỏ (< 20px) tốt hơn thế nào so với phép giảm chiều chuẩn (Standard Strided Conv)?
   - Khối **Global Context (GC)** ở bottleneck có giúp loại bỏ nhiễu phản xạ ánh sáng (specular glare) của dịch ruột hay không?

---

## 2. Ma trận cấu hình thực nghiệm (Ablation Matrix)

Tất cả các biến thể đều tuân thủ nghiêm ngặt ràng buộc: **Số tham số $< 1.0\text{M}$** và chạy theo cùng một huấn luyện chuẩn (100 epochs, AdamW lr=3e-4, Cosine decay, 5 warmup epochs, 352x352 stochastic multi-scale).

| Mã cấu hình | Tên Model CLI / Flag | InfoDown | RFB | GlobalContext | P-branch | I-branch | D-branch | Adaptive PID | Bounded Gate | Boundary Loss ($\lambda_b$) | Params (M) | Giả thuyết kiểm chứng |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M0** | `pid_unet_v2_stddown` | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.2 | 0.995M | Vai trò của Info-Preserving Downsampler |
| **M1** | `pid_unet_v2_nogc` | $\checkmark$ | $\checkmark$ | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.2 | 0.805M | Đóng góp của Bottleneck Global Context |
| **M2** | `pid_unet_v2_nod` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | $\checkmark$ | $\checkmark$ | 0.2 | 0.815M | Loại bỏ nhánh Derivative ($K_d = 0$) |
| **M3** | `pid_unet_v2_noi` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.2 | 0.815M | Loại bỏ nhánh Integral ($K_i = 0$) |
| **M4** | `pid_unet_v2_static` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ (cố định $1/3$) | $\checkmark$ | 0.2 | 0.815M | Dynamic PID vs Static/Equal PID Gains |
| **M5** | `pid_unet_v2_hardgate` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ (Hard gate) | 0.2 | 0.815M | Hard Saliency Gate vs Bounded Soft Gate |
| **M6** | `pid_unet_v2_nobound` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | Soft v1/v5 | 0.2 | 0.815M | SoftPIDBAG (v1/v5) vs BoundedPIDBlock (v2) |
| **M7** | `pid_unet_v2` (`--boundary_weight 0.0`) | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.0 | 0.815M | Vai trò của Giám sát viền tường minh |
| **M8** | **`pid_unet_v2` (Full Proposed)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.2 | **0.815M** | **Mô hình hoàn chỉnh đề xuất** |

---

## 3. Lệnh chạy thực nghiệm từng bước trên Google Colab T4

Trên Colab notebook (`pidunet_v2.ipynb`), sau khi clone code mới nhất:

```bash
!git pull origin main
```

Bạn có thể chạy độc lập từng cấu hình (chỉ mất ~50 phút/cấu hình cho 100 epochs trên GPU T4 Colab):

### Bước 1: Ablation các nhánh PID và Bộ điều khiển (PID Dynamics)
```bash
# 1. w/o Derivative branch (M2)
!python train.py --model pid_unet_v2_nod --seeds 42

# 2. w/o Integral branch (M3)
!python train.py --model pid_unet_v2_noi --seeds 42

# 3. Static/Equal PID weights (M4)
!python train.py --model pid_unet_v2_static --seeds 42
```

### Bước 2: Ablation cơ chế Cổng điều tiết & Viền (Gating & Boundary)
```bash
# 4. Hard Saliency Gate (M5)
!python train.py --model pid_unet_v2_hardgate --seeds 42

# 5. w/o Boundary Supervision (M7)
!python train.py --model pid_unet_v2 --run_name pid_unet_v2_noboundloss --boundary_weight 0.0 --seeds 42
```

### Bước 3: Ablation khối Downsampling & Global Context
```bash
# 6. Standard Downsample (M0)
!python train.py --model pid_unet_v2_stddown --seeds 42

# 7. w/o Global Context (M1)
!python train.py --model pid_unet_v2_nogc --seeds 42
```

---

## 4. Vị trí xuất kết quả tự động

Sau khi chạy xong mỗi câu lệnh, pipeline sẽ tự động:
1. Lưu checkpoint tốt nhất tại: `./snapshots/<tên_model>/best.pth`
2. Đánh giá cross-dataset trên 5 tập dữ liệu: `Kvasir-SEG`, `CVC-ClinicDB`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, `CVC-300`.
3. Lưu bảng kết quả chi tiết:
   - File CSV: `./results/<tên_model>/metrics.csv`
   - Bảng LaTeX: `./results/<tên_model>/metrics.tex`
   - Bảng đo hiệu năng Edge-AI (FPS, Latency, Mem): `./results/<tên_model>/efficiency.csv`

---

## 5. Mẫu bảng tổng hợp số liệu cho Bài báo (LaTeX Template)

```latex
\begin{table*}[t]
\centering
\caption{Ablation study on the architectural components of the proposed PID-UNet. All models are trained under identical conditions on the TrainDataset and evaluated across five public benchmarks.}
\label{tab:ablation}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccccc|c|cc|cc|cc}
\hline
\multirow{2}{*}{\textbf{Configuration}} & \multicolumn{5}{c|}{\textbf{Architectural Components}} & \textbf{Params} & \multicolumn{2}{c|}{\textbf{CVC-ClinicDB}} & \multicolumn{2}{c|}{\textbf{ETIS-Larib}} & \multicolumn{2}{c}{\textbf{Mean (All 5)}} \\
& InfoDown & GC & P-branch & I-branch & D-branch & (M) & mDice & mIoU & mDice & mIoU & mDice & mIoU \\
\hline
(M0) w/o InfoDown & $\times$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & 0.995 & --- & --- & --- & --- & --- & --- \\
(M1) w/o Global Context & $\checkmark$ & $\times$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & 0.805 & --- & --- & --- & --- & --- & --- \\
(M2) w/o Derivative (Kd=0) & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\times$ & 0.815 & --- & --- & --- & --- & --- & --- \\
(M3) w/o Integral (Ki=0) & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\times$ & $\checkmark$ & 0.815 & --- & --- & --- & --- & --- & --- \\
(M4) Static PID Gains & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & 0.815 & --- & --- & --- & --- & --- & --- \\
(M5) Hard Saliency Gate & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & 0.815 & --- & --- & --- & --- & --- & --- \\
(M7) w/o Boundary Loss & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & 0.815 & --- & --- & --- & --- & --- & --- \\
\textbf{(M8) Proposed PID-UNet} & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & \textbf{0.815} & \textbf{---} & \textbf{---} & \textbf{---} & \textbf{---} & \textbf{---} & \textbf{---} \\
\hline
\end{tabular}
}
\end{table*}
```
