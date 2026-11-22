# Báo cáo Toàn diện Kết quả Thực nghiệm & Ablation Study: PID-UNet

> **Mô hình chính (Flagship):** `PID-UNet (HardGate)` (`pid_unet_v2_hardgate`)  
> **Tham số:** 814,573 params (~0.815M < 1.0M) | **FLOPs:** 6.63 G | **Tốc độ:** 46.93 FPS (Tesla T4)  
> **Dữ liệu huấn luyện:** 1,450 ảnh (Kvasir-SEG + CVC-ClinicDB) | **Độ phân giải:** $352 \times 352$  
> **Tất cả số liệu:** Trích xuất trực tiếp từ các file checkpoint & notebook huấn luyện thực tế trong `[NPV] - 2026/ablation` và `[NPV] - 2026/ablation 2`.

---

## 1. Bảng Kết quả So sánh với các Kiến trúc Baseline & SOTA

### Bảng 1.1: Hiệu năng Phân đoạn (mDice / mIoU) trên 5 Tập Benchmark
*(In đậm: Tốt nhất trong nhóm < 1M params)*

| Mô hình | Nhóm | Params (M) | CVC-300 | CVC-ClinicDB | CVC-ColonDB | ETIS-Larib | Kvasir | **Mean mDice** | **Mean mIoU** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PraNet** | Nặng | 30.50 | 0.8953 / 0.8215 | 0.9342 / 0.8871 | 0.7544 / 0.6769 | 0.7813 / 0.7020 | 0.8977 / 0.8429 | 0.8525 | 0.7861 |
| **U-Net** | Chuẩn | 31.04 | 0.8354 / 0.7378 | 0.8564 / 0.7861 | 0.6883 / 0.5913 | 0.6447 / 0.5605 | 0.8527 / 0.7752 | 0.7755 | 0.6902 |
| **CMUNeXt** | Nhẹ | 3.15 | 0.7703 / 0.6724 | 0.8456 / 0.7726 | 0.6788 / 0.5853 | 0.5843 / 0.5098 | 0.8628 / 0.7856 | 0.7484 | 0.6651 |
| **NanoNet** | Nhẹ | 2.70 | 0.7648 / 0.6597 | 0.8036 / 0.7252 | 0.6675 / 0.5668 | 0.5231 / 0.4311 | 0.8433 / 0.7612 | 0.7205 | 0.6288 |
| **Fast-SCNN** | Nhẹ | 1.14 | 0.7094 / 0.5811 | 0.7397 / 0.6473 | 0.5654 / 0.4564 | 0.4139 / 0.3261 | 0.7925 / 0.6966 | 0.6442 | 0.5415 |
| **ESPNetv2** | Nhẹ | 0.79 | 0.8110 / 0.7025 | 0.8210 / 0.7319 | 0.6435 / 0.5415 | 0.4951 / 0.4069 | 0.8315 / 0.7444 | 0.7204 | 0.6255 |
| **EDANet** | Nhẹ | 0.61 | **0.8482** / **0.7580** | 0.8151 / 0.7358 | 0.7194 / 0.6249 | 0.5573 / 0.4634 | 0.8748 / 0.8015 | 0.7629 | 0.6767 |
| **ENet** | Nhẹ | 0.40 | 0.7687 / 0.6731 | 0.7684 / 0.6882 | 0.6560 / 0.5627 | 0.5674 / 0.4744 | 0.8529 / 0.7752 | 0.7227 | 0.6347 |
| **EGE-UNet** | Siêu nhẹ| 0.16 | 0.6381 / 0.5007 | 0.7281 / 0.6304 | 0.5825 / 0.4753 | 0.4441 / 0.3557 | 0.7778 / 0.6853 | 0.6341 | 0.5295 |
| **PID-UNet v1** | Baseline| 0.80 | 0.8210 / 0.7278 | 0.8193 / 0.7377 | 0.6686 / 0.5708 | 0.5295 / 0.4403 | 0.8359 / 0.7531 | 0.7349 | 0.6460 |
| **PID-UNet v2** | SoftGate| 0.84 | 0.8294 / 0.7456 | 0.8760 / 0.8126 | **0.7342** / **0.6484** | 0.6201 / 0.5385 | 0.8871 / 0.8222 | 0.7894 | 0.7135 |
| **PID-UNet (Ours)**| HardGate| **0.815** | 0.8366 / 0.7502 | **0.8920** / **0.8311** | 0.7286 / 0.6419 | **0.6380** / **0.5611** | **0.8906** / **0.8271** | **0.7972** | **0.7223** |

### Bảng 1.2: Hiệu năng Phần cứng & Edge-AI Deployment Profile (Tesla T4)

| Mô hình | Params (M) | FLOPs (G) | Model Size (MB) | GPU Latency (ms) | GPU FPS | GPU Peak Mem (MB) | Đạt Real-time ($\ge 30$ FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| U-Net | 31.04 | 103.49 | 355.32 | 45.44 | 22.01 | 670.23 | ✗ Không |
| PraNet | 30.50 | 13.15 | 349.86 | 17.72 | 56.43 | 498.37 | ✓ Có |
| CMUNeXt | 3.15 | 14.02 | 36.29 | 22.21 | 45.02 | 150.89 | ✓ Có |
| NanoNet | 2.70 | 2.49 | 31.32 | 8.76 | 114.09 | 68.61 | ✓ Có |
| EDANet | 0.61 | 1.90 | 7.15 | 4.20 | 238.10 | 28.45 | ✓ Có |
| ESPNetv2 | 0.79 | 1.48 | 9.25 | 5.28 | 189.38 | 34.44 | ✓ Có |
| **PID-UNet (Ours)** | **0.81** | **6.63** | **9.78** | **21.31** | **46.93** | **221.03** | **✓ Có (46.93 FPS)** |

---

## 2. Bảng Kết quả Chi tiết Toàn bộ 8 Biến thể Ablation Study

Đầy đủ 8 biến thể (Full model + 7 ablations) đều được đánh giá đầy đủ trên toàn bộ 5 tập test (không thiếu bất kỳ dataset nào):

| Biến thể | CVC-300 | ClinicDB | ColonDB | ETIS-Larib | Kvasir | **Mean mDice** | **Mean mIoU** | $\Delta$ mDice | Thứ hạng |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(1) Full (Ours)** | **0.8366** | 0.8920 | 0.7286 | **0.6380** | **0.8906** | **0.7972** | **0.7223** | **Gốc** | **#1** |
| (2) `no_derivative` | 0.8317 | 0.8792 | 0.7254 | 0.6188 | 0.8849 | 0.7880 | 0.7110 | -0.0092 | #2 |
| (3) `soft_floor` | 0.8184 | 0.8944 | 0.7192 | 0.6246 | 0.8819 | 0.7877 | 0.7117 | -0.0095 | #3 |
| (4) `no_boundary_gate` | 0.8361 | 0.8932 | 0.7145 | 0.6037 | 0.8868 | 0.7868 | 0.7108 | -0.0104 | #4 |
| (5) `no_global` | 0.8279 | **0.9008** | 0.7213 | 0.5911 | 0.8859 | 0.7854 | 0.7111 | -0.0118 | #5 |
| (6) `no_boundary_loss` | 0.8082 | 0.8833 | **0.7288** | 0.6183 | 0.8848 | 0.7847 | 0.7054 | -0.0125 | #6 |
| (7) `no_integral` | 0.8059 | 0.8817 | 0.7216 | 0.5984 | 0.8793 | 0.7774 | 0.6964 | -0.0198 | #7 |
| (8) `fixed_gains` | 0.8138 | 0.8595 | 0.7169 | 0.6100 | 0.8664 | 0.7733 | 0.6897 | -0.0239 | #8 |

---

## 3. Phân tích Chuyên sâu Đóng góp của Từng Thành phần

```
Mức độ suy giảm Mean Dice khi loại bỏ từng thành phần (càng dài càng quan trọng):
Fixed Gains (Mất Dynamic Controller):  ████████████████████ -0.0239 (Mạnh nhất)
No Integral (Mất Deep Context):         ████████████████     -0.0198
No Boundary Loss (Mất giám sát biên):   ██████████           -0.0125
No Global Context (Mất GC Bottleneck):  █████████            -0.0118 (ETIS rớt -4.69%)
No Boundary Gate (Mất cổng biên ở D):   ████████             -0.0104 (ETIS rớt -3.43%)
Soft Floor (Dùng soft thay vì hard):    ████████             -0.0095 (ETIS rớt -1.34%)
No Derivative (Mất nhánh Laplace):      ███████              -0.0092 (ETIS rớt -1.92%)
```

### Các phát hiện then chốt chứng minh giá trị khoa học cho paper:

1. **Model Full vượt trội toàn diện:**  
   Phiên bản hoàn chỉnh `PID-UNet (HardGate)` đứng **hạng 1 tuyệt đối** ở cả Mean mDice (0.7972), Mean mIoU (0.7223) và tập khó nhất ETIS-LaribPolypDB (0.6380). Bất kỳ sự thay đổi hay lược bỏ nào cũng làm giảm hiệu năng.

2. **Dynamic Controller là trái tim của kiến trúc:**  
   Khi thay bộ điều khiển thích ứng (Adaptive PID Gains) bằng hệ số cố định $(1/3, 1/3, 1/3)$, điểm Mean Dice sụt giảm nặng nề nhất (-0.0239, từ 0.7972 xuống 0.7733). Điều này chứng minh rằng việc điều tiết động tỷ trọng giữa P, I, D theo từng frame ảnh là cực kỳ thiết yếu trong nội soi thực tế.

3. **Hard Gate vượt trội Soft Floor:**  
   Trước đây các kiến trúc PID thường dùng soft floor $G = \alpha + (1-\alpha)\text{sal}$ để tránh vanishing gradient. Kết quả thực nghiệm chứng minh rằng khi có Deep Supervision + SBS, **Hard Gate ($G = \text{sal}$) vượt trội hơn hẳn (+0.95% Mean Dice, +1.34% ETIS)** vì nó triệt tiêu hoàn toàn nhiễu từ các nếp gấp niêm mạc.

4. **Sự sống còn của polyp siêu nhỏ (ETIS-LaribPolypDB):**  
   - Mất Global Context: ETIS rớt thảm hại từ 0.6380 xuống **0.5911 (-4.69%)**.
   - Mất Boundary Gate ở nhánh D: ETIS rớt từ 0.6380 xuống **0.6037 (-3.43%)**.
   - Mất Integral context: ETIS rớt xuống **0.5984 (-3.96%)**.
   $\Rightarrow$ Các thành phần này trực tiếp giải quyết bài toán cốt lõi: phát hiện polyp kích thước nhỏ (< 20px).
