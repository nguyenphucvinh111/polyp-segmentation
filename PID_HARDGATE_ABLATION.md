# Ablation của model hardgate

Chạy [notebook Colab](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_hardgate_ablation_colab.ipynb).

Notebook người dùng cung cấp chỉ có lệnh `train.py --model pid_unet_v2_hardgate --seeds 42`, không lưu source hoặc git commit. Study này cố định định nghĩa hardgate trên GitHub tại **d667f89**, gồm hybrid RFB (1,2,4,8) có global pooling và GlobalContext cộng thêm. Đây là nguồn có thể xác minh hiện tại; chưa thể chứng minh lần train trong ảnh dùng đúng revision đó nếu thiếu manifest/checkpoint của lần train. Vì vậy cần train lại control trong cùng study.

`gate = saliency` vẫn là sigmoid liên tục, không phải gate nhị phân. Trong paper nên gọi là saliency gating without a floor. P/I/D là phép tương tự trên đặc trưng không gian, không phải điều khiển theo thời gian.

| Variant | Thay đổi so với hardgate đầy đủ |
|---|---|
| full | Control hardgate |
| soft_floor | Thêm floor học được 0.6 sigmoid(alpha), khởi tạo 0.35 |
| no_derivative | D = 0, giữ softmax ba gains, không chuẩn hóa lại P/I |
| no_integral | I = 0, loại bỏ ngữ cảnh sâu tích lũy trong PID block (đánh giá P+D) |
| fixed_gains | Kp = Ki = Kd = 1/3 |
| no_boundary_gate | Bỏ multiplier sigmoid(boundary) trong D |
| no_global | Bỏ GlobalContext cộng thêm; vẫn giữ global pooling bên trong hybrid RFB |
| no_boundary_loss | Giữ nguyên kiến trúc control nhưng tắt loss biên phụ trợ (boundary_weight = 0.0) |

Tên model là `pid_hardgate_ablation_<variant>`. Mỗi variant khởi tạo đầy đủ control trước khi thay thành phần; trọng số chung và RNG khởi tạo giống nhau theo seed. Các tham số không hoạt động như alpha trong hardgate hoặc controller trong fixed_gains vẫn được giữ để cô lập tác động hàm forward; không diễn giải số tham số là chi phí triển khai tối thiểu.

Giữ structure loss bốn head và loss biên hiện có ×0.2 (ngoại trừ variant `no_boundary_loss` dùng 0.0), AdamW 3e-4, warmup 5, cosine 100 epochs, batch 8, multiscale 352, AMP, TTA tắt, train từ đầu. Theo source hardgate được tham chiếu, `drop_last=True` (163 updates với 1305 ảnh train). Điều này cũng tránh BatchNorm trên global pooling với batch cuối chỉ một ảnh. Split cố định 1305/145 từ TrainDataset; checkpoint chọn theo validation Dice.

Sau khi mount Drive và setup repo:

```bash
python run_hardgate_ablation.py --seeds 42 --output_root /content/drive/MyDrive/pid_hardgate_ablation_v1
```

Sau đó bổ sung hai seed cho TẤT CẢ các hàng:

```bash
python run_hardgate_ablation.py --seeds 42 3407 1234 --output_root /content/drive/MyDrive/pid_hardgate_ablation_v1
```

Tổng cộng 8 variants × 3 seeds = 24 runs (hoặc 8 runs cho seed 42 ban đầu). Run hoàn tất được bỏ qua, run dở được resume. Chỉ một runner ghi vào thư mục tại một thời điểm. Source, phiên bản runtime, nội dung dữ liệu và split được khóa trong manifest; không trộn với study v2 cũ. Cần giữ nguyên toàn bộ thư mục Drive, đặc biệt best/last checkpoint, config, manifest, metrics và logs.

```bash
python summarize_hardgate_ablation.py --seeds 42 3407 1234 --output_root /content/drive/MyDrive/pid_hardgate_ablation_v1
```

Báo cáo CSV/Markdown/LaTeX có năm tập, cả 10 metrics, mean ± sample SD và delta so với **hardgate full** cùng seeds. Một seed không có SD. Run thiếu không được tính thành 0. Không chọn epoch hoặc seed bằng CVC-300.

Kết quả ảnh: Mean Dice 0.7972, CVC-300 Dice 0.8366. Đây là kết quả một lần train đã được dùng để chọn ứng viên chính; cần công bố quá trình chọn model và tránh gọi bộ test này là hoàn toàn chưa được quan sát. Chưa có kết quả 18 runs để khẳng định lợi ích từng module hay độ ổn định. Không dùng số v2 trước đó làm control trực tiếp vì backbone/recipe có thể khác.
