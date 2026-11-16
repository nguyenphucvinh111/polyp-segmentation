# Ablation study cho PID-UNet v2

Mục tiêu: đo đóng góp của từng cơ chế trong **v2 gốc**, đặc biệt đối với
CVC-300, đồng thời báo cáo đầy đủ cả năm tập. Đây là thí nghiệm kiểm chứng
kiến trúc, chưa phải kết quả chứng minh model vượt EDANet.

## Kế hoạch đã chốt

1. Kiểm tra bản control trùng v2: trọng số, đầu ra, RNG sau khởi tạo, loss.
2. Train **6 cấu hình × seed 42 = 6 runs**, mỗi run 100 epochs. Đây là kết quả
   sơ bộ để kiểm tra quy trình, chưa đủ đo biến thiên giữa các lần train.
3. Bổ sung **hai seed 3407 và 1234 cho tất cả sáu cấu hình**, tổng cộng 18 runs.
   Chốt trước toàn bộ matrix; không chỉ chạy thêm seed cho biến thể đang thắng.
4. Tổng hợp mean ± sample SD, chênh lệch so với full trên cùng seed, năm tập
   và hiệu năng phần cứng. CVC-300 là phân tích trọng tâm; không bỏ các tập giảm.
5. Dùng kết quả để quyết định thành phần nào có bằng chứng hỗ trợ. Nếu cần
   nghiên cứu tương tác giữa hai thành phần, lập study tiếp theo với matrix mới.

Kiến trúc, loss, train/evaluation của `pid_unet_v2` gốc được giữ nguyên. Các
model ablation có tên và checkpoint riêng. Tất cả train từ đầu, không pretrain.

Khi tích hợp, GitHub đã có bản cập nhật cho `pid_unet` không đánh số phiên bản:
bottleneck đổi sang (1,2,4,8) và thêm nhánh global pooling. Vì v2 kế thừa lớp cha
này, study thêm chế độ tương thích `legacy_v2=True` để giữ bottleneck v2
(1,3,6,12), cấu trúc checkpoint và khởi tạo cũ. `pid_unet` mới và các model
đối thủ vừa được thêm vẫn được giữ. Đã đối chiếu riêng cả bốn tổ hợp tùy chọn
constructor của v2 với snapshot trước khi tích hợp, không chỉ so control với
một v2 đã vô tình bị thay đổi.

## Sáu cấu hình

| Mã | Model dùng trong train.py | Thay đổi duy nhất so với full | Tham số |
|---|---|---|---:|
| full | `pid_unet_v2_ablation_full` | Control tương đương v2 gốc | 835,373 |
| no_global | `pid_unet_v2_ablation_no_global` | Bỏ phép cộng global context ở bottleneck | 825,941 |
| no_derivative | `pid_unet_v2_ablation_no_derivative` | Đặt đóng góp D trong fusion bằng 0 | 835,373 |
| fixed_gains | `pid_unet_v2_ablation_fixed_gains` | Thay hệ số thích nghi bằng Kp = Ki = Kd = 1/3 | 827,501 |
| no_boundary_gate | `pid_unet_v2_ablation_no_boundary_gate` | Thay multiplier sigmoid(boundary) trong D bằng 1 | 835,373 |
| no_gate_floor | `pid_unet_v2_ablation_no_gate_floor` | Đặt mức sàn attention bằng 0, gate = saliency | 835,369 |

Mọi cấu hình được khởi tạo từ cùng quá trình dựng v2 đầy đủ trước khi thay
thành phần. Với cùng seed, các tensor còn dùng chung có giá trị khởi tạo giống
nhau; việc tạo module thay thế không làm lệch CPU RNG sau khởi tạo.

Công thức fusion đầy đủ, viết gọn:

```text
a = 0.6 * sigmoid(alpha)
g = a + (1 - a) * saliency(I_input)          # a khởi tạo bằng 0.35
D = abs(P - AvgPool3x3(P)) * sigmoid(B) * g
(Kp, Ki, Kd) = softmax(MLP(GAP(I_input)))
Y = Fuse(SE(Kp * P * g + Ki * I + Kd * D)) + I
```

- `no_global` chỉ bỏ module additive GlobalContext, **không bỏ nhánh I**, RFB
  đa tỉ lệ hay residual I.
- `no_derivative` không chuẩn hóa lại Kp/Ki: giữ controller softmax ba đầu ra
  và đặt D bằng 0. Logit Kd vẫn ảnh hưởng mẫu số softmax. Vì vậy đây là phép
  thử đóng góp D trong fusion, không phải một mạng PI có controller hai đầu ra.
- `no_derivative` và `no_boundary_gate` vẫn tính boundary map và giữ loss
  biên. Điều này cô lập ảnh hưởng của đường truyền đặc trưng, không đồng thời
  thay supervision. Không diễn giải hai hàng này là “bỏ toàn bộ boundary branch”.
- `no_gate_floor` dùng saliency cho cả P và D. Không dùng
  `PIDUNetV2(bounded_gates=False)`, vì tùy chọn đó quay lại SoftPIDBAG và đồng
  thời thay nhiều chi tiết hơn riêng mức sàn.
- Một số hàng có số tham số bằng nhau vì là **functional ablation**. Không
  suy diễn bỏ D sẽ làm model nhỏ hơn. Chi phí thực đo nằm trong efficiency.csv.

Các thí nghiệm này kiểm tra PID fusion và hai cơ chế của v2. Chúng chưa tách
riêng SE, asymmetric RFB, downsampler, cũng chưa đo tương tác giữa nhiều cơ chế.
P/I/D ở đây là phép tương tự không gian trên đặc trưng ảnh, không phải bộ điều
khiển có tích phân theo thời gian.

## Điều kiện cố định

- AdamW, lr 0.0003, weight decay 0.0001; 5 epoch warmup, cosine; 100 epochs.
- Batch 8; input 352; stochastic multiscale 0.75/1/1.25; AMP trên CUDA.
  Giữ batch cuối như v2 gốc (`drop_last=False`): 164 cập nhật mỗi epoch.
  Các model ngoài họ v2 vẫn dùng cách xử lý batch cuối của bản cập nhật mới.
- Bốn structure-loss segmentation và **loss biên BCE + Dice hiện có × 0.2**.
  Không thay loss, trọng số hay số head supervision.
- Augmentation hiện có, copy-paste tắt như v2.
- Cùng 1,450 cặp TrainDataset, cùng split cố định seed 42:
  1,305 train / 145 validation. Seed train không đổi split.
- Checkpoint tốt nhất chọn bằng **validation mDice**. Test dùng final head,
  resolution 352, TTA tắt, hậu xử lý và metrics hiện có.
- Test: CVC-300 (60), ClinicDB (62), ColonDB (380), ETIS (196), Kvasir (100).
  Runner kiểm tra số cặp và hash nội dung ảnh/mask của các tập trước khi chạy.

Không dùng điểm CVC-300 để chọn epoch, chọn seed, chỉnh threshold hoặc xóa hàng
không thuận lợi. Các test set đã được xem qua nhiều vòng phát triển trước đó;
paper nên minh bạch quá trình này, thay vì coi chúng là một đánh giá hoàn toàn
chưa từng được quan sát. Study mới giữ nguyên matrix và protocol đã chốt.

Cùng seed không bảo đảm CUDA bitwise giống nhau: v2 hiện dùng cuDNN benchmark
và không bật deterministic algorithms. Study giữ cấu hình đó để không đổi
recipe. Báo cáo nhiều seed và môi trường chạy; độ lệch chuẩn ba seed không tự
chứng minh ý nghĩa thống kê. Không lấy kết quả v2 cũ một seed làm control cho
study ba seed mới.

## Chạy trên Colab

Mở [notebook ablation](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_unet_v2_ablation_colab.ipynb).
Notebook mount Drive, lấy source, kiểm tra GPU và lưu toàn bộ study tại
`/content/drive/MyDrive/pid_v2_ablation_v1`.

Sau khi setup/mount Drive, lệnh cho vòng đầu:

```bash
python run_pid_v2_ablation.py --seeds 42 --output_root /content/drive/MyDrive/pid_v2_ablation_v1
```

Khi bổ sung hai seed còn lại, chạy **cùng thư mục**:

```bash
python run_pid_v2_ablation.py --seeds 42 3407 1234 --output_root /content/drive/MyDrive/pid_v2_ablation_v1
```

Các run đã hoàn tất sẽ được kiểm tra và bỏ qua. Run dở tiếp tục từ `last.pth`;
nếu chưa có last nhưng có best thì dùng best với optimizer/RNG đã lưu. Nếu
train xong mà ngắt giữa benchmark, chạy lại sẽ tiếp tục từ epoch cuối và thực
hiện benchmark. Không cần tự ghép 18 lệnh hay nhập đường dẫn từng checkpoint.

Để xem trước lệnh, không tải dữ liệu hay train:

```bash
python run_pid_v2_ablation.py --seeds 42 --dry_run
```

Có thể chạy riêng một nhóm, ví dụ control và D trước:

```bash
python run_pid_v2_ablation.py --variants full no_derivative --seeds 42 --output_root /content/drive/MyDrive/pid_v2_ablation_v1
```

Sau đó chạy lệnh đầy đủ để hoàn thành matrix. Một runner chạy tuần tự trên
một GPU; không chạy đồng thời nhiều runner ghi vào cùng thư mục study.
Theo log v2 T4 đã cung cấp (~74 giây/epoch), 6 runs có thể mất khoảng 12–15 giờ,
18 runs khoảng 36–45 giờ; đây là ước tính, phụ thuộc GPU và tốc độ đọc/ghi Drive.

## Bảo toàn kết quả và resume

`ablation_study.json` khóa protocol, hash source Python, hash dữ liệu, split
và phiên bản môi trường. Notebook quay lại git revision đã ghi khi resume.
Runner từ chối trộn cấu hình, source, dữ liệu hay môi trường khác vào study;
khôi phục môi trường cũ hoặc chọn output root mới nếu chủ động đổi protocol.
Phiên bản Python/PyTorch do Colab cung cấp có thể thay đổi giữa các phiên.

Mỗi variant có `model_architecture` riêng trong checkpoint. Cả kiến trúc,
seed/run tag, optimizer và RNG được kiểm tra khi resume. Checkpoint cũ của v2
không được dùng để khởi tạo những ablation này. File chỉ chứa raw weights
không đủ để tiếp tục một run nghiên cứu.

Runner chỉ đánh dấu hoàn tất sau khi process thành công, đã đủ epochs,
đủ metrics năm tập, đúng split và có các artifact cần thiết. Artifact đã hoàn
tất được kiểm tra SHA-256 trước khi tái dùng hoặc tổng hợp. Nếu ngắt sau ghi
CSV nhưng trước lưu checkpoint, các dòng epoch vượt checkpoint được sao lưu
rồi loại khỏi log dùng tiếp để tránh trùng epoch.

Giữ nguyên cả thư mục Drive, gồm:

```text
ablation_study.json
ablation/v2abl_<variant>_seed<seed>/launch.json, completed.json
snapshots/v2abl_<variant>_seed<seed>/best.pth, last.pth, periodic checkpoints
logs/v2abl_<variant>_seed<seed>_config.json
logs/v2abl_<variant>_seed<seed>/train_log.csv, training_curves.png
results/v2abl_<variant>_seed<seed>/metrics.csv, efficiency.csv, predicted masks
ablation_report/summary.csv, summary.md, dice_table.tex, runs.csv, efficiency_runs.csv
```

## Tổng hợp và diễn giải

```bash
python summarize_pid_v2_ablation.py --output_root /content/drive/MyDrive/pid_v2_ablation_v1 --seeds 42 3407 1234
```

- `summary.csv`: 10 metrics trên từng tập và Mean; mean, SD (ddof=1), n,
  seed thực có, paired delta và SD của delta so với full.
- `summary.md`: bảng Dice năm tập, bảng chi tiết CVC-300 và danh sách run thiếu.
- `dice_table.tex`: bảng LaTeX cho paper; ô chưa có dữ liệu là `--`.
- `runs.csv`: best epoch, validation Dice, epoch cuối mỗi run.
- `efficiency_runs.csv`: số liệu hiệu năng nguyên bản theo từng run/hardware.

Một seed không có ước lượng SD, không ghi giả `±0`. Delta chỉ được tính khi
cả variant và full đủ **toàn bộ seed được yêu cầu**. Mean là trung bình không
trọng số của năm tập trong từng run, rồi mới tổng hợp qua các seed. Không gộp
tất cả ảnh năm tập để thay đổi trọng số dataset.

Delta = variant − full: Dice/IoU/Precision/Recall/Specificity/wFb/Sm/Em cao hơn
là tốt; HD95/MAE thấp hơn là tốt. Với CVC-300, đọc Precision cùng Recall để
phân biệt giảm false positive với giảm khả năng phát hiện polyp. Chênh lệch
nhỏ hơn biến thiên giữa seed cần được diễn giải thận trọng.

Profiler hiện có dùng quy ước THOP ở cột `FLOPs (G)`; không tự quy đổi sang
một quy ước FLOPs khác khi so sánh. File size checkpoint gồm optimizer/RNG,
không chỉ trọng số triển khai. So sánh latency/FPS trên cùng GPU và settings.

## Trạng thái triển khai

Đã có sáu model, runner, checkpoint guards, báo cáo và notebook. Kiểm thử CPU
bao gồm đối chiếu control với v2 gốc, khởi tạo chung/RNG, công thức ablation,
forward ở các resolution multiscale, backward và bước cập nhật qua Trainer
hiện có, checkpoint khác biến thể và thống kê dữ liệu thiếu/nhiều seed.

Máy phát triển hiện không có CUDA. Chưa chạy 6/18 thí nghiệm 100 epochs, chưa
có kết quả Dice mới để điền vào bảng. Kiểm thử phần mềm không thay thế việc
train và đánh giá trên Colab.
