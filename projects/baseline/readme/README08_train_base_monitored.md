
# 先执行train命令还是先SSH
两种先后都行，但**推荐顺序**是：先把可视化通道打通→再开训，这样你能从第一个 step 就看到曲线。

**建议顺序（稳妥版）**

1. **服务器上开 TensorBoard**（注意你的 `out_dir` 改成了 `./runs/base_mean_ks20`，别再用 `runs/base_mean` 了）：

   ```bash
   tensorboard --logdir ./runs/base_mean_ks20 --port 6006 --bind_all
   # 想一次看所有实验也可以：tensorboard --logdir ./runs --port 6006 --bind_all
   ```

   > 脚本会把日志写到：`./runs/base_mean_ks20/<数据集_模型_时间戳>/tb/`，让 TensorBoard 盯住父目录即可。

2. **本地 Windows 开 SSH 转发**：

   ```bash
   ssh -N -L 16006:127.0.0.1:6006 <user>@<remote_host>
   # 浏览器打开 http://127.0.0.1:16006
   ```

3. **服务器上启动训练（DDP 4 卡）**：

   ```bash
   torchrun --nproc_per_node=4 train_base_mean_monitored.py \
     --ddp 1 --gpus 0,1,2,3 \
     --dataset_type msrvtt \
     --clip_global_vision_feats_path ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks20_features.pickle \
     --annotations_path ../datasets/MSRVTT/MSRVTT_data.json \
     --out_dir ./runs/base_mean_ks20 \
     --batch_size 128
   ```

   > 也可以先开训、后开 TensorBoard，TensorBoard 会自动加载新写入的日志文件。

小提示：

* 如果 6006 已被占用，改用 `--port 6007`，本地把 `-L 16007:127.0.0.1:6007` 对应改一下即可。
* 服务器端建议用 `tmux/screen` 开 TensorBoard 和训练，避免会话断开。

要不要我把 TensorBoard 的 `logdir` 直接设成 `./runs`，这样你能对比不同 ks 的多次实验在同一面板里？
