看一下project下方的redcord_logs里面的plant_RL_caption，然后单独所有的这个方案都在rl_caption这个文件下方进行，数据集类可以参考Dataloaders里面的定义然后把新的数据集类单独整理在rl_caption里面，
  总之这个rl_caption外部的数据是可以直接用的，但是代码都在这个下方进行，包括代码撰写，实验测试，环境一定是vcr这个环境哦，必须用conda激活，然后必须在这个额环境下进行哦。然后外部的代码数据
  都不要改，可以增添适配新的预处理的数据提高数据加载和模型运算的效率，但是必须放在rl_captionn下面。要进行方案设计，专业参考，代码撰写，实验运行，实验结果记录，以及实验结果分析。然后日志写
  在/mnt/sda/Disk_D/zhangwei/projects/VC/project/record_logs这里面。用md的形式，测试实验要完整，目前就在MSVD上做实验吧，要有结果，实验数据分析，以及tensorborad记录【风格参考train_base_mean_monitor
  ed】，日志记录。
