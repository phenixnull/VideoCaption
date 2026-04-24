下面是一份**可直接照做、不再试错**的 README，完全按照你现在这台机子的实际路径与网络/权限条件整理，覆盖从 JDK 11 到成功跑通 **SPICE / ROUGE\_L / METEOR / BLEU / CIDEr** 的全流程。你可以直接把整段内容保存成 `README_pycocoevalcap_SPICE.md` 使用。

---

# 在无 sudo、仅国内源/离线条件下，一次性跑通 BLEU / METEOR / ROUGE\_L / CIDEr / SPICE

> 运行环境：
>
> * Linux x86\_64（Ubuntu 22.04）
> * Python 3.8.20（conda 环境名：`vcr`）
> * 无 sudo，仅能访问清华/国内镜像；Windows 可下载后上传到服务器
> * 目标：在同一环境下稳定评测 **BLEU / METEOR / ROUGE\_L / CIDEr / SPICE**

---

## A. 安装并配置 JDK 11（一次性）

> 如你已完成并 `java -version` 正常，可跳过本节。

```bash
# 下载 JDK 11（清华镜像）
cd ~
wget -c https://mirrors.tuna.tsinghua.edu.cn/Adoptium/11/jdk/x64/linux/OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz

# 解压（将得到 jdk-11.0.28+6/）
tar -xzf OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz

# 临时验证（当前会话有效）
export JAVA_HOME=$HOME/jdk-11.0.28+6
export PATH=$JAVA_HOME/bin:$PATH
java -version
```

若输出类似（版本号无所谓，能出 11 即可）：

```
openjdk version "11.0.28" ...
```

可选：写入 `~/.bashrc` 以长期生效：

```bash
echo 'export JAVA_HOME=$HOME/jdk-11.0.28+6' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## B. 安装 Python 依赖与 `pycocoevalcap`

进入你的 `vcr` 环境，并通过清华源安装依赖：

```bash
conda activate vcr
python -m pip install --upgrade pip
pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

离线安装 `pycocoevalcap`（在 **Windows** 下载 zip 再上传）：

* Windows 下载：[https://github.com/salaniz/pycocoevalcap/archive/refs/heads/master.zip](https://github.com/salaniz/pycocoevalcap/archive/refs/heads/master.zip)
* 上传到服务器，例如：
  `/mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/pycocoevalcap-master.zip`

在服务器上解压并安装：

```bash
unzip /mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/pycocoevalcap-master.zip \
  -d /mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/temp/

cd /mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/temp/pycocoevalcap-master/
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## C. 放置 **Stanford CoreNLP 3.6.0** 资源（离线 / 无网）

1. 在 **Windows** 下载官方包（即 3.6.0 版本）：

* [https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip](https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)

2. 上传到你环境中 **`pycocoevalcap/spice/`** 的实际目录（以下是你的真实路径）：

```
/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/
```

上传后，文件实际存在为：

```
/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/stanford-corenlp-full-2015-12-09.zip
```

3. 改名为 `pycocoevalcap` 期望的文件名，并**手动解压**到期望目录名：

```bash
SPICE_DIR="/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"

# 改名为固定识别名
mv "$SPICE_DIR/stanford-corenlp-full-2015-12-09.zip" "$SPICE_DIR/stanford-corenlp-3.6.0.zip"
ls -lh "$SPICE_DIR/stanford-corenlp-3.6.0.zip"

# 解压并将顶层目录改名为 pycocoevalcap 期望的目录名
python - <<'PY'
import os, zipfile, shutil
SPICE_DIR = "/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"
zip_path = os.path.join(SPICE_DIR, "stanford-corenlp-3.6.0.zip")
assert os.path.exists(zip_path), "zip not found"

with zipfile.ZipFile(zip_path) as z:
    z.extractall(SPICE_DIR)

src = os.path.join(SPICE_DIR, "stanford-corenlp-full-2015-12-09")
dst = os.path.join(SPICE_DIR, "stanford-corenlp-3.6.0")
if os.path.exists(dst):
    shutil.rmtree(dst)
os.rename(src, dst)

print("Prepared:", dst)
print("Samples:", sorted(os.listdir(dst))[:8])
PY
```

---

## D. 给 `get_stanford_models.py` 加“已就绪就早退”的安全补丁（防止它误判还去下载）

> 仅在 **`stanford-corenlp-3.6.0/`** 已就位的前提下进行（上一步已完成）。此补丁**不破坏原逻辑**，只是让它在目录已存在时直接 `return`，避免再次尝试下载导致坏 zip。

```bash
python - <<'PY'
import io, os
p = "/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/get_stanford_models.py"
with io.open(p, "r", encoding="utf-8") as f:
    s = f.read()
if "EARLY_GUARD_INSERTED" in s:
    print("Already patched")
else:
    head = "def get_stanford_models():"
    i = s.find(head)
    assert i != -1, "Function header not found"
    j = s.find("\n", i)
    assert j != -1, "No newline after header"
    guard = (
        "    # EARLY_GUARD_INSERTED: skip download/extract if the folder already exists\n"
        "    import os\n"
        "    target_dir = os.path.join(os.path.dirname(__file__), 'stanford-corenlp-3.6.0')\n"
        "    if os.path.isdir(target_dir):\n"
        "        return\n"
    )
    with io.open(p + ".bak2", "w", encoding="utf-8") as b:
        b.write(s)
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(s[:j+1] + guard + s[j+1:])
    print("Patched safely:", p)
PY
```

---

## E. 快速自检（BLEU / METEOR / ROUGE\_L / CIDEr）——Python 内置评测

```bash
python - <<'PY'
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

gts = {
    '1': ['a man riding a horse on the beach', 'a person riding a horse by the sea'],
    '2': ['a group of people are sitting at a table', 'several people sit around a dining table'],
}
res = {
    '1': ['a man riding a horse on the beach'],
    '2': ['people sitting at a table'],
}

def run(metric, name):
    score, _ = metric.compute_score(gts, res)
    print(f"{name}: {score}")

print("---- Quick check (BLEU/METEOR/ROUGE_L/CIDEr) ----")
run(Bleu(4), "BLEU-1..4")
run(Meteor(), "METEOR")
run(Rouge(),  "ROUGE_L")
run(Cider(),  "CIDEr")
PY
```

看到类似数值即 OK（与你先前测试一致）。

---

## F. **SPICE 自检**（采用 classpath 的稳定方式）

> 原版 `pycocoevalcap.spice.Spice` 通过 `java -jar` 调用，有时缺少 CoreNLP 依赖会 `ClassNotFound`。
> 这里用 `-cp` **显式加入** CoreNLP 依赖，稳定可用。

```bash
python - <<'PY'
import os, json, subprocess, tempfile, statistics
SPICE_DIR="/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"

gts = {
    '1': ['a man riding a horse on the beach', 'a person riding a horse by the sea'],
    '2': ['a group of people are sitting at a table', 'several people sit around a dining table'],
}
res = {
    '1': ['a man riding a horse on the beach'],
    '2': ['people sitting at a table'],
}

items = [{"image_id": k, "test": res[k][0], "refs": gts[k]} for k in sorted(res.keys())]

with tempfile.TemporaryDirectory() as td:
    in_json  = os.path.join(td, "in.json")
    out_json = os.path.join(td, "out.json")
    with open(in_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    cmd = [
        "java", "-Xmx4G",
        "-cp", f"{SPICE_DIR}/spice-1.0.jar:{SPICE_DIR}/stanford-corenlp-3.6.0/*",
        "edu.anu.spice.SpiceScorer",
        in_json,
        "-cache", os.path.join(SPICE_DIR, "cache"),
        "-out", out_json,
        "-subset",
        "-silent"
    ]
    subprocess.check_call(cmd, cwd=SPICE_DIR)

    with open(out_json, "r", encoding="utf-8") as f:
        lst = json.load(f)  # 注意：输出是 list

scores = [r["scores"]["All"]["f"] for r in lst]
for r in lst:
    print(f"SPICE image_id={r['image_id']}: F1={r['scores']['All']['f']:.4f}")
print("SPICE overall (mean F1):", statistics.mean(scores))
PY
```

若输出类似：

```
SPICE image_id=1: F1=0.8333
SPICE image_id=2: F1=0.3636
SPICE overall (mean F1): 0.59848...
```

则 SPICE 正常。

---

## G.（可选）一次性打印 5 项指标的极简封装

> 若你只想“确认能跑”，做到 F 即可。若想**一条命令**就打印 5 项分数，可用下段封装（SPICE 仍走 `-cp`）。

```bash
python - <<'PY'
import os, json, subprocess, tempfile
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import pycocoevalcap.spice as spice_pkg

def eval_all(gts: dict, res: dict) -> dict:
    ids = sorted(set(gts) & set(res), key=lambda x: str(x))
    gts_f = {i: gts[i] for i in ids}
    res_f = {i: res[i] for i in ids}

    bleu = Bleu(4);     bleu_score, _   = bleu.compute_score(gts_f, res_f)
    meteor = Meteor();  meteor_score, _ = meteor.compute_score(gts_f, res_f)
    rouge = Rouge();    rouge_score, _  = rouge.compute_score(gts_f, res_f)
    cider = Cider();    cider_score, _  = cider.compute_score(gts_f, res_f)

    SPICE_DIR = os.path.dirname(spice_pkg.__file__)
    items = [{"image_id": str(k), "test": res_f[k][0], "refs": gts_f[k]} for k in ids]
    with tempfile.TemporaryDirectory() as td:
        in_json  = os.path.join(td, "in.json")
        out_json = os.path.join(td, "out.json")
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        cmd = ["java","-Xmx4G","-cp",f"{SPICE_DIR}/spice-1.0.jar:{SPICE_DIR}/stanford-corenlp-3.6.0/*",
               "edu.anu.spice.SpiceScorer", in_json, "-cache", os.path.join(SPICE_DIR,"cache"),
               "-out", out_json, "-subset", "-silent"]
        subprocess.check_call(cmd, cwd=SPICE_DIR)
        with open(out_json, "r", encoding="utf-8") as f:
            lst = json.load(f)
    spice_mean = sum(d["scores"]["All"]["f"] for d in lst) / max(1, len(lst))
    return {
        "BLEU-1": float(bleu_score[0]),
        "BLEU-2": float(bleu_score[1]),
        "BLEU-3": float(bleu_score[2]),
        "BLEU-4": float(bleu_score[3]),
        "METEOR": float(meteor_score),
        "ROUGE_L": float(rouge_score),
        "CIDEr":  float(cider_score),
        "SPICE":  float(spice_mean),
    }

# 小样例
gts = {
    '1': ['a man riding a horse on the beach', 'a person riding a horse by the sea'],
    '2': ['a group of people are sitting at a table', 'several people sit around a dining table'],
}
res = {
    '1': ['a man riding a horse on the beach'],
    '2': ['people sitting at a table'],
}
scores = eval_all(gts, res)
for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR","ROUGE_L","CIDEr","SPICE"]:
    print(f"{k}: {scores[k]}")
PY
```

---

## H. 常见坑位与说明

* **为什么要自己解压 & 改目录名？**
  `pycocoevalcap` 的 `get_stanford_models()` 期望固定的 zip 名/目录名（`stanford-corenlp-3.6.0`）。在无外网/仅国内镜像时，自动下载常失败或生成坏 zip。手动放置 + 改名 + 早退补丁最稳。

* **为什么 SPICE 要用 `-cp` 而不是 `-jar`？**
  `java -jar` 下 `spice-1.0.jar` 有时找不到 CoreNLP 的类（`NoClassDefFoundError`）。用 `-cp` 显式把
  `spice-1.0.jar:stanford-corenlp-3.6.0/*` 放进 classpath，能 100% 把依赖补全，最省心。

* **内存不足**
  默认 `-Xmx4G`。如内存紧张，改小到 `-Xmx2G` 也可正常跑通小批量测试。

---

## I. 关键路径总览（按你的机器）

* Conda env：`vcr`（Python 3.8.20）
* `pycocoevalcap` 安装位置：

  ```
  /mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap
  ```
* SPICE 代码与资源目录：

  ```
  /mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice
  /mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/stanford-corenlp-3.6.0
  /mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/spice-1.0.jar
  ```

---

> 本 README 是在你给出的原始记录基础上，清理掉试错步骤后整理而成，命令与路径均已替换为你机器上的**实参**。&#x20;

如果你要，我也可以把上面内容**另存为 Markdown 文件**并发你下载。
