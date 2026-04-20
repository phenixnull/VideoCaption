覆盖从 `JDK 8` 到成功跑通 **SPICE / ROUGE\_L / METEOR / BLEU / CIDEr** 的全流程。
>用JDK8主要是SPICE的库是在当时JDK8的时候支持的，所以高版本JDK可能会不兼容

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
wget -c https://mirrors.tuna.tsinghua.edu.cn/Adoptium/8/jdk/x64/linux/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
# 解压（将得到 jdk8u462-b08/）
tar -xzf OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz


# 临时验证（当前会话有效）
export JAVA_HOME=$HOME/jdk8u462-b08
export PATH=$JAVA_HOME/bin:$PATH
java -version
```

若输出类似（版本号无所谓，能出 11 即可）：


>openjdk version "1.8.0_462" ...


可选：写入 `~/.bashrc` 以长期生效：

```bash
echo 'export JAVA_HOME=$HOME/jdk8u462-b08' >> ~/.bashrc
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
<mark>这里没有pycocoevlcal是因为要离线安装[下面步骤]

离线安装 `pycocoevalcap`（在 **Windows** 下载 zip 再上传）：

* Windows 下载：[https://github.com/salaniz/pycocoevalcap/archive/refs/heads/master.zip](https://github.com/salaniz/pycocoevalcap/archive/refs/heads/master.zip)
* 上传到服务器，例如：
  `/mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/pycocoevalcap-master.zip`
* 如果服务器能直接魔法可以直接用下面这个步骤安装
```bash
wget -c https://github.com/salaniz/pycocoevalcap/archive/refs/heads/master.zip
```
在服务器上解压并，<mark>进入对应目录</mark>，进行离线安装：

<mark>注意：离线安装这个时，一定要先激活对应的环境，然后进入对应目录离线安装</mark>

```bash
unzip pycocoevalcap-master.zip -d temp/
cd temp/pycocoevalcap-master/
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## C. 放置 **Stanford CoreNLP 3.6.0** 资源（离线 / 无网）






1. 在 **Windows** 下载官方包（即 3.6.0 版本）：

* [https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip](https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)

2. 上传到你环境中 **`pycocoevalcap/spice/`** 的实际目录（以下是你的真实路径）：

<mark>注意</mark>这一步，一定要在安装完pycocoevalcap成功后，再进行这个步骤，这样目录才会存在

```markdown
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/
```
---
<mark>Tips：如何找自己的对应环境目录?</mark>
```bash
conda env list
```
![img_1.png](img_1.png)


---
上传后，文件实际存在为：

```
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/stanford-corenlp-full-2015-12-09.zip
```
我这里用的是`FileZilla`
![img_2.png](img_2.png)

3. 改名为 `pycocoevalcap` 期望的文件名，并**手动解压**到期望目录名：

```bash
SPICE_DIR="/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"

# 改名为固定识别名
mv "$SPICE_DIR/stanford-corenlp-full-2015-12-09.zip" "$SPICE_DIR/stanford-corenlp-3.6.0.zip"
ls -lh "$SPICE_DIR/stanford-corenlp-3.6.0.zip"
```

<mark>注意：这里的`SPICE_DIR`要改成自己的路径\
以下代码临时脚本，直接粘贴进**bash**
```bash
# 解压并将顶层目录改名为 pycocoevalcap 期望的目录名
python - <<'PY'
import os, zipfile, shutil
SPICE_DIR="/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"
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

<mark>注意：这里的`p`要改成自己的路径\
以下代码临时脚本，直接粘贴进**bash**
```bash
python - <<'PY'
import io, os
p = "/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/get_stanford_models.py"
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

<mark>注意，以下为测试临时脚本，直接粘贴进**bash**</mark>

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
![img_3.png](img_3.png)
看到类似指标测试数值即 OK（与测试一致）。

---

## F. **SPICE 自检**（采用 classpath 的稳定方式）

<mark>注意：这里的`SPICE_DIR`要改成自己的路径\
以下代码临时脚本，直接粘贴进**bash**

```markdown
SPICE_DIR="/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"
mkdir -p "$SPICE_DIR/cache"
chmod u+w "$SPICE_DIR/cache"

export JAVA=~/jdk8u462-b08/bin/java
mkdir -p ~/.cache/spice

```

> 原版 `pycocoevalcap.spice.Spice` 通过 `java -jar` 调用，有时缺少 CoreNLP 依赖会 `ClassNotFound`。
> 这里用 `-cp` **显式加入** CoreNLP 依赖，稳定可用。

<mark>注意：这里的`SPICE_DIR`要改成自己的路径\
以下代码临时脚本，直接粘贴进**bash**

```bash
python - <<'PY'
import os, json, subprocess, tempfile, statistics
SPICE_DIR="/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice"

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
SPICE overall (mean F1): 0.5984848484848484

```
![img_4.png](img_4.png)
则 SPICE 正常。

---

## G.（可选）一次性打印 5 项指标的极简封装

> 若你只想“确认能跑”，做到 F 即可。若想**一条命令**就打印 5 项分数，可用下段封装（SPICE 仍走 `-cp`）。

<mark>以下临时脚本在粘贴进**bash**</mark>

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
gts = {#这个是Ground Truth，句子条目不定长度
    '1': ['a man riding a horse on the beach', 'a person riding a horse by the sea'],
    '2': ['a group of people are sitting at a table', 'several people sit around a dining table'],
}
res = {#这个是推理的句子，句子条目为单条
    '1': ['a man riding a horse on the beach'],
    '2': ['people sitting at a table'],
}
scores = eval_all(gts, res)
for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR","ROUGE_L","CIDEr","SPICE"]:
    print(f"{k}: {scores[k]}")
PY
```

---

好的，我把你前面问到的“长句/大批量内存设置、分批评测、环境变量注入、Java 版本”等都补进 **H** 段了——直接把下面这段覆盖你文档里的 **“H. 常见坑位与说明”** 即可：

---

## H. 常见坑位与说明

* **为什么要自己解压 & 改目录名？**
  `pycocoevalcap` 的 `get_stanford_models()` 期望固定的 zip 名/目录名（`stanford-corenlp-3.6.0`）。在无外网/仅国内镜像时，自动下载常失败或生成坏 zip。手动放置 + 改名 + 早退补丁最稳。

* **为什么 SPICE 要用 `-cp` 而不是 `-jar`？**
  `java -jar` 下 `spice-1.0.jar` 有时找不到 CoreNLP 的类（`NoClassDefFoundError`）。用 `-cp` 显式把 `spice-1.0.jar:stanford-corenlp-3.6.0/*` 放进 classpath，能 100% 把依赖补全，最省心。

* **Java 版本要点（强烈建议 Java 8）**
  SPICE 这套老依赖会反射访问 `java.lang` 私有字段；Java 9+ 的模块系统默认禁止，典型报错是 `InaccessibleObjectException`。

  * **推荐**：用 **Java 8** 运行（例如 `~/jdk8u462-b08/bin/java` 或 `conda install -c conda-forge openjdk=8`）。
  * 如必须用 Java 17+，需加一串 `--add-opens`（不如直接用 Java 8 稳定）。

* **缓存目录必须存在且可写**
  `-cache` 指向的路径若不存在/不可写，会报 `org.fusesource.lmdbjni.LMDBException: No such file or directory`。

  * 建议用你有写权限的位置：如 `~/.cache/spice`，先 `mkdir -p ~/.cache/spice`。

* **内存不足（长句/大批量）如何处理？**
  CoreNLP 解析吃内存，句子更长/数量更多就更费内存。

  1. **直接调大堆内存**：把 `-Xmx4G` 改成需要的大小，例如 8G 或 12G：

     ```bash
     ... JAVA -Xmx8G -cp ... edu.anu.spice.SpiceScorer ...
     ```

     也可以同时设置初始堆：`-Xms2G -Xmx8G` 以减少扩容开销。
  2. **用环境变量不改其它代码**：

     ```python
     env = dict(os.environ)
     env["_JAVA_OPTIONS"] = "-Xms2G -Xmx8G"  # 或者用 JAVA_TOOL_OPTIONS
     subprocess.check_call(cmd, cwd=SPICE_DIR, env=env)
     ```

     > 命令行里的 `-Xmx` 和 `_JAVA_OPTIONS` 并存时，以命令行参数为准。
  3. **分批评测（推荐与加内存同时用）**：把大集合拆成若干批（如 100–300/批），每批跑完再合并均值，既稳又节省内存。

* **推荐的内存/批大小经验值**

  * 小批/短句：`-Xmx2G ~ 4G`
  * 中等规模：`-Xmx6G ~ 8G`
  * 大批/长句：`-Xmx12G+`，并将 `batch_size` 调小（如 100–200）

* **确保用的是 Java 8 可执行文件**
  路径写死最省心：`JAVA=~/jdk8u462-b08/bin/java`，或将其放进 `PATH`。不要混用系统自带的 Java 17。

---


## I. 关键路径总览（以本案例机器为准）

* Conda env：`vcr`（`Python 3.8.20`）
* `pycocoevalcap` 安装位置：

```
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/  ```
```

* SPICE 代码与资源目录：

```
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/stanford-corenlp-3.6.0
/home/zhouniu/anaconda3/envs/vcr/lib/python3.8/site-packages/pycocoevalcap/spice/spice-1.0.jar
```

---

