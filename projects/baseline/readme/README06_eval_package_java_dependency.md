# 安装Java JDK8
最好linux是x64的，这里的环境是Ununtu 22.04(linux x64_64)的

<mark>现在下载JDK8</mark>
```bash
cd ~
wget -c https://mirrors.tuna.tsinghua.edu.cn/Adoptium/8/jdk/x64/linux/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
```
成功则结果如下类似：

>(vcr)zhouniu@ubuntu:~$ wget -c https://mirrors.tuna.tsinghua.edu.cn/Adoptium/8/jdk/x64/linux/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
--2025-09-24 11:10:31--  https://mirrors.tuna.tsinghua.edu.cn/Adoptium/8/jdk/x64/linux/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
Resolving mirrors.tuna.tsinghua.edu.cn (mirrors.tuna.tsinghua.edu.cn)... 2402:f000:1:400::2, 101.6.15.130
Connecting to mirrors.tuna.tsinghua.edu.cn (mirrors.tuna.tsinghua.edu.cn)|2402:f000:1:400::2|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 103087414 (98M) [application/octet-stream]
Saving to: ‘OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz’
OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz     100%[=================================================================================================================>]  98.31M  71.7MB/s    in 1.4s
2025-09-24 11:10:33 (71.7 MB/s) - ‘OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz’ saved [103087414/103087414]


然后解压，以后有一个 `jdk8u462-b08/`
```bash
tar -xzf OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
```
验证
![img.png](img.png)

```bash
cd jdk8u462-b08/
```
确认里面有
`bin/java `和
`bin/javac`

临时测试（当前会话有效）
```bash
export JAVA_HOME=$HOME/jdk8u462-b08
export PATH=$JAVA_HOME/bin:$PATH
```
然后命令
```bash
java -version
```
如果显示如下，说明安装成功
>(vcr) zhouniu@ubuntu:~/jdk8u462-b08$ java -version
openjdk version "1.8.0_462"
OpenJDK Runtime Environment (Temurin)(build 1.8.0_462-b08)
OpenJDK 64-Bit Server VM (Temurin)(build 25.462-b08, mixed mode)

接下来永久生效,把下面两行加到` ~/.bashrc `末尾：
```bash
export JAVA_HOME=$HOME/jdk8u462-b08
export PATH=$JAVA_HOME/bin:$PATH
```
然后刷新
```bash
source ~/.bashrc
```

到这里Java JDK8已经安装成功
