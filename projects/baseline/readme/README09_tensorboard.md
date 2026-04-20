# 1、Linux在环境`vcr`中启动TensorBoard
注意，这里要到对应路径下方
```bash

tensorboard   --logdir "$(pwd)/runs/"  --port 6006   --bind_all   --reload_interval 5   --load_fast=false

```


# 2、Windows中启动监听
```markdown
ssh -p 8800 -N -L 16006:127.0.0.1:6006 zhangwei@172.18.232.201
```

# 3、打开网址
```markdown

http://127.0.0.1:16006/?darkMode=true#timeseries&runSelectionState=eyJiYXNlX21lYW5fa3MyMC9tc3J2dHRfYmFzZV9rczIwXzIwMjUwOTI1XzIxNTgzMi90YiI6dHJ1ZSwiYmFzZV9tZWFuX2tzMTIvbXNydnR0X2Jhc2Vfa3MxMl8yMDI1MDkyNV8yMzEwMDcvdGIiOnRydWUsImJhc2VfbWVhbl9sZW5fY29udHJvbF9rczEyL21zcnZ0dF9iYXNlX21lYW5fbGVuX2NvbnRyb2xfa3MxMl8yMDI1MDkyNl8xNDQzNDEvdGIiOnRydWUsImJhc2VfbWVhbl9STF9rczEyL21zcnZ0dF9iYXNlX21lYW5fUkxfa3MxMl8yMDI1MDkyN18xNzUxMDYvdGIiOnRydWUsImJhc2VfbWVhbl9STF9rczEyL21zcnZ0dF9iYXNlX21lYW5fUkxfa3MxMl8yMDI1MDkyN18xNzQ1NDIvdGIiOnRydWV9
```