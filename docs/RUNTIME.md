# 冻结渲染运行时

`strict_compositor/render.py` 加载随仓库保留的
`strict_compositor/_render_runtime.cpython-313.pyc`。
历史开发中，渲染源文件被中断的外部进程截断，编译后的实现仍然保留。
当前 facade 重导出该实现，并提供可读的 `contact_profile` 修补。

当前运行要求是 **CPython 3.13**。不能用 Python 3.12 或 3.14 直接导入；
更不能删除此文件后将仓库描述为仍可运行的完整渲染系统。
它也不能替代完整源码审阅。源码恢复与跨版本迁移属于后续工作。

发布时对 33 个 code object 的 `co_filename` 做了递归替换，统一为
`strict_compositor/render_runtime.py`，清理历史机器路径。
执行指令、非 code 常量与原 16 字节 pyc 头保持不变；没有反编译或改写算法。
该相对路径是堆栈元数据，不表示仓库提供了同名完整源码文件。

```text
Original SHA-256:
251e49b210907b9b0ca6c248fc281ad6e24f49675f47626248b01f8f376b5ef3

Release SHA-256:
f36657166f34db5804fd37ef434f31911e64561165b822daf07c4a68231fab64
```

处理脚本是 `tools/normalize_runtime_metadata.py`。运行时的最终完整性由上述
发布哈希确认；它被 `.gitignore` 明确列为唯一允许提交的 `.pyc`。

对应模块的测试覆盖合成、Alpha、投影、反射、身份恢复等行为；
这些测试并不能证明冻结实现的全部内部逻辑已被审阅。
