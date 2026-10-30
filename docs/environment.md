# 用户环境约定

Windows 长路径修复（2026-10-11）：TensorBoard 写入与浏览入口使用原生扩展绝对路径，避免其内部 bytes 文件名在 260 字符边界失败。新实验目录使用短名称前缀与完整名称 SHA 前缀，完整实验名称保留在 config.json，旧实验目录保持原样。无需修改 Windows 注册表或重新安装 CUDA/依赖。

用户于 2026-10-09 明确指定默认 Python：

`C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe`（3.9.13）。

后续工作优先基于该解释器创建隔离环境。不要因为终端 PATH 默认返回 `D:\zotero\pyy\python.exe`（3.12.10）就把它作为用户默认实验环境，也不要修改全局 PATH 或默认解释器。

本项目实验使用 `.venv-came39\Scripts\python.exe`。安装脚本 `scripts/bootstrap.ps1` 显式使用指定解释器，优先安装 `requirements/phase_02_windows-cu118.lock.txt`。用户原 `.venv` 保留；未使用的 `.venv-came`（3.11）已按空间要求清理，见 `artifacts/audit/unused_runtime_cleanup.json`。

版本锁定及安装后冻结列表位于 `requirements/` 和 `artifacts/audit/`。实际 CUDA 可用性以 `scripts/check_environment.py` 的图运算和反向测试为准。

PyCharm 编号入口 `scripts/run_*.py` 支持由用户基础 Python39 直接启动：在导入实验依赖前显式打印并转交到现有 `.venv-came39`，保留命令参数及退出码。科学计算仍只在项目环境执行，不改全局默认环境，也不允许 CPU fallback。内部 `-m came_repro.*` 命令仍须指定项目解释器。
