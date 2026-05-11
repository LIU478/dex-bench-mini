# Day 1: Isaac Lab 环境搭建完成 + Cartpole 训练验证

## 完成的事
- AutoDL 实例开通(RTX 4090D 24GB,Miniconda + CUDA 11.8)
- PyTorch 2.5.1 + CUDA 12.1 安装
- 系统图形库安装(libSM/libXt 等 30+ 个库)
- Isaac Sim 4.5 安装(pip 方式)
- Isaac Lab v2.1.0 安装(5 个核心模块全部装上)
- pip 源切换到清华(解决 aliyun 缺包问题)
- numpy 降级到 1.26(满足 Isaac Sim < 2.0 要求)
- setuptools 降级到 69.5.1(兼容 flatdict 老库)
- flatdict 用 --no-build-isolation 装上

## 里程碑
**Cartpole RL 训练 100 次迭代完美成功**:
- Mean episode length: 288.83(接近最大值 300)
- Episode_Reward/alive: 1.0000
- Episode_Termination/cart_out_of_bounds: 0.0000
- 训练时间:35 秒(RTX 4090D)
- 计算性能:2880 steps/s

## 踩的坑及解法
1. AutoDL 默认 pip 源 aliyun 缺 setuptools → 改清华源
2. PyTorch 默认装到 2.11(错误版本) → 手动指定 2.5.1+cu121
3. Isaac Sim 启动时 isaacsim.asset.importer.urdf segfault → 用 isaaclab.sh 包装而非直接调 SimulationApp
4. Isaac Lab 核心模块 flatdict 因 pkg_resources 缺失失败 → setuptools 降级到 69.5.1 + --no-build-isolation
5. numpy 必须 < 2.0(isaacsim-core 4.5.0.0 强制要求)→ 装 numpy 1.26.4

## 下一步
- 保存 AutoDL 自定义镜像
- 回到 Phase 0:Python + PyTorch 学习(Day 6)
