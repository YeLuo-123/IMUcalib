# IMU 外参标定程序 for 四轮车 (ROS2)

这是一个用于四轮车（仅平面运动）的 IMU 标定程序。它可以标定 IMU 相对于机器人中心 `base_link` ）的角度偏差（`yaw`、`roll`、`pitch`）。程序通过让机器人执行直线运动，结合 IMU、里程计，利用非线性优化求解安装参数。

## 功能特点

- **静态标定**：通过静止时的加速度数据计算 IMU 的 roll 和 pitch 安装角度。
- **直线标定**：让机器人直线运动，主要标定 yaw 角度偏差。
- **输出结果**：生成 YAML 格式的标定文件，并发布标定后的 TF 变换。

## 依赖

- ROS2 (Humble/Iron/Jazzy)
- Python 3.8+
- Python 包：`numpy`, `scipy`, `matplotlib`, `pyyaml`, `tf_transformations`
- 机器人需发布以下话题：
  - `/imu/data` (sensor_msgs/Imu)
  - `/odom` (nav_msgs/Odometry) —— 里程计原点应为机器人中心

安装依赖：
```bash
sudo apt install ros-${ROS_DISTRO}-tf-transformations
pip3 install numpy scipy matplotlib pyyaml
```

## 文件结构

```
imu_calibration/
├── imu_calibration
│   └── imu_calibrator_4wd.py   # 主程序
├── launch
│   └── imu_calibration_launch.py  # (可选) launch文件
├── config
│   └── params.yaml             # (可选) 参数文件
└── README.md
```

## 使用方法

### 1. 准备工作

- 将机器人置于开阔平坦区域。
- 确保所有话题正常发布。

### 2. 运行标定节点

```bash
python3 imu_calibrator_4wd.py
```

### 3. 交互式控制

程序启动后显示如下控制指令：

```
控制指令:
  t - 静态标定
  s - 直线标定
  r - 旋转标定
  p - 绘图
  q - 退出
```

**建议顺序**：
1. 按 `t` 进行静态标定（机器人静止，采集约5秒数据）。
2. 按 `s` 进行直线标定（机器人前进指定距离，默认2米）。
3. 按 `r` 进行旋转标定（机器人原地旋转指定角度，默认360度）。
4. 程序自动完成直线和旋转标定后，会进行联合优化并输出最终结果。
5. 按 `p` 可绘制轨迹、IMU 角度。

## 输出结果

标定结果保存在 `save_path` 目录下，包含两个文件：

- `imu_static_时间戳.yaml`：静态标定结果，包括 roll、pitch 偏移。
- `imu_calib_时间戳.yaml`：联合标定最终结果，包括最终安装参数（`final_dyaw`, ...）和相对于 URDF 的修正量（`correction_*`）。

控制台会打印最终安装参数（例如）：
```
优化结果（最终安装参数）:
  dyaw: 0.3245 rad (18.59°)
  droll: 0.3001 rad (17.19°)
  dpitch: 0.1000 rad (5.73°)
```

同时程序会发布 TF 变换 `base_link` → `imu_link`，使用最终的安装参数。

