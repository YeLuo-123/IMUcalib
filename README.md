# IMU 外参标定程序 for 四轮车 (ROS2)

这是一个用于四轮车（仅平面运动）的 IMU 标定程序。它可以标定 IMU 相对于机器人中心 `base_link` 的安装位置偏差（`dx`, `dy`）和角度偏差（`yaw`、`roll`、`pitch`）。程序通过让机器人执行特定的运动（直线、旋转），结合 IMU、里程计和激光雷达数据（无需其他传感器），利用非线性优化求解安装参数。

## 功能特点

- **静态标定**：通过静止时的加速度数据计算 IMU 的 roll 和 pitch 安装角度。
- **直线标定**：让机器人直线运动，主要标定 yaw 角度偏差。
- **旋转标定**：让机器人原地旋转，通过向心加速度效应标定 `dx`、`dy` 平移偏差。
- **联合优化**：将直线和旋转数据合并，同时考虑里程计、IMU 和激光雷达（可选）信息，提高标定精度。
- **输出结果**：生成 YAML 格式的标定文件，并发布标定后的 TF 变换。
- **支持激光雷达**：利用墙面特征辅助标定（可选）。

## 依赖

- ROS2 (Humble/Iron/Jazzy)
- Python 3.8+
- Python 包：`numpy`, `scipy`, `matplotlib`, `pyyaml`, `tf_transformations`
- 机器人需发布以下话题：
  - `/imu/data` (sensor_msgs/Imu)
  - `/odom` (nav_msgs/Odometry) —— 里程计原点应为机器人中心
  - `/scan` (sensor_msgs/LaserScan) —— 若使用激光雷达

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

- 将机器人置于开阔平坦区域，周围最好有垂直于机器人前进方向的墙面（若使用激光雷达）。
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
5. 按 `p` 可绘制轨迹、IMU 角度和激光特征（若有）。

## 参数说明

可在启动时通过 `--ros-args -p` 设置参数，或修改代码中的默认值。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `calib_distance` | double | 2.0 | 直线标定距离（米） |
| `calib_speed` | double | 0.2 | 直线运动速度（米/秒） |
| `calib_angular_speed` | double | 0.5 | 旋转角速度（弧度/秒） |
| `rotation_angle` | double | 6.283 | 旋转总角度（弧度，默认 2π） |
| `sample_rate` | int | 20 | 数据采样率（Hz） |
| `use_lidar` | bool | true | 是否使用激光雷达辅助 |
| `use_odom` | bool | true | 是否使用里程计（必须为 true） |
| `lidar_feature_threshold` | double | 0.05 | 激光雷达直线拟合误差阈值 |
| `save_path` | string | "./calibration_results" | 标定结果保存目录 |
| `static_time` | double | 5.0 | 静态标定持续时间（秒） |
| `gravity_norm` | double | 9.81 | 当地重力加速度 |
| `urdf_dx` | double | 0.3 | URDF 中 imu_joint 的 x 偏移 |
| `urdf_dy` | double | 0.0 | URDF 中 imu_joint 的 y 偏移 |
| `urdf_dz` | double | 0.075 | URDF 中 imu_joint 的 z 偏移 |
| `urdf_droll` | double | 0.3 | URDF 中 imu_joint 的 roll 角 |
| `urdf_dpitch` | double | 0.1 | URDF 中 imu_joint 的 pitch 角 |
| `urdf_dyaw` | double | 0.3 | URDF 中 imu_joint 的 yaw 角 |
| `prior_weight` | double | 0.1 | 先验约束的权重 |


## 输出结果

标定结果保存在 `save_path` 目录下，包含两个文件：

- `imu_static_时间戳.yaml`：静态标定结果，包括 roll、pitch 偏移。
- `imu_calib_时间戳.yaml`：联合标定最终结果，包括最终安装参数（`final_dx`, `final_dy`, `final_dyaw`, ...）和相对于 URDF 的修正量（`correction_*`）。

控制台会打印最终安装参数（例如）：
```
优化结果（最终安装参数）:
  dx: 0.3124 m (修正量 0.0124)
  dy: -0.0231 m (修正量 -0.0231)
  dyaw: 0.3245 rad (18.59°)
  droll: 0.3001 rad (17.19°)
  dpitch: 0.1000 rad (5.73°)
```

同时程序会发布 TF 变换 `base_link` → `imu_link`，使用最终的安装参数。

