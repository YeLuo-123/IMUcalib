#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
from scipy.optimize import minimize
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt
from collections import deque
import tf2_ros
from geometry_msgs.msg import TransformStamped
import yaml
import os
from datetime import datetime
import time

class IMUCalibration4WDNode(Node):
    def __init__(self):
        super().__init__('imu_calibration_4wd_node')
        
        # 参数声明
        self.declare_parameter('calibration_distance', 2.0)  # 标定移动距离(m)
        self.declare_parameter('calibration_speed', 0.2)     # 标定移动速度(m/s)
        self.declare_parameter('sample_rate', 20)            # 采样率(Hz)
        self.declare_parameter('use_lidar', True)            # 是否使用激光雷达
        self.declare_parameter('use_odom', True)             # 是否使用里程计
        self.declare_parameter('lidar_feature_threshold', 0.1)  # 激光雷达特征阈值
        self.declare_parameter('save_path', './calibration_results')  # 保存路径
        self.declare_parameter('static_calibration_time', 5.0)  # 静态标定时间(秒)
        self.declare_parameter('gravity_norm', 9.81)  # 重力加速度标准值
        
        # 初始化变量
        self.imu_data = []           # 存储IMU数据
        self.static_imu_data = []     # 存储静态IMU数据
        self.lidar_data = []          # 存储激光雷达数据
        self.odom_data = []           # 存储里程计数据
        self.robot_pose = []          # 存储机器人位姿
        self.is_calibrating = False
        self.is_static_calibrating = False
        self.calibration_complete = False
        self.static_calibration_complete = False
        
        # 订阅话题
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # 发布控制指令
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # 发布标定后的TF变换
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # 创建定时器用于数据采样
        self.timer = self.create_timer(1.0/self.get_parameter('sample_rate').value, self.sample_data)
        
        # 创建保存目录
        self.save_path = self.get_parameter('save_path').value
        os.makedirs(self.save_path, exist_ok=True)
        
        self.get_logger().info('四轮麦轮车IMU标定节点已启动')
        self.get_logger().info('请将机器人放置在开阔区域，按s键开始标定')
        
    def imu_callback(self, msg):
        """IMU数据回调"""
        # 提取IMU数据
        timestamp = self.get_clock().now().nanoseconds
        orientation_q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, 
            orientation_q.z, orientation_q.w])
        
        # 检查数据有效性
        if (math.isnan(msg.linear_acceleration.x) or 
            math.isnan(msg.linear_acceleration.y) or 
            math.isnan(msg.linear_acceleration.z)):
            return
        
        if self.is_calibrating:
            self.imu_data.append({
                'timestamp': timestamp,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'angular_vel': msg.angular_velocity,
                'linear_acc': msg.linear_acceleration
            })
        
        if self.is_static_calibrating:
            # 计算加速度模长，确保数据有效
            acc_norm = math.sqrt(
                msg.linear_acceleration.x**2 + 
                msg.linear_acceleration.y**2 + 
                msg.linear_acceleration.z**2
            )
            g_lower = self.get_parameter('gravity_norm').value * 0.8
            g_upper = self.get_parameter('gravity_norm').value * 1.2
            
            if g_lower < acc_norm < g_upper:
                self.static_imu_data.append({
                    'timestamp': timestamp,
                    'linear_acc': [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z
                    ],
                    'angular_vel': [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z
                    ],
                    'orientation': [
                        orientation_q.x, orientation_q.y, 
                        orientation_q.z, orientation_q.w
                    ],
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw
                })
    
    def lidar_callback(self, msg):
        """激光雷达数据回调"""
        if self.is_calibrating and self.get_parameter('use_lidar').value:
            timestamp = self.get_clock().now().nanoseconds
            # 提取激光雷达数据
            ranges = np.array(msg.ranges)
            angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
            
            # 滤除无效数据
            valid_idx = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
            ranges = ranges[valid_idx]
            angles = angles[valid_idx]
            
            if len(ranges) > 0:
                # 转换为笛卡尔坐标
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)
                
                # 检测直线特征（假设墙面）
                if len(x) > 10:
                    # 使用RANSAC算法检测墙面
                    wall_features = self.detect_wall_features(x, y)
                    
                    self.lidar_data.append({
                        'timestamp': timestamp,
                        'x': x,
                        'y': y,
                        'wall_features': wall_features
                    })
    
    def odom_callback(self, msg):
        """里程计数据回调"""
        if self.is_calibrating and self.get_parameter('use_odom').value:
            timestamp = self.get_clock().now().nanoseconds
            pose = msg.pose.pose
            x = pose.position.x
            y = pose.position.y
            
            # 提取位姿
            orientation_q = pose.orientation
            _, _, yaw = euler_from_quaternion([
                orientation_q.x, orientation_q.y, 
                orientation_q.z, orientation_q.w])
            
            self.odom_data.append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'yaw': yaw
            })
    
    def detect_wall_features(self, x, y):
        """检测激光雷达中的墙面特征"""
        features = []
        
        # 简单的直线检测
        if len(x) > 20:
            # 使用最小二乘法拟合直线
            A = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # 计算拟合误差
            y_pred = k * x + b
            error = np.mean(np.abs(y - y_pred))
            
            if error < self.get_parameter('lidar_feature_threshold').value:
                features.append({
                    'type': 'wall',
                    'slope': k,
                    'intercept': b,
                    'error': error,
                    'points': len(x)
                })
        
        return features
    
    def sample_data(self):
        """采样数据并处理"""
        if not self.is_calibrating:
            return
            
        # 检查是否完成标定移动
        if len(self.odom_data) > 10:
            start_pose = self.odom_data[0]
            current_pose = self.odom_data[-1]
            
            # 计算移动距离
            distance = math.sqrt(
                (current_pose['x'] - start_pose['x'])**2 + 
                (current_pose['y'] - start_pose['y'])**2)
            
            if distance >= self.get_parameter('calibration_distance').value:
                self.stop_calibration()
    
    def start_static_calibration(self):
        """开始静态标定"""
        self.get_logger().info('开始IMU静态标定...')
        
        # 清除旧数据
        self.static_imu_data.clear()
        
        self.is_static_calibrating = True
        static_time = self.get_parameter('static_calibration_time').value
        
        self.get_logger().info(f'请保持机器人静止，正在采集 {static_time} 秒的静态数据...')
        
        # 创建定时器，一段时间后自动停止静态标定
        self.static_timer = self.create_timer(static_time, self.stop_static_calibration, oneshot=True)
    
    def stop_static_calibration(self):
        """停止静态标定"""
        self.is_static_calibrating = False
        
        if hasattr(self, 'static_timer'):
            self.static_timer.cancel()
        
        self.get_logger().info('静态数据采集完成，开始计算静态标定参数...')
        
        # 计算静态标定参数
        self.calculate_static_calibration()
    
    def calculate_static_calibration(self):
        """计算静态标定结果"""
        if len(self.static_imu_data) < 50:
            self.get_logger().error(f'静态数据不足，仅采集到 {len(self.static_imu_data)} 个有效数据点')
            return
        
        # 计算平均加速度向量
        acc_data = np.array([d['linear_acc'] for d in self.static_imu_data])
        avg_acc = np.mean(acc_data, axis=0)
        std_acc = np.std(acc_data, axis=0)
        
        self.get_logger().info(f'加速度统计:')
        self.get_logger().info(f'  平均值: [{avg_acc[0]:.6f}, {avg_acc[1]:.6f}, {avg_acc[2]:.6f}] m/s²')
        self.get_logger().info(f'  标准差: [{std_acc[0]:.6f}, {std_acc[1]:.6f}, {std_acc[2]:.6f}] m/s²')
        self.get_logger().info(f'  模长: {np.linalg.norm(avg_acc):.6f} m/s² (理论值: {self.get_parameter("gravity_norm").value} m/s²)')
        
        # 计算 Roll 和 Pitch
        ax, ay, az = avg_acc
        
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        roll = math.atan2(ay, az)
        
        self.get_logger().info(f'静态标定结果:')
        self.get_logger().info(f'  Roll: {roll:.3f} rad ({math.degrees(roll):.2f}°)')
        self.get_logger().info(f'  Pitch: {pitch:.3f} rad ({math.degrees(pitch):.2f}°)')
        
        # 保存静态标定结果
        self.save_static_calibration_result(roll, pitch, avg_acc, std_acc)
        
        self.static_calibration_complete = True
    
    def save_static_calibration_result(self, roll, pitch, avg_acc, std_acc):
        """保存静态标定结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_path, f'imu_static_calibration_{timestamp}.yaml')
        
        result = {
            'static_calibration': {
                'roll_offset': float(roll),
                'pitch_offset': float(pitch),
                'roll_offset_deg': float(math.degrees(roll)),
                'pitch_offset_deg': float(math.degrees(pitch)),
                'average_acceleration': {
                    'x': float(avg_acc[0]),
                    'y': float(avg_acc[1]),
                    'z': float(avg_acc[2])
                },
                'std_acceleration': {
                    'x': float(std_acc[0]),
                    'y': float(std_acc[1]),
                    'z': float(std_acc[2])
                },
                'sample_count': len(self.static_imu_data),
                'timestamp': timestamp
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f'静态标定结果已保存到 {filename}')
    
    def start_calibration(self):
        """开始标定"""
        self.get_logger().info('开始IMU标定...')
        
        # 清除旧数据
        self.imu_data.clear()
        self.lidar_data.clear()
        self.odom_data.clear()
        self.robot_pose.clear()
        
        self.is_calibrating = True
        
        # 发布前进指令
        cmd = Twist()
        cmd.linear.x = self.get_parameter('calibration_speed').value
        self.cmd_pub.publish(cmd)
        
        self.get_logger().info('机器人正在前进，请等待...')
    
    def stop_calibration(self):
        """停止标定"""
        # 停止机器人
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        
        self.is_calibrating = False
        
        self.get_logger().info('数据采集完成，开始计算标定参数...')
        
        # 计算标定参数
        self.calibrate_imu()
    
    def calibrate_imu(self):
        """计算IMU标定参数"""
        if len(self.imu_data) < 10:
            self.get_logger().error('IMU数据不足，无法标定')
            return
        
        # 对齐时间戳
        aligned_data = self.align_timestamps()
        
        if len(aligned_data) < 5:
            self.get_logger().error('时间戳对齐失败')
            return
        
        # 使用优化方法求解IMU安装参数
        # 参数: [dx, dy, dyaw, droll, dpitch] 平移x, 平移y, 旋转角yaw, 旋转角roll, 旋转角pitch
        initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        result = minimize(
            self.cost_function,
            initial_guess,
            args=(aligned_data),
            method='BFGS'
        )
        
        if result.success:
            dx, dy, dyaw, droll, dpitch = result.x
            
            self.get_logger().info('='*50)
            self.get_logger().info('IMU标定结果:')
            self.get_logger().info(f'X方向偏移: {dx:.3f} m')
            self.get_logger().info(f'Y方向偏移: {dy:.3f} m')
            self.get_logger().info(f'Roll角偏移: {droll:.3f} rad ({math.degrees(droll):.2f}°)')
            self.get_logger().info(f'Pitch角偏移: {dpitch:.3f} rad ({math.degrees(dpitch):.2f}°)')
            self.get_logger().info(f'Yaw角偏移: {dyaw:.3f} rad ({math.degrees(dyaw):.2f}°)')
            self.get_logger().info('='*50)
            
            # 保存标定结果
            self.save_calibration_result(dx, dy, dyaw, droll, dpitch)
            
            # 发布TF变换
            self.publish_imu_tf(dx, dy, dyaw, droll, dpitch)
            
            self.calibration_complete = True
        else:
            self.get_logger().error('标定优化失败')
    
    def align_timestamps(self):
        """对齐IMU和里程计的时间戳"""
        aligned = []
        
        for odom in self.odom_data:
            # 找到时间最接近的IMU数据
            closest_imu = min(
                self.imu_data,
                key=lambda imu: abs(imu['timestamp'] - odom['timestamp'])
            )
            
            # 找到时间最接近的激光雷达数据
            closest_lidar = None
            if self.lidar_data:
                closest_lidar = min(
                    self.lidar_data,
                    key=lambda lidar: abs(lidar['timestamp'] - odom['timestamp'])
                )
            
            time_diff = abs(closest_imu['timestamp'] - odom['timestamp'])
            if time_diff < 0.1e9:  # 时间差小于0.1秒
                aligned.append({
                    'odom': odom,
                    'imu': closest_imu,
                    'lidar': closest_lidar
                })
        
        return aligned
    
    def cost_function(self, params, aligned_data):
        """代价函数：计算预测误差"""
        dx, dy, dyaw, droll, dpitch = params
        
        total_error = 0
        
        for i in range(1, len(aligned_data)):
            prev_data = aligned_data[i-1]
            curr_data = aligned_data[i]
            
            # 1. 计算时间差
            delta_time = (curr_data['odom']['timestamp'] - prev_data['odom']['timestamp']) / 1e9
            if delta_time <= 0:
                continue
            
            # 2. 根据里程计数据计算机器人速度和加速度
            # 位置变化
            delta_x = curr_data['odom']['x'] - prev_data['odom']['x']
            delta_y = curr_data['odom']['y'] - prev_data['odom']['y']
            
            # 速度
            prev_vel_x = delta_x / delta_time
            prev_vel_y = delta_y / delta_time
            
            # 3. 根据IMU测量的加速度，计算IMU相对于基坐标系的加速度
            # IMU测量的加速度（机体坐标系）
            imu_acc_x_body = curr_data['imu']['linear_acc'].x
            imu_acc_y_body = curr_data['imu']['linear_acc'].y
            
            # 将IMU加速度转换到基坐标系（考虑安装旋转）
            # 安装旋转矩阵
            cos_alpha = math.cos(dyaw)
            sin_alpha = math.sin(dyaw)
            
            imu_acc_x_base = imu_acc_x_body * cos_alpha - imu_acc_y_body * sin_alpha
            imu_acc_y_base = imu_acc_x_body * sin_alpha + imu_acc_y_body * cos_alpha
            
            # 4. 计算IMU安装位置对加速度的影响
            # 计算机器人基坐标系的旋转
            robot_yaw = curr_data['odom']['yaw']
            cos_robot_yaw = math.cos(robot_yaw)
            sin_robot_yaw = math.sin(robot_yaw)
            
            # IMU在基坐标系中的位置
            imu_x = curr_data['odom']['x'] + dx * cos_robot_yaw - dy * sin_robot_yaw
            imu_y = curr_data['odom']['y'] + dx * sin_robot_yaw + dy * cos_robot_yaw
            
            # 5. 计算理论加速度（基于机器人运动）
            # 这里假设机器人做匀速运动，理论加速度为0
            # 但实际上应该考虑机器人的实际加速度
            # 简化处理，使用速度变化计算加速度
            theoretical_acc_x = 0.0
            theoretical_acc_y = 0.0
            
            # 6. 误差计算
            # 加速度误差
            acc_error = (imu_acc_x_base - theoretical_acc_x)**2 + (imu_acc_y_base - theoretical_acc_y)**2
            
            # 姿态误差
            imu_measured_yaw = curr_data['imu']['yaw'] - dyaw
            attitude_error = (imu_measured_yaw - robot_yaw)**2 * 100  # 增加姿态误差的权重
            
            # 7. 位置误差（基于速度积分）
            # 根据IMU加速度积分计算位置变化
            imu_predicted_delta_x = prev_vel_x * delta_time + 0.5 * imu_acc_x_base * delta_time**2
            imu_predicted_delta_y = prev_vel_y * delta_time + 0.5 * imu_acc_y_base * delta_time**2
            
            # 位置误差
            position_error = (delta_x - imu_predicted_delta_x)**2 + (delta_y - imu_predicted_delta_y)**2
            
            # 8. 总误差
            total_error += acc_error + attitude_error + position_error
        
        # 正则化项，防止参数过大
        regularization = (dx**2 + dy**2 + dyaw**2) * 0.1
        total_error += regularization
        
        return total_error
    
    def publish_imu_tf(self, dx, dy, dyaw, droll, dpitch):
        """发布IMU的TF变换"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'imu_link'
        
        # 设置平移
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        t.transform.translation.z = 0.0
        
        # 设置旋转
        quat = quaternion_from_euler(droll, dpitch, dyaw)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('已发布IMU标定后的TF变换')
    
    def save_calibration_result(self, dx, dy, dyaw, droll, dpitch):
        """保存标定结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_path, f'imu_calibration_{timestamp}.yaml')
        
        result = {
            'imu_calibration': {
                'x_offset': float(dx),
                'y_offset': float(dy),
                'z_offset': 0.0,
                'roll_offset': float(droll),
                'pitch_offset': float(dpitch),
                'yaw_offset': float(dyaw),
                'roll_offset_deg': float(math.degrees(droll)),
                'pitch_offset_deg': float(math.degrees(dpitch)),
                'yaw_offset_deg': float(math.degrees(dyaw)),
                'timestamp': timestamp
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        self.get_logger().info(f'标定结果已保存到 {filename}')
    
    def plot_results(self):
        """绘制标定结果（可选）"""
        if not self.calibration_complete:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 绘制里程计轨迹
        if self.odom_data:
            odom_x = [d['x'] for d in self.odom_data]
            odom_y = [d['y'] for d in self.odom_data]
            plt.subplot(2, 2, 1)
            plt.plot(odom_x, odom_y, 'b-', label='Odometry')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('机器人轨迹')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
        
        # 绘制IMU角度
        if self.imu_data:
            imu_roll = [math.degrees(d['roll']) for d in self.imu_data]
            imu_pitch = [math.degrees(d['pitch']) for d in self.imu_data]
            imu_yaw = [math.degrees(d['yaw']) for d in self.imu_data]
            
            plt.subplot(2, 2, 2)
            plt.plot(imu_roll, 'r-', label='IMU Roll')
            plt.plot(imu_pitch, 'g-', label='IMU Pitch')
            plt.plot(imu_yaw, 'b-', label='IMU Yaw')
            plt.xlabel('Sample')
            plt.ylabel('Angle (deg)')
            plt.title('IMU角度')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = IMUCalibration4WDNode()
    
    # 简单的键盘控制
    import sys
    import select
    import tty
    import termios
    
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    print("\n控制指令:")
    print("s - 开始动态标定")
    print("t - 开始静态标定")
    print("p - 显示结果")
    print("q - 退出")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 's':
                    node.start_calibration()
                elif key == 't':
                    node.start_static_calibration()
                elif key == 'p':
                    node.plot_results()
                elif key == 'q':
                    break
                    
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()