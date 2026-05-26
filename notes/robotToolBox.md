# 机器人运动学基础--**库： Robotics Toolbox** 

**库： Robotics Toolbox** 

资料：Peter Corke 的 Robotics Toolbox for Python 文档： https://petercorke.com/toolboxes/robotics-toolbox/ 

[Robotics Toolbox for Python 官方文档](https://petercorke.github.io/robotics-toolbox-python/?utm_source=chatgpt.com)

[Introduction — Robotics Toolbox for Python documentation](https://petercorke.github.io/robotics-toolbox-python/intro.html#spatial-math-layer)

GitHub：

[robotics-toolbox-python GitHub](https://github.com/petercorke/robotics-toolbox-python?utm_source=chatgpt.com)

正确顺序应该是：

```
1. 坐标变换 SE(3)
↓
2. DH参数
↓
3. 正运动学 FK
↓
4. 雅可比 Jacobian
↓
5. 逆运动学 IK
↓
6. 轨迹与控制
```

## （一）空间变换 / SE(3) / 姿态表示（最基础）

机器人和计算机视觉需要在 **三维空间中描述位置（position）、姿态（orientation）和位姿（pose）**。

因此我们需要工具来表示这些量，例如：

- 刚体变换（rigid-body transformations）
- 旋转（rotation）
- 欧拉角 / RPY角
- 四元数（unit quaternion）

Robotics Toolbox 中的 Spatial Math 层主要用于表示和操作机器人位姿。相比直接使用 NumPy 矩阵，它通过 SE3、SO3 等类对位姿进行封装，使得运算具备明确的物理语义，同时保证类型安全和代码可读性。这种抽象方式是机器人系统建模和控制的基础。

PYTHON中通过**SMTB-P**数学库实现，主要用于处理三维空间中的位姿（位移 + 旋转）、坐标变换、刚体运动与相关的线性代数运算。它通常配合机器人仿真、控制算法与运动学/动力学推导使用。 

**主要功能**

- **空间变换与坐标系处理**
  提供齐次变换矩阵、旋转矩阵、四元数、欧拉角、轴角等多种表示方式，并支持相互转换和链式组合，方便在多个坐标系之间进行点和向量的变换。
- **刚体运动与姿态运算**
  支持位姿插值（如 SLERP）、合成与分解姿态、计算相对位姿以及常见的 3D 旋转操作，适合轨迹规划、相机位姿标定等任务。
- **线性代数与数值工具**
  在 NumPy 基础上封装更贴合空间数学的工具函数，如批量矩阵运算、向量化几何操作、常用的数值稳定处理等，使空间计算代码更简洁可读。
- **与机器人/仿真生态的衔接**
  通常能较容易与 ROS、Gazebo、Mujoco 等机器人或物理仿真框架集成，用于统一各种传感器、关节、末端执行器的姿态与坐标表示。



示例代码：代码运行顺序 **顺序是从右往左作用的！**

```
from spatialmath.base import *
T = transl(0.5, 0.0, 0.0) @ rpy2tr(0.1, 0.2, 0.3, order='xyz') @ trotx(-90, 'deg')
```

`transl` → 平移

`rpy2tr` → RPY角转变换矩阵

`trotx` → 绕x轴旋转

`@` → 矩阵乘法（Python）



**与matlab代码：**

在 MATLAB：

- 所有数组都是 2D（哪怕是向量）

在 Python（NumPy）：

- 向量是 **1D数组**

### 1.位姿 SE（n）

数学表达：

```
T = [ R  t
      0  1 ]
```

- R：旋转（方向）
- t：位置（坐标）

这个东西叫**SE(3)**

#### 实现方式 

##### （1）矩阵 

```
T = transl(...) @ rpy2tr(...) @ trotx(...)
```

| 函数   | 含义        |
| ------ | ----------- |
| transl | 平移        |
| rpy2tr | RPY角转矩阵 |
| trotx  | 绕x轴旋转   |

主要不可用原因：

没有类型，运算不安全，时间序列难处理 



##### （2）采用类的方式

| 类             | 含义   |
| -------------- | ------ |
| SE3            | 3D位姿 |
| SO3            | 3D旋转 |
| UnitQuaternion | 四元数 |
| Twist3         | 速度   |

```
from spatialmath import *

T = SE3(0.5, 0.0, 0.0) * SE3.RPY([0.1, 0.2, 0.3], order='xyz') * SE3.Rx(-90, unit='deg')
```

表示位姿组合：先平移 → 再旋转 → 再绕x轴转 

最终版结果：、

```
T =
[ 0.9752  -0.1987  -0.0978   0.5
  0.1538   0.2896   0.9447   0
 -0.1593  -0.9363   0.3130   0
  0        0        0        1 ]
```

使用该对象：**语义访问**

###### 1️⃣ 取旋转

```
T.R
```

------

左上角（3×3）

```
R =
[ 0.9752  -0.1987  -0.0978
  0.1538   0.2896   0.9447
 -0.1593  -0.9363   0.3130 ]
```

###### 2️⃣ 取平移

```
T.t
```

------

右上角 

```
t = [0.5, 0, 0]
```



###### 3️⃣ 转欧拉角

```
T.eul()
```

###### 4.可视化

```
T.plot()
```



### 2.四元数 

```
UnitQuaternion.Rx(0.3)
UnitQuaternion.AngVec(0.3, [1, 0, 0])
```

本质都是绕 x 轴旋转 0.3 rad 

四元数是用“旋转轴 + 半角三角函数”来表示旋转的一种方式，它比欧拉角更稳定，比旋转矩阵更紧凑，是机器人和3D系统中最常用的旋转表示方法。 



### 3.多重数值 

为了支持“多个值的序列”，这些类型（比如 SE3、SO3 等）都继承了 Python 的collections.UserList，因此它们具备类似 列表（list） 的特性。SMTB-P中的任意位姿类，都可以包含多个值（一个序列）。

有些构造函数支持传入“数组”，从而一次性创建多个位姿对象。

```python
from spatialmath import SE3
import numpy as np
R = SE3.Rx(np.linspace(0, np.pi/2, num=100)) #创建100个旋转，从0 → 90°
len(R)
```

R = SE3.Rx(np.linspace(0, np.pi/2, num=100)) 相当于对每一个角度，生成一百个旋转矩阵

]R = [
    SE3.Rx(θ1),
    SE3.Rx(θ2),
    SE3.Rx(θ3),
    ...
    SE3.Rx(θ100)
]

R 这个对象里面，包含了 100 个旋转矩阵。当你用一个“单个位姿”去和这个序列做乘法时，这个**单个位姿**会自动**应用到序列中的每一个元素上**



### 4.Common constructors

这些工具箱中的类具有一定的多态性（polymorphic），并且提供了多种“不同形式的构造函数”，可以用来创建对象。也就是说：同一个 SE3 / SO3，可以用很多种方式创建。

(1)姿态（orientation）可以通过多种方式来表示，例如：

基本轴旋转（绕 x/y/z 轴）
欧拉向量（Euler vector）
角度 + 旋转轴（angle-vector）
欧拉角 / RPY角
方向向量 + 接近向量（orientation & approach vectors）

(2)可以通过随机值来创建对象：`.Rand()` 

(3)SE3、SE2、SO3 和 SO2 还支持一种**矩阵指数（exponential）构造方法**，其输入参数是对应的**李代数（Lie algebra）元素**。 

se(3) --exp--> SE(3)

(4)可以创建一个“空对象”（empty），即：没有任何值（长度为0） SE3.Empty()

(5)可以创建一个长度为 N 的对象数组，并且所有元素都初始化为该对象的“单位值（identity）”

```
SE3.Alloc(5) 
```

得到：

```
[T0, T0, T0, T0, T0]
```

其中：

```
T0 = 单位位姿（不动）
```



### 5.Common methods and operators

这些类型（SE3、SO3 等）都提供了一个求逆的方法.inv()；

并且支持使用 / 运算符来表示“与逆的组合”；T1 / T2---->T1 * T2.inv()  

还支持使用 `**` 运算符进行整数次幂运算（即重复组合）T ** 3===T * T * T

其他重载运算符包括：

*：位姿组合:T1 * T2,坐标变换链
*=：原地组合
**：重复组合
/：乘以逆:从 T2 坐标系 → T1 坐标系
==、!=：比较
+、-：加减



### 6.性能 

| 函数/方法       | 执行时间 |
| --------------- | -------- |
| base.rotx()     | 4.07 μs  |
| base.trotx()    | 5.79 μs  |
| SE3.Rx()        | 12.3 μs  |
| SE3 * SE3       | 4.69 μs  |
| 4x4 @           | 0.986 μs |
| SE3.inv()       | 7.62 μs  |
| base.trinv()    | 4.19 μs  |
| np.linalg.inv() | 4.49 μs  |



### 7.机器人模型 

| 方法            | 含义                    |
| --------------- | ----------------------- |
| DH              | Denavit-Hartenberg 参数 |
| MDH             | 改进 DH                 |
| ETS             | 变换序列                |
| rigid-body tree | 刚体树                  |
| URDF            | 机器人文件              |

### （1）DH 

调用模型： rtb.models.DH.Panda()

要用 DH 方法定义机器人，需要：

- 创建一个 `DHRobot` 类
- 传入一个“连杆（link）列表”

```
robot = DHRobot(
    [
        RevoluteDH(alpha=pi/2),
        RevoluteDH(a=0.4318),
        RevoluteDH(d=0.15005, a=0.0203, alpha=-pi/2),
        RevoluteDH(d=0.4318, alpha=pi/2),
        RevoluteDH(alpha=-pi/2),
        RevoluteDH()
    ], name="Puma560")
```

| 参数 | 含义        |
| ---- | ----------- |
| θ    | 关节角      |
| d    | 沿 z 轴平移 |
| a    | 连杆长度    |
| α    | 扭转角      |

关节模型：

维度1：关节类型
类型	含义
Revolute	旋转关节（转）
Prismatic	平移关节（滑）
📌 维度2：DH建模方式
类型	含义
DH	标准 DH
MDH	改进 DH

所以：

⭐ 所以组合出来：
类名	含义
RevoluteDH	旋转 + 标准DH
PrismaticDH	滑动 + 标准DH
RevoluteMDH	旋转 + 改进DH
PrismaticMDH	滑动 + 改进DH

还可以额外指定：

质量（mass）、重心（CoG）、惯性、摩擦、关节限制 



推荐为封装写法：

```
class Puma560(DHRobot):

    def __init__(self):
        super().__init__(
                [
                    RevoluteDH(alpha=pi/2),
                    RevoluteDH(a=0.4318),
                    RevoluteDH(d=0.15005, a=0.0203, alpha=-pi/2),
                    RevoluteDH(d=0.4318, alpha=pi/2)
                    RevoluteDH(alpha=-pi/2),
                    RevoluteDH()
                ], name="Puma560"
                        )
```

使用：

```
import roboticstoolbox as rtb
puma = rtb.models.DH.Puma560()                  # instantiate robot model
print(puma)
```

实例化

```
import roboticstoolbox as rtb
puma = rtb.models.DH.Puma560()                  # instantiate robot model
print(puma)
DHRobot: Puma 560 (by Unimation), 6 joints (RRRRRR), dynamics, geometry, standard DH parameters
┌────┬────────┬────────┬────────┬─────────┬────────┐
│θⱼ  │   dⱼ   │   aⱼ   │   ⍺ⱼ   │   q⁻    │   q⁺   │
├────┼────────┼────────┼────────┼─────────┼────────┤
│ q1 │ 0.6718 │      0 │  90.0° │ -160.0° │ 160.0° │
│ q2 │      0 │ 0.4318 │   0.0° │ -110.0° │ 110.0° │
│ q3 │   0.15 │ 0.0203 │ -90.0° │ -135.0° │ 135.0° │
│ q4 │ 0.4318 │      0 │  90.0° │ -266.0° │ 266.0° │
│ q5 │      0 │      0 │ -90.0° │ -100.0° │ 100.0° │
│ q6 │      0 │      0 │   0.0° │ -266.0° │ 266.0° │
└────┴────────┴────────┴────────┴─────────┴────────┘

┌─┬──┐
└─┴──┘

┌─────┬─────┬──────┬───────┬─────┬──────┬─────┐
│name │ q0  │ q1   │ q2    │ q3  │ q4   │ q5  │
├─────┼─────┼──────┼───────┼─────┼──────┼─────┤
│  qr │  0° │  90° │ -90°  │  0° │  0°  │  0° │
│  qz │  0° │  0°  │  0°   │  0° │  0°  │  0° │
│  qn │  0° │  45° │  180° │  0° │  45° │  0° │
│  qs │  0° │  0°  │ -90°  │  0° │  0°  │  0° │
└─────┴─────┴──────┴───────┴─────┴──────┴─────┘

print(puma.qr)
[ 0.      1.5708 -1.5708  0.      0.      0.    ]
T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # forward kinematics
print(T)
   0.1217   -0.6067   -0.7856    0.2478    
   0.8184    0.5092   -0.2665   -0.1259    
   0.5617   -0.6105    0.5584    1.146     
   0         0         0         1         

sol = puma.ikine_LM(T)                          # inverse kinematics
print(sol)
IKSolution: q=[0.1, 2.025, 2.936, -2.894, -2.273, -2.025], success=True, iterations=14, searches=1, residual=4.91e-10
```



print(puma) 

print(puma.qr)：读取姿态---一组关节角（单位：弧度） 

T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])：正运动学,给定关节角 → 计算末端位姿



### （2）ETS表示法

ETS：基本变换序列 

它是由一系列**简单的刚体变换组成**：

- 纯平移（translation）
- 纯旋转（rotation）

每个变换：

- 要么是常数参数
- 要么是变量（关节变量）

```python
from roboticstoolbox import ET
import roboticstoolbox as rtb
# Puma dimensions (m), see RVC2 Fig. 7.4 for details
l1 = 0.672; l2 = -0.2337; l3 = 0.4318; l4 = 0.0203; l5 = 0.0837; l6 = 0.4318
e = ET.tz(l1) * ET.Rz() * ET.ty(l2) * ET.Ry() * ET.tz(l3) * ET.tx(l4) * ET.ty(l5) * ET.Ry() * ET.tz(l6) * ET.Rz() * ET.Ry() * ET.Rz()
print(e)
tz(0.672) ⊕ Rz(q0) ⊕ ty(-0.2337) ⊕ Ry(q1) ⊕ tz(0.4318) ⊕ tx(0.0203) ⊕ ty(0.0837) ⊕ Ry(q2) ⊕ tz(0.4318) ⊕ Rz(q3) ⊕ Ry(q4) ⊕ Rz(q5)
robot = rtb.Robot(e) 
print(robot)
ERobot: , 6 joints (RRRRRR)
┌─────┬────────┬───────┬────────┬───────────────────────────────────────────────┐
│link │  link  │ joint │ parent │              ETS: parent to link              │
├─────┼────────┼───────┼────────┼───────────────────────────────────────────────┤
│   0 │ link0  │     0 │ BASE   │ tz(0.672) ⊕ Rz(q0)                            │
│   1 │ link1  │     1 │ link0  │ ty(-0.2337) ⊕ Ry(q1)                          │
│   2 │ link2  │     2 │ link1  │ tz(0.4318) ⊕ tx(0.0203) ⊕ ty(0.0837) ⊕ Ry(q2) │
│   3 │ link3  │     3 │ link2  │ tz(0.4318) ⊕ Rz(q3)                           │
│   4 │ link4  │     4 │ link3  │ Ry(q4)                                        │
│   5 │ @link5 │     5 │ link4  │ Rz(q5)                                        │
└─────┴────────┴───────┴────────┴───────────────────────────────────────────────┘
```



```
带 q → 关节变量
不带 → 固定变换
```

其中⊕表示依次组合

也就是说robot = rtb.Robot(e) 表示，也就是把ETS转换成机器人模型 

系统会自动把这些变换拆分成：

- link（连杆）
- joint（关节）

| 列     | 含义             |
| ------ | ---------------- |
| link   | 当前连杆         |
| joint  | 关节编号         |
| parent | 父节点           |
| ETS    | 从父到当前的变换 |



### （3）URDF建模

导入 URDF 文件，这是ROS中描述机器人的标准文件，工具箱内置了解析器，可以直接解析：URDF和xacro 文件

使用URDF模型：rtb.models.URDF.Panda()

```
import roboticstoolbox as rtb
panda = rtb.models.URDF.Panda()
print(panda)
ERobot: panda (by Franka Emika), 7 joints (RRRRRRR), 1 gripper, geometry, collision
┌─────┬──────────────┬───────┬─────────────┬────────────────────────────────────────────────┐
│link │     link     │ joint │   parent    │              ETS: parent to link               │
├─────┼──────────────┼───────┼─────────────┼────────────────────────────────────────────────┤
│   0 │ panda_link0  │       │ BASE        │ SE3()                                          │
│   1 │ panda_link1  │     0 │ panda_link0 │ SE3(0, 0, 0.333) ⊕ Rz(q0)                      │
│   2 │ panda_link2  │     1 │ panda_link1 │ SE3(-90°, -0°, 0°) ⊕ Rz(q1)                    │
│   3 │ panda_link3  │     2 │ panda_link2 │ SE3(0, -0.316, 0; 90°, -0°, 0°) ⊕ Rz(q2)       │
│   4 │ panda_link4  │     3 │ panda_link3 │ SE3(0.0825, 0, 0; 90°, -0°, 0°) ⊕ Rz(q3)       │
│   5 │ panda_link5  │     4 │ panda_link4 │ SE3(-0.0825, 0.384, 0; -90°, -0°, 0°) ⊕ Rz(q4) │
│   6 │ panda_link6  │     5 │ panda_link5 │ SE3(90°, -0°, 0°) ⊕ Rz(q5)                     │
│   7 │ panda_link7  │     6 │ panda_link6 │ SE3(0.088, 0, 0; 90°, -0°, 0°) ⊕ Rz(q6)        │
│   8 │ @panda_link8 │       │ panda_link7 │ SE3(0, 0, 0.107)                               │
└─────┴──────────────┴───────┴─────────────┴────────────────────────────────────────────────┘

┌─────┬─────┬────────┬─────┬───────┬─────┬───────┬──────┐
│name │ q0  │ q1     │ q2  │ q3    │ q4  │ q5    │ q6   │
├─────┼─────┼────────┼─────┼───────┼─────┼───────┼──────┤
│  qr │  0° │ -17.2° │  0° │ -126° │  0° │  115° │  45° │
│  qz │  0° │  0°    │  0° │  0°   │  0° │  0°   │  0°  │
└─────┴─────┴────────┴─────┴───────┴─────┴───────┴──────┘

T = panda.fkine(panda.qz, end='panda_hand')
print(T)
   0.7071    0.7071    0         0.088     
   0.7071   -0.7071    0         0         
   0         0        -1         0.926     
   0         0         0         1         

```

其中 @panda_link8表示末端执行器 

可以指定任一link来计算他的位姿 

同时URDF模型包含了：mesh 网格模型（Collada 格式） ，提供了几何形状和颜色

**可视化：**panda.plot(qz, backend="swift") 



**总结：** 

ETS 表示法将机器人建模为一系列基本的旋转和平移变换，比 DH 更直观；而 URDF 则是工程中使用的标准机器人描述格式，支持复杂结构和真实几何模型。



## (二)正运动学 FK 

给定关节角 → 计算末端位姿





## （三）逆运动学（IK）

1.数值法ikine_LM：sol = puma.ikine_LM(T)： 

2.解析法ikine_a：puma.ikine_a(T, config="lun") 

config（“lun”）表示选择解： 

```
l / r → 左手 / 右手
u / d → 肘上 / 肘下
n / f → 手腕翻转 / 不翻转
```



给定末端位姿 → 求关节角

```
Levenberg-Marquardt 最小化算法的数值迭代方法
```

在逆运动学求解中：

- **初始关节角**会影响**收敛速度**和**最终解**
- 一个**末端位姿**可能对应**多个关节解**
- 对于冗余机械臂，虽然可以找到解，但无法自动控制多余自由度（null-space）
- 对于自由度不足或部分约束问题，需要指定哪些笛卡尔自由度可以忽略

| 方法   | 函数     | 特点           |
| ------ | -------- | -------------- |
| 数值法 | ikine_LM | 慢、通用       |
| 解析法 | ikine_a  | 快、特定机器人 |



## （四）雅可比矩阵

##### ⭐ 核心关系

$$
\dot{x} = J(q) \, \dot{q}
$$



------

##### 👉 含义

| 符号 | 意思          |
| ---- | ------------- |
| q    | 关节角        |
| x    | 末端位置/姿态 |
| J    | 雅可比矩阵    |
| ẋ    | 末端速度      |
| q̇    | 关节速度      |

```
J = robot.jacob0(q)  #在基坐标系下的 Jacobian
```

```
J = robot.jacobe(q)  #末端坐标系下
```



## （五）轨迹



主要有两种轨迹：

- **关节空间轨迹**：直接在机器人的各个关节角度上规划路径。
- **笛卡尔空间轨迹**：在三维空间中的直线路径（比如让机器人手沿着一条直线移动）。



### （1）关节空间轨迹 

Puma 机器人从零角度位姿到直立（或就绪）位姿在 100 个步长内的关节空间轨迹如下：

```
>>> import robotistoolbox as rtb
>>> puma = rtb.models.DH.Puma560()
>>> traj = rtb.jtraj(puma.qz, puma.qr, 100)
>>> traj.q.shape
```

- `puma.qz`：零角度位姿（所有关节在0度位置）
- `puma.qr`：直立（就绪）位姿
- `100`：分成 100 个时间步

`traj` 是一个**命名元组**，包含三个数组：

- `q`：关节角度
- `qd`：关节速度（`q̇`）
- `qdd`：关节加速度

每个数组的维度是 `(100, 6)` → 100个时间步，每个步有6个关节的角度/速度/加速度。

**轨迹特点**

- 使用**五阶多项式**（保证加加速度连续 → 运动平滑，不抖动）
- 默认起点和终点的速度为 0（平滑启停）

**绘制轨迹**

```
rtb.qplot(traj.q)
```



### （2）笛卡尔空间轨迹

不是直接在关节上规划，而是让**末端执行器（手）在空间中走直线**。
两点之间的路径用 **SE(3)** 位姿（位置 + 姿态）来描述。

```
# 伪代码逻辑
Ts = 生成笛卡尔直线轨迹（200个位姿点）
for 每个位姿点:
    使用 ikine_LM 方法计算对应的关节角度
```



- 每个时间步的逆运动学解，以上一步的关节角度作为初始值（提高效率和稳定性）
- 最终得到一个 `(200, 6)` 的关节角度数组

##### 核心思想总结

| 类型           | 规划对象 | 优点               |
| :------------- | :------- | :----------------- |
| 关节空间轨迹   | 关节角度 | 计算快、避免奇异点 |
| 笛卡尔空间轨迹 | 末端位姿 | 路径直观（直线）   |

最后通过**逆运动学**把笛卡尔路径转换回关节角度，让机器人实际执行。



## （六）符号计算 

机器人工具箱（Robotics Toolbox）支持**用符号变量代替具体数字**来推导机器人的运动学公式。它底层用的是 SymPy（Python 的符号计算库）。 

- **符号计算**：用 `θ₁, θ₂, ...` 这样的符号 → 推导出数学公式 

### 1. 用符号计算旋转矩阵（RPY角→旋转矩阵）

python

```
phi, theta, psi = base.sym.symbol('φ, θ, ψ')
base.rpy2r(phi, theta, psi)
```



- `φ, θ, ψ`：滚动角、俯仰角、偏航角（符号变量，没有具体数值）
- `rpy2r`：将欧拉角转换为旋转矩阵

**结果**：生成一个 3×3 的矩阵，矩阵里全是 `sin(φ)`、`cos(θ)` 这样的**符号表达式**，而不是数字。

> 例如：矩阵第一行第一个元素是 `cos(φ)*cos(θ)`。

------

### 2. 符号化正向运动学（symbolic=True）

```python
puma = rtb.models.DH.Puma560(symbolic=True)
```

- 正常不加 `symbolic=True`：得到的是一个数值机器人模型
- 加了这个参数：模型中的 DH 参数（连杆长度、偏置等）变成**符号常量**



```python
q = base.sym.symbol("q_6")
T = puma.fkine(q)
```



- `q`：6 个关节角度的符号变量（`q₁, q₂, ...`）
- `fkine(q)`：正向运动学，计算机器人末端的位置和姿态

python

```
T.t[0]
```



- `T.t`：末端位置向量（x, y, z）
- `T.t[0]`：**x 坐标的符号表达式**

它看起来类似这样（简化理解）：

text

```
0.15005*sin(q₀) - 0.0203*sin(q₁)*sin(q₂)*cos(q₀) - 0.4318*sin(q₁)*cos(q₀)*cos(q₂) - ...
```



> 文中说“显示为红色”，指的是在 Jupyter Notebook 等环境下，符号常量会用红色字体显示，提醒你这不是数字。



作用：

| 用途     | 说明                                             |
| :------- | :----------------------------------------------- |
| 推导公式 | 直接得到机器人末端位置与各关节角度的数学关系     |
| 导出代码 | SymPy 可以把表达式转成 C / Python / MATLAB 代码  |
| 写论文   | 直接转成 LaTeX 公式（`\sin(q_1)\cos(q_2)` 这种） |
| 分析特性 | 比如雅可比矩阵、奇异点、工作空间边界             |

把机器人的运动学公式用**符号变量**推导出来，而不是代入数字——方便你分析公式结构、导出代码、写论文，甚至直接生成 LaTeX 公式。 



## （七）微分运动学 

该工具箱可以计算雅可比矩阵：

python

```
J = puma.jacob0(q)   # 在基坐标系中
J = puma.jacobe(q)   # 在末端坐标系中
```

返回结果为 **NumPy 数组**。在奇异位形下：

```
import roboticstoolbox as rtb
puma = rtb.models.DH.Puma560()
J = puma.jacob0(puma.qr)
np.linalg.matrix_rank(J)   # 输出 5
rtb.jsingu(J)              # 输出：column 5 = 2.6e-15 column_0 + 1 column_3
```

对于符号型关节变量，也可以像之前正向运动学那样计算雅可比矩阵。

此外，我们还可以计算海森矩阵：

python

```
H = puma.hessian0(q)   # 基坐标系
H = puma.hessiane(q)   # 末端坐标系
```

返回结果为 **3D NumPy 数组**。

对于所有机器人类，我们还可以计算可操作度（manipulability）：

````
```
import roboticstoolbox as rtb
puma = rtb.models.DH.Puma560()
m = puma.manipulability(puma.qn)
print("Yoshikawa manipulability is", m)
```
#输出：Yoshikawa manipulability is 0.07861716534599998
m = puma.manipulability(puma.qn, method="asada")
print("Asada manipulability is", m)
#输出：Asada manipulability is 0.0043746137281665005
````



以上分别对应 Yoshikawa（吉川） 和 Asada（浅田） 两种可操作度度量方法。

还可以指定只考虑任务空间中的平动自由度：

python

```
m = puma.manipulability(puma.qn, axes="trans")
print("Yoshikawa manipulability is", m)

输出：Yoshikawa manipulability is 0.11118146146764128
```

此外，我们还可以计算**可操作度雅可比矩阵**：

python

```
Jm = puma.manipm(q, J, H)
```

满足关系：


![image-20260526201839424](C:\Users\15504\AppData\Roaming\Typora\typora-user-images\image-20260526201839424.png)



## （八）动力学

Python 工具箱支持多种计算动力学的方法。对于使用标准 DH 或改进 DH 表示的模型，我们使用经典的递归牛顿-欧拉算法（Python 或 C 实现）。

> **注意**：与 RTB-M 相同的 C 代码可直接从 Python 调用，且不使用 NumPy。

### 逆动力学示例

python

```
import roboticstoolbox as rtb
puma = rtb.models.DH.Puma560()
tau = puma.rne(puma.qn, np.zeros((6,)), np.zeros((6,)))
print(tau)
# 输出：[-0.     31.6399  6.0351  0.      0.0283  0.    ]
```



这个结果就是机器人在位形 `qn` 下的**重力矩**。

### 动力学项的计算

分别使用以下方法计算惯性、科里奥利/离心力和重力项：

```
puma.inertia(q)      # 惯性矩阵
puma.coriolis(q, qd) # 科里奥利/离心力项
puma.gravload(q)     # 重力项
```

这些方法基于 Orin 和 Walker 的逆动力学方法，计算结果包含了电机惯性和摩擦力的影响。

### 正动力学

```
qdd = puma.accel(q, tau, qd)   # 给定力矩计算加速度
```

可以通过时间积分得到轨迹：

```
q = puma.fdyn(5, q0, mycontrol, ...)
```

该方法使用 SciPy 中的 RK45 数值积分，根据可选的控制器函数求解关节轨迹，控制器函数的调用格式为：

```
tau = mycontrol(robot, t, q, qd, **args)
```

