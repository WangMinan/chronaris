# 航空人机异步双流仿真基准规格

日期：2026-07-10  
状态：仿真生成器、oracle 真值、压力场景和审计的锁定规格

## 1. 目标与非目标

仿真基准的目标是提供现实数据中无法获得的可控真值：统一时间、时钟偏移/漂移、机动状态边界、潜在负荷和生理响应时延。

仿真不用于：

- 增加“真实样本量”。
- 生成有利于某个模型的融合特征。
- 替代专家工作负荷评价。
- 复刻现有鼎新 111 个窗口后再当独立数据。

## 2. 防止循环论证的设计

### 2.1 生成器 API

公开 API 只接受场景、profile 和随机种子：

```python
def generate_sortie(
    config: AviationScenarioConfig,
    pilot: PilotProfile,
    latent_seed: int,
    observation_seed: int,
) -> SimulatedDualStreamSortie:
    ...
```

API 不接受：

- `method_name`、`model_name` 或候选编号。
- Chronaris、MulT、ContiFormer 的配置。
- checkpoint、融合维度或评价结果。

### 2.2 两个生成族

- G1：半马尔可夫机动 + 线性/弱非线性状态空间 + 一阶生理响应。
- G2：事件驱动分段样条 + 非线性饱和耦合 + gamma 响应核。

G1 用于开发和训练；G2 只用于锁定测试。Chronaris 的 ODE-RNN 不能直接复用生成器方程或矩阵。

## 3. 仿真时间与基本规模

- 每条潜在架次时长：180 秒。
- 潜在真值网格：20 Hz，`dt=0.05 s`。
- 航电名义采样率：10 Hz。
- 生理特征名义采样率：2 Hz。
- 下游 query grid：每 5 秒 16 点，30 秒上下文 96 点。
- 每条潜在架次至少包含 2 个完整机动事件和 1 个高负荷区间。

这里生成的是生理特征级时序，不模拟原始 EEG 波形；字段必须能映射到当前仓库生理/航电 schema 的类别和单位。

## 4. 机动状态过程

### 4.1 状态

```text
steady -> entry -> sustained -> exit -> recovery -> steady
```

允许从 recovery 重新进入 entry，但不允许从 steady 直接跳到 exit/sustained。

### 4.2 持续时间

| 状态 | G1 持续时间 | G2 持续时间 |
| --- | --- | --- |
| steady | Uniform(20, 60) s | 分段事件间隔 15–70 s |
| entry | Uniform(2, 5) s | spline ramp 1.5–6 s |
| sustained | Uniform(5, 25) s | event plateau/oscillation 4–30 s |
| exit | Uniform(2, 5) s | spline release 1.5–6 s |
| recovery | Uniform(5, 20) s | nonlinear recovery 4–25 s |

生成后若 180 秒内少于两个完整事件，则使用同 seed 继续采样事件计划，不修改观测噪声。

### 4.3 机动类型

每个事件从以下类别采样，类别只用于生成动力学，不直接提供给下游：

- 俯仰主导。
- 滚转主导。
- 偏航/转弯主导。
- 加减速主导。
- 复合机动。

训练/验证/测试均覆盖全部类别；G2 的事件波形与 G1 不同。

## 5. G1 航电状态空间生成族

### 5.1 状态与控制

航电潜在状态：

```text
x_v = [speed, altitude, vertical_speed,
       roll, pitch, yaw,
       roll_rate, pitch_rate, yaw_rate,
       longitudinal_acc, lateral_acc, normal_load]
```

控制输入：

```text
u = [stick_longitudinal, stick_lateral, pedal, throttle]
```

### 5.2 演化

在真值网格上：

```text
x_v(t+dt) = A_state(dt) x_v(t)
           + B_state(dt) u(t)
           + g_state(x_v(t), u(t))
           + process_noise
```

其中 `g_state` 只允许小幅饱和、阻尼和控制耦合，不使用神经网络。

控制输入由 Ornstein-Uhlenbeck 背景过程叠加机动事件模板产生。不同机动类型只改变模板和状态矩阵，不改变观测器。

### 5.3 物理关系

干净 G1 真值需近似满足：

```text
d(altitude)/dt ~= vertical_speed
d(speed)/dt ~= longitudinal_acc
d(roll)/dt ~= roll_rate
d(pitch)/dt ~= pitch_rate
d(yaw)/dt ~= yaw_rate
normal_load ~= 1 + vertical_acc / g
```

生成器同时保存每项 residual 真值，不把 residual 当输入特征。

## 6. G2 航电事件驱动生成族

G2 不使用 G1 的 `A/B` 矩阵。生成步骤：

1. 为每个机动事件采样关键节点和连续导数约束。
2. 用三次 Hermite spline 生成姿态、速度和高度轨迹。
3. 通过数值微分得到 rate/acceleration，再加入非线性限幅、迟滞和交叉轴耦合。
4. 复合机动叠加非正弦振荡和短时 overshoot。
5. 物理关系只保持有界近似，不保证与 G1 一阶模型同构。

G2 仍需输出物理 residual，用于检验模型在不完全满足训练方程时是否稳健。

## 7. 潜在负荷状态

### 7.1 Pilot profile

每个 profile 固定：

| 参数 | 范围 |
| --- | --- |
| baseline workload | 0.10–0.30 |
| persistence half-life | 8–30 s |
| maneuver sensitivity | 0.6–1.4 |
| control-change sensitivity | 0.5–1.5 |
| overload sensitivity | 0.5–1.5 |
| fatigue slope | 0–0.0015 / s |
| physiology response lag | 2–30 s |
| physiology sensitivity | 0.6–1.4 |
| recovery rate | 0.5–1.5 |

profile 参数从先验范围以固定 seed 采样，跨数据 split 不复用 profile。

### 7.2 G1 负荷演化

```text
drive(t) = alpha_i * maneuver_intensity(t)
         + beta_i  * abs(du/dt)
         + gamma_i * abs(normal_load(t) - 1)
         + fatigue_i(t)

w(t+dt) = clip(
    decay_i(dt) * w(t)
    + (1 - decay_i(dt)) * (baseline_i + drive(t))
    + eta(t),
    0, 1
)
```

### 7.3 G2 负荷演化

G2 使用事件脉冲、交互项和饱和函数：

```text
drive(t) = sigmoid(a_i * event_energy(t)
                 + b_i * control_entropy(t)
                 + c_i * event_energy(t) * control_entropy(t))
w(t) = clip(baseline + gamma_kernel_i * drive + fatigue + eta, 0, 1)
```

gamma kernel 的 shape/scale 与 G1 一阶响应不同。

## 8. 生理特征生成

固定生成至少六类特征：

- EEG 低频/高频相对能量类特征。
- EEG 复杂度或波动类特征。
- SpO₂ 类缓慢变化特征。
- 心率/节律类特征（仅当 schema 映射允许）。
- 生理综合波动量。
- 个体基线通道。

### 8.1 G1 响应

```text
x_p,k(t+dt) = rho_i,k * x_p,k(t)
              + sensitivity_i,k * h_k(w(t - lag_i))
              + baseline_i,k
              + noise_i,k(t)
```

`h_k` 为预声明的单调或反向单调函数；方向写入 generator manifest。

### 8.2 G2 响应

- 使用 gamma convolution、阈值响应和饱和/恢复迟滞。
- 允许不同生理字段具有不同延迟，均围绕 profile 主延迟波动 ±20%。
- 不复用 G1 一阶递推系数。

## 9. 观测过程

潜在真值与传感器观测分离。每个模态独立应用：

```text
t_observed = t_true + offset + drift_ppm * 1e-6 * t_true + jitter
y_observed = scale * x_true + bias + sensor_noise
```

### 9.1 采样

- 航电默认 10 Hz，可在 5–20 Hz 间按场景变化。
- 生理默认 2 Hz，可在 0.5–5 Hz 间变化。
- irregular 场景通过随机丢弃采样时刻和扰动间隔实现，不先生成规则流再插值回规则流。

### 9.2 缺失

- 随机点缺失按 Bernoulli mask 生成。
- 连续缺失段独立采样起点和长度。
- 模态整段缺失只用于训练增强和 mixed-severe 场景，不超过单条流的 20%。
- 缺失 mask 与观测值分开保存；缺失值不得预先填零。

### 9.3 噪声

- 过程噪声在潜在状态层加入。
- 传感器噪声在观测层加入。
- clean 仍保留小幅传感器噪声，不生成完全无噪声排行榜。

## 10. 数据规模与 split

### 10.1 潜在轨迹

| split | 生成族 | profile 数 | 每 profile 轨迹数 | 潜在架次数 |
| --- | --- | ---: | ---: | ---: |
| train | G1 | 16 | 6 | 96 |
| validation | G1 | 4 | 6 | 24 |
| locked test | G2 | 8 | 6 | 48 |

profile、latent seed 和 event plan 不跨 split。

### 10.2 规范观测场景

每条潜在轨迹至少渲染六个观测版本：

1. clean-asynchronous。
2. sampling-jitter。
3. clock-offset-and-drift。
4. random-missing。
5. block-missing-and-long-lag。
6. mixed-severe。

同一潜在轨迹的六个版本共享状态、事件和负荷真值，只改变 observation seed 与观测配置。

## 11. 压力等级

锁定测试额外生成单因素配对 sweep：

| 因素 | 等级 |
| --- | --- |
| timestamp jitter std | 0、20、50、100 ms |
| absolute clock offset | 0、0.25、1、3 s；正负方向成对 |
| linear clock drift | 0、50、100、250 ppm；正负方向成对 |
| random point missing | 0%、10%、30%、50% |
| contiguous gap | 0、5、15、30 s |
| physiology response lag | 2、5、15、30 s |
| observation SNR | 30、20、10、5 dB |

mixed-severe 固定为：100 ms jitter、3 秒 offset、250 ppm drift、30% 随机缺失、15 秒连续缺失、30 秒响应时延、10 dB SNR。

退化曲线必须使用同一潜在 trajectory ID 配对，不同强度不能重新采样更容易/更难的事件。

## 12. Oracle 真值

每条观测版本必须保存：

- `true_time_s`。
- `observed_physiology_time_s`、`observed_vehicle_time_s`。
- 每个模态的真实 offset、drift、jitter 和 missing mask。
- 航电/生理潜在状态。
- 机动状态、机动类型和精确边界。
- 潜在负荷 `w(t)`。
- 主生理响应时延和字段级时延。
- 物理 residual。
- latent seed、observation seed、generator family/version。

模型输入 loader 只能读取 observed values/time/masks；oracle 只能由任务 builder、机制指标和审计模块读取。

## 13. 重型输出合同

本机忽略目录中的每个 scenario root：

```text
scenario_manifest.json
raw_dual_stream.npz
ground_truth.npz
task_manifest.jsonl
generation.log
```

仓库内紧凑 run 只保留：

- generator config 与 seed manifest。
- split/profile/scenario summary。
- 数据质量指标和紧凑图件。
- raw 文件 SHA-256 与本机相对路径。
- resume command。

## 14. 生成器验收

### 14.1 确定性与方法无关

- 同 config/profile/seeds 重复运行字节级 hash 一致。
- 生成器公开签名不存在 methods 参数。
- 搜索生成器源码不得出现 `chronaris`、`mult`、`contiformer` 或模型候选 ID。

### 14.2 状态与标签覆盖

- 每条轨迹至少两个完整机动事件。
- train/validation/test 全部覆盖五个状态和五种机动类型。
- 全局负荷低/中/高三档各占至少 15%。
- locked test 每个 profile 至少有一个高负荷事件。

### 14.3 数值与物理

- 所有有效观测为有限数；mask 外允许 NaN。
- clean G1 的标准化物理 residual 中位数小于 0.10。
- G2 residual 有界，但允许高于 G1；95% 分位数小于 0.35。
- clean 条件下，互相关估计的主响应 lag 在真值 ±1 秒内的轨迹比例至少 90%。

### 14.4 观测场景

- 随机缺失实际比例与配置差异不超过 0.02。
- offset 估计真值与配置误差小于一个真值时间步。
- 各 stress 等级的实际 jitter/drift 单调递增。
- 同 trajectory 不同场景的 latent hash 一致。

### 14.5 分布审计

输出：

- 边际范围、分位数和异常值比例。
- ACF/CCF 与响应 lag 图。
- 物理 residual 图。
- 状态持续时间和转移矩阵。
- 缺失/噪声/时钟扰动实测统计。
- synthetic-vs-real discriminator 诊断。

discriminator 只用于暴露差异，不作为反复调生成器直到“像真实数据”的优化目标。

## 15. 测试清单

- `test_generator_api_has_no_method_argument`
- `test_generator_is_seed_reproducible`
- `test_latent_trajectory_unchanged_across_observation_scenarios`
- `test_g1_g2_use_distinct_generation_paths`
- `test_ground_truth_clock_mapping_is_invertible`
- `test_maneuver_state_and_boundary_consistency`
- `test_workload_threshold_coverage`
- `test_physiology_response_lag_matches_profile`
- `test_physical_residual_ranges`
- `test_missingness_and_noise_match_config`
- `test_profile_and_seed_split_is_disjoint`
- `test_oracle_fields_are_not_exposed_to_model_loader`

## 16. LLM 使用边界

LLM 可以：

- 为 scenario manifest 生成中文场景说明草稿。
- 复核字段中文名称和单位说明。
- 组织异常案例人工复核材料。

LLM 不可以：

- 生成高频数值流。
- 决定 oracle 标签或负荷真值。
- 根据模型结果修改场景难度。
- 把生成内容写成专家评价。

