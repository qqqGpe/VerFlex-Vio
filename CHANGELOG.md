# VerFlex-VIO 更新日志 (Changelog)

本文件记录 VerFlex-VIO 的关键优化,用于长期追溯。提交日期为实际 commit 日期。

---

## 2026-06-20 · `feature/decouple-ros` 近期三笔提交

### 1. `75a21d8` FIX: resolve MH_04/MH_05 crash via covariance PSD recovery + clone td self-covariance

**问题:** MH_04/MH_05 在立体初始化后的第一次 ESKF update 即 `std::exit` 崩溃。根因是初始化协方差非 PSD(传播期积累的 p-v / p-bias 交叉项 + 克隆 td 耦合不完整),Joseph form 产生负对角元,旧代码的 PSD 守卫直接杀进程。

**优化(两处,均在 `src/solver/eskf_solver.h`):**

- **(a) 克隆 td 自协方差补全(Fix 2):** `StochasticClone` 的 td 耦合只加了 state↔clone 的**交叉协方差**,漏了 clone 自身对角块上的 `J_td·Cov(td,td)·J_tdᵀ` 自协方差项。补全后增广成为完整的合同变换,保证 PSD。详见下文 [td_visual 优化的数学解释](#td_visual-优化的数学解释)。
- **(b) PSD 恢复替代硬退出(Fix 3):** `update()` 中把 `std::exit(EXIT_FAILURE)`(任意负对角即杀进程)换成**特征值投影**:对协方差做 `SelfAdjointEigenSolver`,把负特征值钳到 `1e-12` 下限再重建,记 warning 继续。这样数值漂移(低特征帧被病态 Kalman 增益放大)不再杀进程。

**效果:** MH_04/MH_05 从崩溃 → 跑通(MH_04 ≈ 0.69 m,MH_05 ≈ 5.05 m,此为 warp 关闭时;后续提交进一步优化)。

> 备注:调试中还发现初始化协方差本身非 PSD(`P_marg` 有负特征值 -0.379)——这是更深的根因,Fix 3 只在 update 时钳位、未根治 init/clone 的非 PSD。若日后要让投影卡方真正生效,需正确修 init 协方差(保留传播对角、只清交叉项),而非早期尝试过的「整个重置成对角先验」(那会丢掉合法的不确定性,导致别的序列回退)。

---

### 2. `f0edc14` REF: remove anchored MSCKF inverse-depth representation (no effect on MSCKF updates)

**问题:** 评估 anchored MSCKF 逆深度表示(`ANCHORED_MSCKF_INVERSE_DEPTH`)是否对 MSCKF 点有用。

**方法:** A/B 实测 —— 全 9 个 EuRoC 序列上 `ANCHORED_MSCKF_INVERSE_DEPTH` vs `GLOBAL_3D`,APE 完全相同(Δ = 0.000,RMSE/中位/max 逐项一致)。

**原因(数学):** MSCKF 特征通过**零空间投影**被边缘化,特征本身的 3 个自由度在投影后被消去。因此特征参数化(全局 3D vs 锚定逆深度)对**状态更新**无影响 —— 锚定逆深度的优势(对远特征更好的数值条件/线性度)只对**保留在状态里的 SLAM 特征**有效,对被边缘化的 MSCKF 特征无效。

**优化:** 移除 anchored 路径(删 `LandmarkRepresentation.h`、`test_anchored_msckf_jacobian.cpp`,清 `feat_rep_msckf` 选项、anchor 簿记字段、anchor Jacobian 块、CMake test 目标)。MSCKF 统一用 `GLOBAL_3D`。

---

### 3. `413f8c3` ENH: rotation-compensated warp KLT + projected adaptive chi-square gate

**优化(前端 + 后端):**

- **前端 warp KLT(`src/core/vioFrontend.cpp/h`):** 旋转补偿 KLT 用于 stereo 时间跟踪。原 warp 路径存在但对 stereo 配置被门控关死(时间跟踪调用传了 `is_stereo_tracking=true`),且 `K(LEFT_CAM)` 硬编码。修复:按相机选 K(`warp_cam_id`)、最小角度门(<1° 不 warp,避免慢序列插值噪声)、3 层前向 LK。原理:用 IMU 预测的帧间旋转 `Rij` warp 目标图,使 LK 只处理平移 —— 直接对症快旋转跟踪丢失。
- **后端投影自适应马氏卡方门(`src/core/visualManager.cpp`):** 替代禁用的简化检查 `‖sum(res)‖/cnt>4`(几乎不触发)。逐特征:零空间投影消去特征自由度后,`χ² = res_projᵀ S⁻¹ res_proj`,`S = H_proj·P_marg·H_projᵀ + σ_pix²·I`(`P_marg` 由 `eskfSolver::MarginalCovariance` 取)。within-pass 自适应:限制每次 update 的拒绝率(提高乘子直到 ≤50% 被拒)、特征稀缺(<20)时跳过门,避免困难序列被饿死。χ²₉₅ 用 Wilson-Hilferty 近似。同时加了三角化条件数 + 深度界检查。

**效果:** V1_02_medium 2.652 → 0.285 m(9 倍);8/9 序列优于 warp-only 基线(MH_01 0.068、MH_04 0.802、V2_01 0.065 等)。

**已知限制:** MH_05_difficult 发散(147 m)。投影门退化成非投影门 —— 因为 `R = σ_pix² = 4` 在 `S` 中主导,状态不确定性项 `H_proj·P_marg·H_projᵀ` 可忽略,`S ≈ R` ⇒ `χ²_proj ≈ resᵀR⁻¹res` = 非投影重投影卡方。于是仍会拒掉 MH_05 上信息量大的高视差特征(其高重投影残差来自对位姿误差的敏感)。接受此限制(8/9 优秀)。

**EuRoC 评估(APE RMSE,SE(3) 对齐,米):**

| 序列 | RMSE | 中位数 | max | 基线(warp off) |
|---|---|---|---|---|
| MH_01_easy | 0.068 | 0.055 | 0.20 | 0.086 |
| MH_02_easy | 0.104 | 0.082 | 0.23 | 0.167 |
| MH_03_medium | 0.152 | 0.116 | 0.32 | 0.456 |
| MH_04_difficult | 0.802 | 0.524 | 4.61 | CRASH |
| MH_05_difficult | 147.316 | 32.534 | 1010.77 | CRASH(发散,接受) |
| V1_01_easy | 0.113 | 0.092 | 0.61 | 0.105 |
| V1_02_medium | 0.285 | 0.118 | 3.31 | 2.652 |
| V2_01_easy | 0.065 | 0.046 | 0.60 | 0.136 |
| V2_02_medium | 0.109 | 0.096 | 0.55 | 0.139 |

---

### 4. `2eb510b` FIX: sqrt-root ESKF Kalman gain (U⁻¹ → U⁻ᵀ) — 修复 sqrt-root 求解器精度

**问题:** sqrt-root 求解器(`solver_type=1`)在 Vicon 序列(V1_01/V2_01/V2_02)和 MH_05 上发散/精度差(V1_01 1.17m、V2_01 0.58m、MH_05 270m),而 eskf(`solver_type=0`)正常。所有序列均不崩溃(0 negative-diag-exit)。

**根因(数学):** sqrt-root `update()` 的 Kalman 增益 K 算错了。QR 分解后增广矩阵
`M_all = [R^{1/2}  0 ; S·Hᵀ  S]`(其中 `Cov = SᵀS`)经 Givens 旋转得上三角
`rhks = [U  W ; 0  S_new]`,其中:
- `UᵀU = S_innov`(新息协方差,U 上三角、非对称)
- `UᵀW = H·P`  ⇒  `P·Hᵀ = Wᵀ·U`

正确增益:`K = P·Hᵀ·S_innov⁻¹ = Wᵀ·U·(UᵀU)⁻¹ = Wᵀ·U⁻ᵀ`。
原代码:`K = K_hat · qr.solve(I) = Wᵀ·U⁻¹`(`qr.solve(I) = U⁻¹`)。

**`U⁻¹ ≠ U⁻ᵀ`**(U 来自 QR 上三角、非对称)→ 增益错误 → 状态更新 `dx = K·res` 错误
→ 状态漂移 → 快旋转序列发散。注意:协方差 `S_new` 是正确的
(`S_newᵀS_new = P_post`),所以**不崩溃**,只是状态估错 —— 这正是 sqrt-root
「能跑但精度差」的原因。

**修复(`src/solver/sqrt_eskf_solver.h`):**
`K = (U⁻¹·W)ᵀ = qr.solve(W).transpose()`。

**效果(全 9 序列,sqrt-root + K 修复 + warp + chi2):**

| 序列 | sqrt(K 修复) | eskf | sqrt(原bug) |
|---|---|---|---|
| MH_01_easy | 0.064 | 0.068 | 0.081 |
| MH_02_easy | 0.111 | 0.104 | 0.087 |
| MH_03_medium | 0.168 | 0.152 | 0.168 |
| MH_04_difficult | 0.574 | 0.802 | 0.578 |
| MH_05_difficult | **4.099** | 147.316 | 270.036 |
| V1_01_easy | 0.095 | 0.113 | 1.174 |
| V1_02_medium | 0.285 | 0.285 | 0.299 |
| V2_01_easy | 0.071 | 0.065 | 0.578 |
| V2_02_medium | 0.094 | 0.109 | 0.415 |
| **Σ RMSE** | **5.561** | 149.014 | 273.417 |

- Σ RMSE **5.561**(eskf 149,buggy sqrt 273)。**最优配置**。
- **MH_05: 270 → 4.099 m**(不再发散;eskf 147 仍发散)。本次最大的单项改善。
- 8/9 优于或持平 eskf。0 崩溃(全部 9 序列跑通)。

**结论:** sqrt-root(K 修复后)成为最优求解器 —— 结构性 PSD(`Cov = SᵀS`,
无 eskf 的非 PSD / PSD 恢复问题)+ 正确增益。默认配置切换到 `solver_type: 1`。

> 注:eskf 那边的 td_visual 改善(Fix 2 自协方差、Fix 3 PSD 恢复)sqrt-root **不需要适配** ——
> sqrt-root 的克隆增广(`clone_S = imu_pose_S + td_S·J_tdᵀ`)经 `Cov = SᵀS` 天然含全 4 项
> (含 td 自协方差),且 `Cov = SᵀS` 恒 PSD。eskf 的 Fix 2/Fix 3 是协方差形式特有的补救。

---

## td_visual 优化的数学解释

> 对应提交 `75a21d8` 的 Fix 2(克隆 td 自协方差补全)。这是本次最需要从数学层面理解的优化。

### 背景:td_visual 与克隆

`td_visual`(记为 `td`)是相机–IMU 时间偏移,约定 **`t_imu = t_cam + td`**(IMU 为参考时钟,相机时间加 `td` 对齐到 IMU 时间,与 OpenVINS 一致)。

MSCKF 在每个图像帧把当前 IMU 位姿**随机克隆**(stochastic clone)到状态里,作为该帧的位姿锚点。克隆位姿 `c` 是图像时刻 `t_cam` 处的 IMU 位姿,而 IMU 状态在 `t_imu`。由 `t_cam = t_imu − td`,克隆位姿相对当前 IMU 位姿需要按 `−td` 外推。对 `td` 一阶线性化:

$$
c \approx \text{imu\_pose} + J_{td}\cdot td,\qquad J_{td} = \frac{\partial c}{\partial td} = \begin{bmatrix}\omega \\ v\end{bmatrix}\in\mathbb{R}^{6\times1}
$$

其中 `ω`、`v` 是 IMU 角速度/线速度(`6×1`,与 OpenVINS `augment_clone` 的 `dnc_dt = [ω; v]` 一致)。即**克隆位姿对 `td` 的灵敏度 = 该时刻的速度旋量**。

### 增广协方差 = 合同变换

增广状态 `x_aug = [x_old; c]`,其协方差由合同变换给出:

$$
\text{Cov}_{aug} = J_{full}\,\text{Cov}_{old}\,J_{full}^{\top},\qquad
J_{full} = \begin{bmatrix} I_n & 0 \\ J_{pose} & J_{td} \end{bmatrix}
$$

其中 `J_pose` 从 `x_old` 中选出 IMU 位姿对应的行(克隆 = IMU 位姿 + td 修正),`n = dim(x_old)`。

展开,克隆自协方差块为:

$$
\text{Cov}(c,c) = \underbrace{J_{pose}\,\text{Cov}_{old}\,J_{pose}^{\top}}_{\text{Cov}(\text{imu\_pose},\text{imu\_pose})}
\;+\;\underbrace{J_{pose}\,\text{Cov}_{old}(:,td)\,J_{td}^{\top}}_{\text{Cov}(\text{imu\_pose},td)\,J_{td}^{\top}}
\;+\;\underbrace{J_{td}\,\text{Cov}(td,:)\,J_{pose}^{\top}}_{J_{td}\,\text{Cov}(td,\text{imu\_pose})}
\;+\;\underbrace{J_{td}\,\text{Cov}(td,td)\,J_{td}^{\top}}_{\text{td 自协方差贡献}}
$$

state↔clone 交叉协方差为:

$$
\text{Cov}(\text{state},c) = \text{Cov}_{old}\,J_{pose}^{\top} + \text{Cov}_{old}(:,td)\,J_{td}^{\top}
$$

### 原代码的错误:对角块漏了 3 个 td 项

原 `StochasticClone`:
- 交叉协方差 `Cov(state,c)`:**正确**(含 `Cov_old(:,td)·J_tdᵀ`)。
- 自协方差 `Cov(c,c)`:**只设了** `Cov(imu_pose, imu_pose)`,**漏了后 3 个 td 项**:

$$
\text{Cov}(\text{imu\_pose},td)\,J_{td}^{\top}\;\;+\;\;J_{td}\,\text{Cov}(td,\text{imu\_pose})\;\;+\;\;J_{td}\,\text{Cov}(td,td)\,J_{td}^{\top}
$$

### 为什么漏掉这 3 项会破坏 PSD(数学原因)

**(1) 合同变换保 PSD 的前提是「完整」。**
对任意 `J`,`J·P·Jᵀ` 在 `P ⪰ 0` 时 `⪰ 0`(合同变换保半正定)。但原代码的 `Cov_aug` **不是**完整的 `J_full·Cov_old·J_fullᵀ`(对角块不完整),因此**不享有 PSD 保证** —— 可以出现负特征值。Joseph form 在非 PSD 的 `Cov_old` 上产生负对角元 ⇒ 旧 PSD 守卫 `std::exit` ⇒ MH_04/MH_05 崩溃。

**(2) Schur 补 / Cauchy–Schwarz 破裂。**
考虑 clone–td 子块:

$$
\begin{bmatrix}\text{Cov}(c,c) & \text{Cov}(c,td) \\ \text{Cov}(td,c) & \text{Cov}(td,td)\end{bmatrix}
$$

PSD 要求 Schur 补 `Cov(c,c) − Cov(c,td)·Cov(td,td)⁻¹·Cov(td,c) ⪰ 0`,等价于 `|Cov(c,td)|² ≤ Cov(c,c)·Cov(td,td)`(Cauchy–Schwarz)。漏掉 `J_td·Cov(td,td)·J_tdᵀ` 使 `Cov(c,c)` 被**低估**,而 `Cov(c,td)`(含 `J_td·Cov(td,td)`)是**正确**的。当 `J_td` 较大(快旋转/快平移 ⇒ `ω`、`v` 大)时,`|Cov(c,td)|²` 超过被低估的 `Cov(c,c)·Cov(td,td)` ⇒ Schur 补变负 ⇒ 非 PSD。**这正好解释了为何只在 difficult 序列(MH_04/MH_05 快运动)崩溃**:easy 序列 `J_td` 小,漏掉的项可忽略,恰好不越 PSD 边界。

**(3) 物理含义。**
`J_td·Cov(td,td)·J_tdᵀ` 是**克隆位姿因 `td` 不确定性而产生的方差**。漏掉它 = 克隆位姿的不确定性被低估(没算上 `td` 诱导的部分),却保留了 state↔clone 经由 `td` 的**相关性**。这种「有相关性、无匹配方差」的不一致正是 PSD 的破坏点 —— 协方差矩阵要求任意两个分量的相关性不超过各自方差的几何均值。

### 修复(Fix 2)

在克隆对角块上补上漏掉的 3 项,使 `Cov_aug` 等于完整的合同变换 `J_full·Cov_old·J_fullᵀ`:

```cpp
Cov_aug.block(insert_idx, insert_idx, 6, 6) +=
      Cov_aug.block(0, td_id, 6, 1) * J_td.transpose()      // Cov(imu_pose,td)·J_tdᵀ
    + J_td * Cov_aug.block(td_id, 0, 1, 6)                   // J_td·Cov(td,imu_pose)
    + J_td * Cov_aug.block(td_id, td_id, 1, 1) * J_td.transpose();  // J_td·Cov(td,td)·J_tdᵀ
```

补全后增广协方差是真正的合同变换 ⇒ `Cov_old ⪰ 0 ⇒ Cov_aug ⪰ 0`,克隆增广不再引入非 PSD。

### 与 OpenVINS 的对比

OpenVINS 的 `augment_clone`(`ov_msckf/src/state/StateHelper.cpp`)同样**只加交叉协方差、省略** `J_td·Cov(td,td)·J_tdᵀ` 自项。OpenVINS 不崩,是因为它没有「负对角即 `std::exit`」的硬守卫,靠整体数值鲁棒性容忍这一微小非 PSD。本仓库 Fix 2 **比 OpenVINS 更严格**(补全合同变换),从源头消除非 PSD,而非靠容忍。

---

## 附:提交链(feature/decouple-ros)

```
（本次） FIX: sqrt-root ESKF Kalman gain (U⁻¹ → U⁻ᵀ); switch default to solver_type=1
413f8c3 ENH: rotation-compensated warp KLT + projected adaptive chi-square gate
f0edc14 REF: remove anchored MSCKF inverse-depth representation (no effect on MSCKF updates)
75a21d8 FIX: resolve MH_04/MH_05 crash via covariance PSD recovery + clone td self-covariance
07c8102 ENH: optimize frontend tracking accuracy   (此前)
```
