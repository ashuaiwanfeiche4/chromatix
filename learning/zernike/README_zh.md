# Zernike 像差拟合：从 MATLAB 到 Python/JAX

本独立学习示例基于 `docs/examples/zernike_fitting.ipynb`，基准提交 `ce1482906fc663298613bb252bbf425e3be59839`。原始 notebook 和 main 分支没有修改。先看 `learn_zernike_zh.ipynb`，或运行对应的 `learn_zernike.py`。完整官方复现及状态问题对照在 `reproduce_official.py`。

## 运行

在仓库根目录的 PowerShell 中：

```powershell
# 现有独立环境已安装完成；新机器可执行以下两行：
# python -m venv .venv
# & .\.venv\Scripts\python.exe -m pip install -e . matplotlib
& .\.venv\Scripts\python.exe learning/zernike/reproduce_official.py
& .\.venv\Scripts\python.exe learning/zernike/learn_zernike.py
```

Python >=3.12；此次 Python 3.13.9、JAX 0.12.0.1、CPU。固定环境见 `results/requirements-lock.txt`；其中 editable 安装指向运行机器本地路径，新机器应使用上面的 `pip install -e .`，不要原样复制该路径。脚本强制默认 CPU，若使用 GPU，应在导入 JAX 前设置适合设备的 JAX_PLATFORMS；这不是 GPU 性能测试。

## 官方复现、错误原因与修复

官方原版：ANSI 1..10，从全零系数开始，1152×1152、像面像素 0.03125 µm、固定随机种子 42 的 Poisson 目标，Adam 学习率 0.001，1000 次更新。官方 notebook 保存的最终 Loss 约 0.00174069、系数 MSE 约 6.355736e-6 µm²。此次实际数值与之在 float32 精度范围接近，详见 `results/official_summary.json`。

**状态遗漏不是语法错误，原版仍会运行和下降。** 官方在 update 内写 `updates, opt_state = optimizer.update(...)`，但仅返回 `model, metrics`。Python 函数中的局部变量不会更新外层同名变量；JAX/Optax 状态也是函数式的，必须显式传回。外层一直传入初始状态，Adam 的步数始终为 0，每次内部计算都是第一步。这样近似变成按梯度符号走固定步长的算法，并非正常累积动量的 Adam。

正确写法：

```python
updates, new_state = optimizer.update(gradient, state, coefficients)
new_coefficients = optax.apply_updates(coefficients, updates)
return new_coefficients, new_state, loss
# 外层也必须接收并传给下一步
coefficients, state, loss = step(coefficients, state)
```

Adam 应累积 m_t=β1*m_(t-1)+(1-β1)*g_t，v_t=β2*v_(t-1)+(1-β2)*g_t²，再作偏差校正。修正版运行后 count=1000；原版外层 count=0。这是直接验证状态是否持久更新的证据。

**真实遇到的兼容性报错：** 仅修复状态返回后，当前 JAX 报 `ValueError: Cache keys should be comparable using __eq__ ... arrays cannot be passed as metadata fields`。原因是官方 `ansi_indices` 被标记为静态元数据，但默认值为 NumPy 数组；数组的相等比较返回一组布尔值，不能作为缓存键的普通相等判断。复现脚本只在内存副本中将 `np.arange(1,11)` 改成 `tuple(range(1,11))`，未改变 Zernike 模式、模型或数值参数，原文件未动。首次完整错误栈保存在 `results/official_error.txt`，首次运行记录为 `results/first_attempt.log`。Agg 的非交互绘图警告只表示 plt.show() 没有窗口，不影响计算。

**优化失败也如实保留：** 教学版起初沿用官方较大的系数、去噪声并改为 256×256。正常 Adam 从零出发，2000 步后 Loss 约 0.11；程序用于检查收敛的 assert 因此触发 AssertionError（不是 JAX 计算失败）。将学习率从 0.001 改成 0.01 仍停在约 0.110214。首次记录见 `results/learning_first_attempt.log`。单张强度图的相位反演是非凸问题，正确 Adam 不保证比原版的意外算法更好，也不保证唯一系数解。

因此教学版明确采用 **官方真值的 0.2 倍、无噪声、直接 256×256、2000 步**，保持相同物理模型和 ANSI 模式，专门用于理解正确优化。它不能与官方含噪大像差结果混作同一实验。完整大像差的修正前/后结果仍单独保存，不会用小像差的成功代替官方复现。

## 模型与单位

全部长度采用 µm，真空波长 λ=0.532 µm、f=100 µm、介质折射率 n=1.33、NA=0.8。圆瞳半径 a=f*NA/n≈60.15 µm。这里是标量近轴傅里叶光学模型；高 NA 偏振模型需要另看 highNA_PSF。

OPD 波前 W(ρ,θ)=Σ c_k Z_k(ρ,θ)，相位 φ=(2π/λ)W。c_k 单位是 **µm 光程差**，不是 rad、不是镜面高度。反射式校正镜在正入射、空气中常有 OPD≈2h，不能直接把本示例 c_k 当执行器位移。normalize=False 意味着未作单位 RMS 归一化，系数不能直接套用另一套归一化 Zernike 标定。

瞳面 U=P exp(iφ)，像面 U_img∝FFT(U)，PSF=|U_img|²。`objective_point_source(z=0)` 产生均匀圆瞳场；`phase_change` 施加 φ；`ff_lens` 执行前后焦平面间的傅里叶变换并更新空间采样。采样关系 Δ_img=λf/(n*N*Δ_pupil)。教学版 Δ_pupil=1.25 µm，Δ_img=0.125 µm，不另作相机积分/重采样。默认 power=1000 是模拟功率归一化；PSF 是该模拟体系下的强度密度，不能没有标定就称为 W/m² 或光子计数。

官方用 camera_shape*upsample+pad=256*4+128=1152；pupil spacing≈1.11111 µm，实际输出 Δ_img=0.125/4=0.03125 µm。crop=450 只裁图，留下 252×252；不是重采样，也不是修改用于 Loss 的完整数据。官方直接用 `shot_noise` 把模拟强度当 Poisson 参数，适合复现演示；真实光子模型应先进行像素面积积分、曝光及量子效率标定。本学习版无噪声，不混淆这两种单位。

Loss=mean((prediction-target)²)/mean(target²)，无量纲。比较的是同一模型、同一采样、同一强度标度。自动梯度 g_k=∂Loss/∂c_k，单位 µm⁻¹。Optax 的学习率是数值步长超参数，依赖参数量纲与尺度；不能视为独立于单位的物理常数。

## 重要数组 shape

Python shape 顺序是 (行 y,列 x)，索引从 0 开始。系数一维数组 `(10,)` 没有 MATLAB 的行/列向量方向。

|变量|shape|单位/含义|
|---|---|---|
|INDICES|长度 10 的 tuple|ANSI 模式 1..10，无量纲|
|coefficients / truth / fitted|`(10,)`|OPD 系数，µm|
|jnp.eye(10)|`(10,10)`|逐个模式的单位系数输入|
|basis|`(10,256,256)`|每 1 µm OPD 系数对应相位，rad/µm|
|phase|`(256,256)`|瞳面相位，rad|
|pupil.u、field.u|`(256,256)`|复振幅，当前版本单波长标量场没有额外单例维，模拟单位|
|field.intensity|`(256,256)`|单波长标量模拟强度|
|initial / target / optimized|`(256,256)`|PSF；本版本 squeeze 不改变尺寸|
|loss|`()`|标量，无量纲|
|gradient|`(10,)`|Loss 对 OPD 系数的导数，µm⁻¹|
|updates|`(10,)`|本次数值系数增量，µm|
|state|PyTree 元组|Adam 步数 `()`，一阶矩 `(10,)`，二阶矩 `(10,)` 等|
|history|`(2001,)`|0 次至 2000 次更新后的 Loss，无量纲|

`basis` 已包含 2π/λ，计算 phase 时不要再乘一次波数。索引 k=0 对应 ANSI=1。ANSI 0（piston）未拟合，因为全局相位不能由单张 PSF 强度观测。

|ANSI|径向阶数 n_z / 方位阶数 m|含义|
|---|---|---|
|1,2|1/−1，1/+1|y、x 倾斜|
|3,4,5|2/−2，2/0，2/+2|斜像散、离焦、正像散|
|6,7,8,9|3/−3，3/−1，3/+1，3/+3|三叶、彗差、彗差、三叶|
|10|4/−4|四叶；不是球差|

注意 n_z 是 Zernike 阶数，和介质 n 不同。库中 theta=atan2(y,x)，负 m 用 sin，正 m 用 cos；与 MATLAB 的 Zernike 符号、排序及 y 轴方向比较时必须核对。本图 origin=lower，使 y 正向向上。

## MATLAB 对照与核心代码

- NumPy 用于保存和绘图；JAX NumPy（jnp）用于可微计算。不要在 jit 内把被追踪的数组强行转成普通 NumPy。
- `make_forward()` 用闭包固定孔径和 basis，返回只接收 c 的 forward。类似 MATLAB 中用匿名函数固定其它参数；可训练参数只有 10 个系数，不需要先学 Equinox 类和 PyTree 模型。
- `jax.vmap` 批量生成 10 个模式，等价于对单位系数向量逐行做循环。
- `jnp.einsum('k,kyx->yx',c,basis)` 是沿 k 维求和，相当于 MATLAB 的各 Zernike 模式乘对应系数再累加。
- `jax.value_and_grad(loss_fn)` 同时返回标量 Loss 与 10 维梯度，使用反向自动微分，不是有限差分优化。脚本额外以中心差分作独立核对；相对误差约 1.11e-4，容差 0.02。
- `jax.jit(step)` 编译固定 shape 的计算；首次调用包含编译，不应该当作稳态运行时间。优化循环显式传递新 state，避免官方遗漏。
- `squeeze()` 只去掉长度为 1 的维度，不是插值、下采样或卷积。

单张焦内 PSF 常存在相位恢复歧义。对中心对称实瞳，φ(r) 与 −φ(−r) 可产生相同强度；因此 PSF 很接近不等于任意情况下系数唯一正确。这里同时报告系数误差；真实像差校正可加入已知离焦的多平面 phase diversity、先验或多初值策略。

## 输出

- `results/psfs.png`：初始、目标、优化 PSF，三者相同裁剪及线性色标。
- `results/fit.png`：Loss 曲线及真值/拟合系数。
- `results/coefficients.csv`：逐模式 OPD 系数、误差及相位系数。
- `results/loss.csv`：每步 Loss；`learning_arrays.npz` 保存完整数值数组。
- `results/learning_summary.json`：Loss、系数 RMSE、状态步数及梯度核对。
- `results/official_comparison.png`、`official_summary.json`、`official_comparison.npz`：完整官方参数的修正前后对照。

数值无噪声教学实验达到的极小误差是自洽合成实验结果，不代表真实测量精度。所有比较结果均由脚本运行产生；原始 notebook 保留。


### 此次实际结果（2026-10-29）

|实验|最终归一化 MSE|系数误差|外层 Adam 步数|
|---|---|---|---|
|完整官方原版，1000 步|0.001740705|MSE 6.355817e-6 µm²|0|
|完整官方状态修正版，1000 步|0.08849667|MSE 0.007318244 µm²|1000|
|无噪声小像差教学版，2000 步|约 2.8e-14|RMSE 约 2.44e-8 µm|2000|

前两行是相同数据、参数和迭代数下的对照；教学版是另一组更容易的实验，不能直接横向比较 Loss。修正状态后并未在 1000 步内复原官方系数，这一限制保留在结果中。教学版核对梯度、状态计数、有限值及 Loss 下降；系数结果见 CSV。图中的强度不用于实际光子数标定。
