from pathlib import Path
import json, base64, io, contextlib
import numpy as np
from reproduce_official import plot_comparison, OUT
# 已保存的真实计算结果补画官方比较图；无需重复 2000 次大尺寸优化。
a=dict(np.load(OUT/'official_comparison.npz'))
s=json.loads((OUT/'official_summary.json').read_text())
for name in ('original','fixed'):
    a[name+'_coefficients']=np.array(s[name]['coefficients'])
np.savez_compressed(OUT/'official_comparison.npz',**a)
plot_comparison(a)
# 将完整教学代码分为可逐步执行的 notebook 单元，执行这些单元后保存真实输出。
source=Path(__file__).with_name('learn_zernike.py').read_text(encoding='utf-8')
imports,rest=source.split('def make_forward():',1)
forward,rest=('def make_forward():'+rest).split('def fit(',1)
fit,rest=('def fit('+rest).split('def save_results(',1)
save=('def save_results('+rest).split("if __name__ == '__main__':",1)[0]
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
md('# Zernike 像差拟合：Python/JAX 学习版\n\n基于官方 zernike_fitting.ipynb。原版不动。先读 README_zh.md 中的官方复现和错误记录。这里使用官方系数的 0.2 倍、无噪声、256×256；与完整官方复现实验分开。下方代码单元已由 Python 顺序执行并保存输出。')
md('## 1. 参数与导入\n\n全部长度为 µm；SHAPE 是实际输出像素数，IMAGE_DX 是像面采样。jnp 用于可微计算，np 用于保存/绘图。ANSI 1..10 不含不可观测的全局相位 piston。')
code(imports)
md('## 2. PSF 正向模型\n\nc `(10,)` 为 OPD 系数 [µm]；basis `(10,256,256)` 为 rad/µm；phase `(256,256)` 为 rad；PSF `(256,256)` 为模拟强度。vmap 批量计算基函数，einsum 沿模式维求和。仅固定 basis，梯度仍通过相位和 FFT 传回系数。')
code(forward)
md('## 3. Loss、自动求导与正确的优化循环\n\nLoss 是标量 `()`，gradient 为 `(10,)` [µm⁻¹]。value_and_grad 同时给出 Loss 和梯度；jit 编译更新步骤。Adam 的 state 必须返回并传入下一次迭代。保留中心差分核对，以独立检查梯度。单张 PSF 相位反演非凸；此小像差例的成功不代表一般唯一性。')
code(fit)
md('## 4. 执行拟合并检查结果\n\n2000 步，学习率 0.001。history `(2001,)` 包含初始和最终更新后的 Loss。下面打印来自实际运行的指标。')
code('arrays, summary = fit()\nprint(json.dumps(summary, indent=2))\n')
md('## 5. 保存初始/目标/优化 PSF、Loss 和系数结果\n\n三个 PSF 使用相同色标；csv 中系数单位为 µm OPD，额外提供 rad 相位系数。npz 保存完整数值。像面图横纵坐标为 µm。')
code(save)
code("out = Path('learning/zernike/results') if Path('learning/zernike').exists() else Path('results')\nsave_results(arrays, summary, out)\nprint('Saved:', out.resolve())\n")
md('## 6. 完整官方复现的结论\n\n原版 1000 步 Loss≈0.0017407，外层 Adam count=0；仅修正状态后 Loss≈0.0884967，count=1000。相同学习率和迭代数下正确 Adam 没有更好，必须重新调参或考虑多初值/phase diversity。静态 ANSI 数组在当前 JAX 触发缓存元数据比较错误，独立副本以 tuple 解决。详细错误栈、物理公式、数组 shape 和 MATLAB 对照见 README_zh.md。')
ns={}
count=0
for cell in cells:
    if cell['cell_type']!='code': continue
    count+=1; buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(''.join(cell['source']),f'notebook-cell-{count}','exec'),ns)
    cell['execution_count']=count
    if buffer.getvalue(): cell['outputs'].append(dict(output_type='stream',name='stdout',text=buffer.getvalue().splitlines(keepends=True)))
    if count==6:
        for name in ('psfs.png','fit.png'):
            cell['outputs'].append(dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode((OUT/name).read_bytes()).decode(),'text/plain':[name]}))
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13.9')),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells): cell['id']=f'zernike-{i:02}'
Path(__file__).with_name('learn_zernike_zh.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Notebook cells executed:',count)

