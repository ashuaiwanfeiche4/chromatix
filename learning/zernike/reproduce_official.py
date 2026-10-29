"""执行官方模型，比较丢弃/保留 Adam 状态；不修改原始 notebook。"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import json
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(arrays):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    center = arrays['target'].shape[0] // 2
    radius = 126
    extent = np.array([-radius-.5, radius-.5, -radius-.5, radius-.5]) * .03125
    vmax = max(arrays[name].max() for name in ('initial', 'target', 'original_psf', 'fixed_psf'))
    for ax, name in zip(axs.flat, ('initial', 'target', 'original_psf', 'fixed_psf')):
        image = arrays[name][center-radius:center+radius, center-radius:center+radius]
        im = ax.imshow(image, origin='lower', extent=extent, cmap='afmhot',
                       vmin=0, vmax=vmax)
        ax.set(title=name, xlabel='x (um)', ylabel='y (um)')
    for name in ('original', 'fixed'):
        axs[1,1].semilogy(arrays[name+'_history'], label=name)
    axs[1,1].set(title='Official large-aberration target', xlabel='Updates completed', ylabel='Normalized MSE')
    axs[1,1].legend()
    axs[1,2].plot(range(1,11), arrays['truth'], 'o-', label='truth')
    for name in ('original', 'fixed'):
        axs[1,2].plot(range(1,11), arrays[name+'_coefficients'], 'x--', label=name)
    axs[1,2].set(xlabel='ANSI index', ylabel='OPD coefficient (um)'); axs[1,2].legend()
    fig.colorbar(im, ax=[axs[0,0],axs[0,1],axs[0,2],axs[1,0]], label='Simulation intensity', shrink=.6)
    fig.savefig(OUT/'official_comparison.png', dpi=150)
    plt.close(fig)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)

def main():
    notebook = json.loads((ROOT / 'docs/examples/zernike_fitting.ipynb').read_text(encoding='utf-8'))
    ns = {}
    # 跳过 Colab 安装和 notebook magic；模型、参数、噪声、Loss 完全使用官方代码。
    for index in (2, 4, 6, 8):
        source = ''.join(notebook['cells'][index]['source'])
        # 当前 JAX 比较静态元数据时，numpy 数组的 == 返回数组而非 bool。
        # 将静态 ANSI 索引改成不可变 tuple；基函数和数值参数不变。
        if index == 4:
            source = source.replace('np.arange(1, 11)', 'tuple(range(1, 11))')
        exec(source, ns)
    jax, jnp, optax = (ns[k] for k in ('jax', 'jnp', 'optax'))
    optimizer, data, loss_fn = (ns[k] for k in ('optimizer', 'psf_truth', 'loss_fn'))

    @jax.jit
    def fixed_update(model, state, target):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(model, target)
        updates, state = optimizer.update(grads, state, model)
        return optax.apply_updates(model, updates), state, metrics

    summary = {}
    arrays = {'target': np.asarray(data).squeeze(), 'truth': np.asarray(ns['coefficients_truth'])}
    for label in ('original', 'fixed'):
        model = ns['ZernikePSF'](jnp.zeros(10))
        state = optimizer.init(model)
        arrays['initial'] = np.asarray(model()).squeeze()
        history = []
        start = time.perf_counter()
        for step in range(1000):
            if label == 'original':
                model, metrics = ns['update'](model, state, data)
            else:
                model, state, metrics = fixed_update(model, state, data)
            history.append(float(metrics['loss']))
            if step % 100 == 0:
                print(label, step, history[-1], flush=True)
        final_loss = float(loss_fn(model, data)[0])
        coeff = np.asarray(model.coefficients)
        summary[label] = dict(initial_loss=history[0], final_loss=final_loss,
            coefficient_mse=float(np.mean((coeff-arrays['truth'])**2)), coefficients=coeff.tolist(),
            adam_count=int(state[0].count), seconds=time.perf_counter()-start)
        arrays[label+'_psf'] = np.asarray(model()).squeeze()
        arrays[label+'_coefficients'] = coeff
        arrays[label+'_history'] = np.asarray(history + [final_loss])
        print(label, summary[label], flush=True)
    np.savez_compressed(OUT/'official_comparison.npz', **arrays)
    summary['environment'] = dict(jax=jax.__version__, devices=[str(d) for d in jax.devices()])
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_comparison(arrays)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        (OUT/'official_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
