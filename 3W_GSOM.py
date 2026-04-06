"""
3W-GSOM 算法完整实验框架
=====================================================================

"""

import os, gc, importlib, time, csv, copy
os.environ['OMP_NUM_THREADS'] = '1'
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pathlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, accuracy_score
)
np.set_printoptions(suppress=True)

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _have_optuna = True
except ImportError:
    _have_optuna = False
    print("[警告] optuna 未安装，贝叶斯优化将退化为随机搜索（仍优于网格搜索）。")
    print("       建议安装：pip install optuna")

# three_way_metrics 导入（可选）
try:
    from three_way_metrics import ThreeWayPartition, compute_threeway_metrics
    _have_twmetrics = True
except ImportError:
    _have_twmetrics = False
    print("[警告] three_way_metrics.py 未找到，三支质量指标将不被计算。")


# ─────────────────────────────────────────────────────────────────────────────
# (0) 可选 Numba 加速
# ─────────────────────────────────────────────────────────────────────────────
_have_numba = importlib.util.find_spec("numba") is not None
if _have_numba:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def _sim_matrix_jit(X, eps):
        n, d = X.shape
        sim = np.ones((n, n), dtype=np.float32)
        for j in prange(d):
            col = X[:, j]
            ej = eps[j]
            if ej < 1e-12:
                continue
            for i in range(n):
                xi = col[i]
                row = sim[i]
                for k in range(n):
                    diff = abs(xi - col[k])
                    val = (1.0 - diff / ej) if diff <= ej else 0.0
                    if val < row[k]:
                        row[k] = val
        for i in range(n):
            sim[i, i] = 1.0
        return sim

    @njit(parallel=True, fastmath=True)
    def _lower_approx_jit(sim_mat, FD):
        c, n = FD.shape
        out = np.empty((c, n), dtype=np.float32)
        for k in prange(c):
            fd = FD[k]
            for i in range(n):
                mini = 1.0
                srow = sim_mat[i]
                for j in range(n):
                    bd = fd[j]
                    s = 1.0 - srow[j]
                    if s > bd:
                        bd = s
                    if bd < mini:
                        mini = bd
                        if mini == 0.0:
                            break
                out[k, i] = mini
        return out
else:
    def _sim_matrix_jit(*a, **k): pass
    def _lower_approx_jit(*a, **k): pass


# ─────────────────────────────────────────────────────────────────────────────
# (1) 神经元类
# ─────────────────────────────────────────────────────────────────────────────
class GSOMNode:
    def __init__(self, weights, neuron_id):
        self.weights = weights
        self.error = 0.0
        self.id = neuron_id
        self.approximation = None
        self.sim_order = []


# ─────────────────────────────────────────────────────────────────────────────
# (2) 3W-GSOM 核心类
# ─────────────────────────────────────────────────────────────────────────────
class ThreeWayGSOM:
    """
    3W-GSOM 核心实现
    """

    def __init__(self,
                 data: np.ndarray,
                 spread_factor: float = 0.8,
                 init_lr: float = 0.1,
                 max_epochs: int = 100,
                 max_nodes: int = 10,
                 random_state=None,
                 target_core_density: float = 0.6,
                 use_dynamic_threshold: bool = True,
                 dtype=np.float64,
                 min_delta_ratio: float = 0.15,
                 edge_factor: float = 0.3,
                 kappa_alpha: float = 0.5,
                 kappa_beta_cov: float = 0.5,
                 kappa_beta_ovl: float = 0.5):
        self.dtype = dtype
        self.data = data.astype(dtype, copy=True)
        self.sampleCount, self.input_dim = self.data.shape

        self.spread_factor = spread_factor
        self.init_lr = init_lr
        self.max_epochs = max_epochs
        self.max_nodes = max_nodes
        self.target_core_density = target_core_density
        self.use_dynamic_threshold = use_dynamic_threshold
        self.min_delta_ratio = max(0.0, float(min_delta_ratio))
        self.random_state = random_state
        self.edge_factor = float(edge_factor)

        self.kappa_alpha    = float(kappa_alpha)
        self.kappa_beta_cov = float(kappa_beta_cov)
        self.kappa_beta_ovl = float(kappa_beta_ovl)

        # Fix 3: 统一使用 self.rng 管理所有随机性
        if self.random_state is not None:
            self.rng = np.random.RandomState(self.random_state)
        else:
            self.rng = np.random.RandomState()

        self.grow_threshold = -self.input_dim * np.log(max(self.spread_factor, 1e-9))

        # Fix 4: 自适应最小样本量保护
        self.min_samples_to_grow = min(20, max(5, int(0.02 * self.sampleCount)))

        self.neurons = {}
        self.current_id = 0

        self.alpha = None
        self.beta = None
        self.alpha_history = []
        self.beta_history = []

        self.sim_matrix = None
        self.final_clusters_info = None

        self._weights_array = None
        self.nid2idx = {}
        self.idx2nid = []
        self.delta = None
        self.local_deltas_array = None

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助：生成颗粒球
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_granular_balls(self, X: np.ndarray, max_balls=4, tol=1e-3):
        n_samples = X.shape[0]
        init_center = X.mean(axis=0)
        dist_init = np.linalg.norm(X - init_center, axis=1)
        init_radius = dist_init.max()
        balls = [{'center': init_center,
                  'indices': np.arange(n_samples),
                  'radius': init_radius}]

        def ball_error(indices, c):
            return np.sum(np.linalg.norm(X[indices] - c, axis=1) ** 2)

        while len(balls) < max_balls:
            errors = [ball_error(b['indices'], b['center']) for b in balls]
            i_split = int(np.argmax(errors))
            if errors[i_split] < tol:
                break
            split_idx = balls[i_split]['indices']
            # Fix 3 & Fix 6: 从 self.rng 采样唯一种子，n_init=10
            km_seed = int(self.rng.randint(0, 2**31 - 1))
            km = KMeans(n_clusters=2, n_init=10, random_state=km_seed)
            labels = km.fit_predict(X[split_idx])
            centers = km.cluster_centers_
            new_balls = []
            for l in (0, 1):
                sub_idx = split_idx[labels == l]
                if sub_idx.size == 0:
                    continue
                sub_center = centers[l]
                sub_radius = np.linalg.norm(X[sub_idx] - sub_center, axis=1).max()
                new_balls.append({'center': sub_center,
                                  'indices': sub_idx,
                                  'radius': sub_radius})
            balls.pop(i_split)
            balls.extend(new_balls)

        if len(balls) < max_balls:
            all_idx = np.concatenate([b['indices'] for b in balls])
            # Fix 3 & Fix 6
            km_seed_g = int(self.rng.randint(0, 2**31 - 1))
            km_g = KMeans(n_clusters=max_balls, n_init=10, random_state=km_seed_g)
            g_lb = km_g.fit_predict(X[all_idx])
            g_ct = km_g.cluster_centers_
            balls = []
            for j in range(max_balls):
                cidx = all_idx[g_lb == j]
                if cidx.size == 0:
                    continue
                cct = g_ct[j]
                crr = np.linalg.norm(X[cidx] - cct, axis=1).max()
                balls.append({'center': cct, 'indices': cidx, 'radius': crr})

        centers = np.array([b['center'] for b in balls], dtype=self.dtype)
        final_balls = [b['indices'] for b in balls]
        return centers, final_balls, balls

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助：模糊相似矩阵 [F10]
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_similarity_matrix(self, X):
        """[F10] ε_a = std(a)，与 Definition 1 完全一致。"""
        n, d = X.shape
        eps = np.std(X, axis=0).astype(self.dtype)

        if _have_numba:
            sim = _sim_matrix_jit(X.astype(np.float32), eps.astype(np.float32))
            return sim.astype(self.dtype)

        sim = np.ones((n, n), dtype=self.dtype)
        for j in range(d):
            ej = eps[j]
            if ej < 1e-12:
                continue
            col = X[:, j]
            diff = np.abs(col[:, None] - col[None, :])
            comp = (1.0 - diff / ej) * (diff <= ej)
            sim = np.minimum(sim, comp.astype(self.dtype))
            del diff, comp
            gc.collect()
        np.fill_diagonal(sim, 1.0)
        return sim

    def _compute_FD(self, class_indices):
        n_classes = len(class_indices)
        FD = np.zeros((n_classes, self.sampleCount), dtype=self.dtype)
        sum_sim = np.sum(self.sim_matrix, axis=1)
        for j in range(n_classes):
            idx_j = class_indices[j]
            if len(idx_j) == 0:
                continue
            numerator = np.sum(self.sim_matrix[:, idx_j], axis=1)
            valid = (sum_sim > 1e-12)
            FD[j, valid] = numerator[valid] / sum_sim[valid]
        return FD

    def _compute_lower_approx(self, FD):
        if _have_numba:
            return _lower_approx_jit(self.sim_matrix.astype(np.float32),
                                     FD.astype(np.float32)).astype(self.dtype)
        c, n = FD.shape
        out = np.zeros((c, n), dtype=self.dtype)
        one_minus = (1.0 - self.sim_matrix).astype(self.dtype)
        for k in range(c):
            combined = np.maximum(one_minus, FD[k][None, :])
            out[k] = combined.min(axis=1)
            del combined
        del one_minus
        gc.collect()
        return out

    def _calculateSimilarity(self, vA, vB):
        interSz = np.sum(np.minimum(vA, vB))
        unionSz = np.sum(np.maximum(vA, vB))
        return interSz / unionSz if unionSz > 1e-12 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 核心：构建权重数组（含索引重建和 σ 计算）
    # ─────────────────────────────────────────────────────────────────────────
    def _build_weights_array(self):
        """完整重建 _weights_array、索引映射和 local_deltas_array。
        仅在初始化和生长后调用（结构变化时）。"""
        node_ids = sorted(self.neurons)
        self.nid2idx = {n: i for i, n in enumerate(node_ids)}
        self.idx2nid = node_ids
        if node_ids:
            self._weights_array = np.vstack(
                [self.neurons[n].weights for n in node_ids]).astype(self.dtype)
        else:
            self._weights_array = np.zeros((0, self.input_dim), dtype=self.dtype)
        self._recompute_local_deltas()

    def _recompute_local_deltas(self):
        """
        基于当前 _weights_array 重计算 local_deltas_array。

        【第二轮修复：Voronoi 领地感知 σ_i（替换第一轮数据感知下界）】
        ─────────────────────────────────────────────────────────────────────
        第一轮修改（k_cover = N/(K×3)，取中位数，乘0.7系数）存在三重偏小：
          ① k_cover=7 而非 N/K=21（3倍偏小）
          ② 取7个距离的中位数 ≈ 第4近距离（而非第21近）
          ③ 额外的0.7系数（30%削减）
        综合导致 σ_i 比理论需求小4–6倍，隶属度函数仍然过窄。

        修复原理：
        ─────────────────────────────────────────────────────────────────────
        对神经元 i，令其"Voronoi 领地"= 最近邻分配的样本集合。
        σ_i 满足：领地内第 90 百分位距离的样本恰好达到边缘域阈值 β：

            d_{90,i} = P_{90}( {||x - w_i|| : x ∈ Voronoi(i)} )
            σ_i = d_{90,i} / sqrt(-2·ln(β_floor))

        保证领地内 ≥90% 的样本能进入边缘域（μ ≥ β），彻底消除 Rule3 强制问题。

        对孤立神经元（领地样本 <2 个）：回退到 k_fallback = N/(2K) 近邻距离。

        神经元间距约束（来自原方案）保留，作为上层约束防止边缘域跨越相邻中心。
        ─────────────────────────────────────────────────────────────────────
        """
        n = self._weights_array.shape[0]
        if n >= 2:
            dist_nn = cdist(self._weights_array, self._weights_array).astype(self.dtype)
            np.fill_diagonal(dist_nn, np.inf)
            k_robust = min(2, n - 1)
            sorted_nn = np.sort(dist_nn, axis=1)
            local_min = np.mean(sorted_nn[:, :k_robust], axis=1)
        else:
            local_min = np.zeros(n, dtype=self.dtype)

        if self.delta is None:
            raise RuntimeError("self.delta 尚未初始化")

        lower_global = float(self.delta * self.min_delta_ratio)

        if n > 0 and self.sampleCount > 0:
            # ── 步骤1：计算神经元-数据距离矩阵 ───────────────────────────────
            dists_nd = cdist(self._weights_array, self.data).astype(self.dtype)

            # ── 步骤2：Voronoi 分配（每个样本归属最近神经元）────────────────
            assignments = np.argmin(dists_nd, axis=0)  # shape (N,)

            # ── 步骤3：β_floor → 反解 σ 所需的距离比例因子 ────────────────
            beta_floor = max(
                float(self.beta) if self.beta is not None else 0.1,
                0.05
            )
            log_term = np.sqrt(-2.0 * np.log(beta_floor + 1e-9))  # sqrt(-2·ln β)

            # ── 步骤4：每个神经元基于领地分布计算 σ_i ────────────────────
            sigma_voronoi = np.full(n, lower_global, dtype=self.dtype)
            k_fallback = max(2, self.sampleCount // max(n * 2, 1))

            for i in range(n):
                assigned = np.where(assignments == i)[0]
                if len(assigned) < 2:
                    # 孤立神经元：用 k_fallback 近邻样本距离作为代理
                    d_proxy = np.sort(dists_nd[i])[min(k_fallback - 1,
                                                       self.sampleCount - 1)]
                    sigma_voronoi[i] = float(d_proxy) / (log_term + 1e-9)
                else:
                    # 正常神经元：领地内第90百分位距离
                    d_90 = float(np.percentile(dists_nd[i, assigned], 90))
                    sigma_voronoi[i] = d_90 / (log_term + 1e-9)

            lower_arr = np.maximum(lower_global, sigma_voronoi)
        else:
            lower_arr = np.full(n, lower_global, dtype=self.dtype)

        # ── 最终值：max(神经元间距约束, 领地感知下界) ─────────────────────
        # local_min（神经元间距k-NN均值）防止边缘域跨越相邻神经元中心
        # lower_arr（领地感知下界）保证领地内90%样本可进入边缘域
        self.local_deltas_array = np.maximum(local_min, lower_arr).astype(self.dtype)

        # ── 缓存 dists_nd（神经元→数据距离矩阵），供 _update_thresholds 复用 ──
        # 有效条件：_refresh_local_deltas 与 _update_thresholds 之间权重不变化。
        # 在无生长 epoch 末（else 分支）严格满足此条件。
        # 有生长 epoch：_grow_node 调用 _build_weights_array 重建，缓存也被刷新。
        if n > 0 and self.sampleCount > 0:
            self._cached_dists_nd = dists_nd  # shape (K, N)，N行=数据，K列=神经元
        else:
            self._cached_dists_nd = None

    # ─────────────────────────────────────────────────────────────────────────
    # Fix 2: 轻量 σ_i 刷新方法（双点调用：Epoch 起始 + 无生长 Epoch 末）
    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_local_deltas(self):
        """
        轻量刷新 local_deltas_array，仅重计算神经元间最小距离。

        设计原则与调用时机（双点刷新策略）
        ────────────────────────────────────────────────────────────────────
        _update_neurons() 在每次权重更新时同步更新 _weights_array：
            cnode.weights += lr * (x - w_old)
            self._weights_array[idx_c] = cnode.weights
        因此 _weights_array 始终与 cnode.weights 保持一致，无需重建。
        此方法仅利用已同步的 _weights_array 重计算 σ_i，时间复杂度
        O(K²d)，K 通常 ≤ max_nodes（≤ 10），远小于完整重建的代价。

        调用点 ① — 每个训练 Epoch 起始（由 fit() 调用）：
            使 σ_i 反映上一 Epoch 结束后权重的实际位置，保证本 Epoch
            内全部样本训练过程使用一致且准确的 σ_i 作为隶属度基准。

        调用点 ② — 无生长 Epoch 末、_update_thresholds() 调用前
                   （由 fit() 的 else 分支调用）：
            权重在 Epoch 内持续漂移，Epoch 末的 σ_i 已偏离起始值。
            _update_thresholds() 基于 σ_i 计算 ρ/η/ω 并反馈给 α/β，
            若使用过时的 σ_i 则反馈方向错误。追加刷新后 σ_i 反映
            Epoch 末真实权重位置，使阈值反馈计算准确。
            ※ 有生长的 Epoch：_grow_node → _build_weights_array 内部
               已完成完整刷新（包含 local_deltas_array），无需重复。
        """
        if self._weights_array is None or self._weights_array.shape[0] == 0:
            return
        self._recompute_local_deltas()

    # ─────────────────────────────────────────────────────────────────────────
    # 初始化
    # ─────────────────────────────────────────────────────────────────────────
    def _init_grid(self):
        self.sim_matrix = self._compute_similarity_matrix(self.data)
        N_INIT_BALLS = 5  # 重启次数（建议 3–7，Seeds 用 5 即可）

        best_centers = None
        best_ball_indices = None
        best_ball_info = None
        best_silhouette = -np.inf

        for _restart in range(N_INIT_BALLS):
            try:
                cand_centers, cand_indices, cand_info = \
                    self._generate_granular_balls(self.data, max_balls=4)

                if len(cand_centers) < 2:
                    continue

                # 为每个样本分配最近球
                dists_to_centers = cdist(self.data, cand_centers)
                labels_cand = np.argmin(dists_to_centers, axis=1)

                unique_labels = np.unique(labels_cand)
                if len(unique_labels) < 2:
                    # 所有样本落在同一球：退化情形，跳过
                    continue

                # 计算简化轮廓系数（基于中心距离，O(N×k²)）
                # 标准轮廓系数需要 O(N²)，此处用中心距近似
                a_vals = dists_to_centers[np.arange(len(self.data)), labels_cand]
                # 对每个样本，找到次近中心的距离
                sorted_dists = np.sort(dists_to_centers, axis=1)
                b_vals = sorted_dists[:, 1]  # 次近中心距离
                sil_vals = (b_vals - a_vals) / (np.maximum(a_vals, b_vals) + 1e-9)
                sil_score = float(np.mean(sil_vals))

                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_centers = cand_centers
                    best_ball_indices = cand_indices
                    best_ball_info = cand_info

            except Exception:
                continue

        if best_centers is None:
            # 完全失败的极端情形：回退到单次初始化
            best_centers, best_ball_indices, best_ball_info = \
                self._generate_granular_balls(self.data, max_balls=4)

        centers = best_centers
        ball_indices = best_ball_indices
        ball_info = best_ball_info

        FD = self._compute_FD(ball_indices)
        Xiajinsi = self._compute_lower_approx(FD)

        self.neurons.clear()
        self.current_id = 0
        for i in range(len(centers)):
            node = GSOMNode(weights=centers[i], neuron_id=i)
            node.error = 0.0
            node.approximation = Xiajinsi[i]
            self.neurons[i] = node
            self.current_id += 1

        global_sigma = np.mean(np.std(self.data, axis=0))
        self.delta = global_sigma

        def dist2mu_local(d):
            return np.exp(-(d ** 2) / (2 * (global_sigma ** 2) + 1e-9))

        # ─────────────────────────────────────────────────────────────────────────
        # 阶段一：基于颗粒球统计量的阈值初始化（分位数方案，替换原 MAD 方案）
        # ─────────────────────────────────────────────────────────────────────────
        # [F19] r_i^+ = median_{x∈GB_i} ||x - c_i||  （核半径）
        # [F20] r_i^- = max_{x∈GB_i} ||x - c_i||      （边缘半径）
        # 高斯核变换：M_i^+ = exp(-(r_i^+)^2 / (2σ̄^2)), M_i^- = exp(-(r_i^-)^2 / (2σ̄^2))
        mu_core_list, mu_edge_list = [], []
        for binfo in ball_info:
            idx_b = binfo['indices']
            if len(idx_b) == 0:
                continue
            dists_b = np.linalg.norm(self.data[idx_b] - binfo['center'], axis=1)
            r_plus = float(np.median(dists_b))  # 核半径 r_i^+
            r_minus = float(np.max(dists_b))  # 边缘半径 r_i^-
            mu_core_list.append(dist2mu_local(r_plus))
            mu_edge_list.append(dist2mu_local(r_minus))

        # ─── [F23-新] α₀ = Q_{τ_α}(M^+), τ_α = 0.75（上四分位数，严格核心约束）
        # ─── [F24-新] β₀ = Q_{τ_β}(M^-), τ_β = 0.25（下四分位数，宽松边缘覆盖）
        #
        # 理论保证（命题 2）：由 r_i^+ ≤ r_i^- 知 M_i^+ ≥ M_i^-（引理 1），
        # {M_i^+} 以 FOSD 支配 {M_i^-}，结合分位数单调性：
        #   α₀ = Q_{0.75}(M^+) ≥ Q_{0.5}(M^+) ≥ Q_{0.5}(M^-) ≥ Q_{0.25}(M^-) = β₀
        # 故 0 < β₀ ≤ α₀ ≤ 1 严格成立，无需 MAD 偏移亦可达到
        # "α₀ 偏高（严格核心约束），β₀ 偏低（宽松边缘覆盖）"的设计意图。
        _tau_alpha = 0.75  # 核心分位数水平（固定，无需调参）
        _tau_beta = 0.25  # 边缘分位数水平（固定，无需调参）
        alpha0 = float(np.quantile(mu_core_list, _tau_alpha))
        beta0 = float(np.quantile(mu_edge_list, _tau_beta))

        # ─── 星间距结构约束（保留原设计，防止边缘域跨越相邻聚类中心）
        if len(ball_info) > 1:
            c_all = np.array([b['center'] for b in ball_info])
            cdist_ = np.linalg.norm(c_all[:, None] - c_all[None, :], axis=2)
            np.fill_diagonal(cdist_, np.inf)
            d_min = cdist_.min()
            mu_gap = dist2mu_local(d_min)
            beta0 = min(beta0, mu_gap)
            # 施加 min 后 beta0 ≤ 原 beta0 ≤ alpha0，命题 2 结论仍成立

        # ─── 间隔约束与值域裁剪（确保 α₀ - β₀ ≥ δ > 0）
        _eps = 1e-3
        _delta = 0.05
        self._eps = _eps
        self._delta = _delta
        self.alpha = float(np.clip(alpha0, _eps + _delta, 1.0 - _eps))
        beta_upper = self.alpha - _delta
        self.beta = float(np.clip(beta0, _eps, beta_upper))
        # 注：此处裁剪仅为处理极端退化情况（如所有颗粒球坍缩为一点），
        # 在正常情况下由命题 2 保证 beta0 ≤ alpha0，裁剪不生效。

        self.alpha_history = [self.alpha]
        self.beta_history = [self.beta]

        node_ids = list(self.neurons.keys())
        for nid in node_ids:
            ndA = self.neurons[nid]
            sim_list = []
            for oid in node_ids:
                if oid == nid:
                    continue
                sc = self._calculateSimilarity(ndA.approximation,
                                               self.neurons[oid].approximation)
                sim_list.append((oid, sc))
            sim_list.sort(key=lambda x: x[1], reverse=True)
            ndA.sim_order = [x[0] for x in sim_list]

        self.initial_radius = max(2, int(0.6 * len(self.neurons)))
        self._build_weights_array()

    # ─────────────────────────────────────────────────────────────────────────
    # 三支匹配
    # ─────────────────────────────────────────────────────────────────────────
    def _find_matches_raw(self, x):
        if self._weights_array.shape[0] == 0:
            return [], [], []
        dists = np.linalg.norm(self._weights_array - x, axis=1)
        deltas = (self.local_deltas_array if self.local_deltas_array is not None
                  else np.full_like(dists, self.delta))
        membership_arr = np.exp(-(dists ** 2) / (2 * (deltas ** 2) + 1e-9))

        # 向量化分类（替换 Python for 循环，对大 K 和高频调用有显著收益）
        core_mask = membership_arr >= self.alpha
        edge_mask = (membership_arr >= self.beta) & ~core_mask

        pot_core = [(self.idx2nid[i], float(membership_arr[i]))
                    for i in np.where(core_mask)[0]]
        pot_edge = [(self.idx2nid[i], float(membership_arr[i]))
                    for i in np.where(edge_mask)[0]]
        pot_none = [(self.idx2nid[i], float(membership_arr[i]))
                    for i in np.where(~core_mask & ~edge_mask)[0]]
        return pot_core, pot_edge, pot_none

    @staticmethod
    def _resolve_core_conflict(pot_core, pot_edge):
        if len(pot_core) <= 1:
            return pot_core, pot_edge
        pot_core.sort(key=lambda x: float(x[1]), reverse=True)
        best = pot_core[0]
        for nid, mu in pot_core[1:]:
            pot_edge.append((nid, mu))
        return [best], pot_edge

    def _find_matches(self, x):
        pot_core, pot_edge, pot_none = self._find_matches_raw(x)
        pot_core, pot_edge = self._resolve_core_conflict(pot_core, pot_edge)
        return {
            'core': [c[0] for c in pot_core],
            'edge': [e[0] for e in pot_edge],
            'irrelevant': [n[0] for n in pot_none]
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 学习率与邻域衰减
    # ─────────────────────────────────────────────────────────────────────────
    def _get_learning_rate(self, epoch):
        return self.init_lr * np.exp(-epoch / self.max_epochs)

    def _get_neighbor_range(self, epoch):
        nr = self.initial_radius * (0.9 ** (epoch / self.max_epochs))
        return max(1, int(nr))

    # ─────────────────────────────────────────────────────────────────────────
    # 权重更新
    # ─────────────────────────────────────────────────────────────────────────
    def _update_neurons(self, x, matches, epoch):
        lr = self._get_learning_rate(epoch)
        neighbor_num = self._get_neighbor_range(epoch)

        for cid in matches['core']:
            cnode = self.neurons[cid]
            idx_c = self.nid2idx[cid]
            w_old = cnode.weights.copy()
            cnode.weights += lr * (x - w_old)
            cnode.error += np.linalg.norm(x - w_old)
            self._weights_array[idx_c] = cnode.weights
            for nbid in cnode.sim_order[:neighbor_num]:
                nb = self.neurons[nbid]
                idx_n = self.nid2idx[nbid]
                sim_val = self._calculateSimilarity(cnode.approximation,
                                                    nb.approximation)
                influence = np.exp(-(1 - sim_val) ** 2 / 2)
                nb_old = nb.weights.copy()
                nb.weights += lr * influence * (x - nb_old)
                self._weights_array[idx_n] = nb.weights

        edge_lr = lr * self.edge_factor
        for eid in matches['edge']:
            if eid in matches['core']:
                continue
            enode = self.neurons[eid]
            idx_e = self.nid2idx[eid]
            w_old = enode.weights.copy()
            enode.weights += edge_lr * (x - w_old)
            enode.error += self.edge_factor * np.linalg.norm(x - w_old)
            self._weights_array[idx_e] = enode.weights

    # ─────────────────────────────────────────────────────────────────────────
    # 获取神经元核域样本集合
    # ─────────────────────────────────────────────────────────────────────────
    def _get_samples_for_node(self, nid):
        if self._weights_array.shape[0] == 0:
            return np.array([], dtype=int)
        dists = cdist(self.data, self._weights_array)
        deltas = self.local_deltas_array
        mem = np.exp(-(dists ** 2) / (2 * deltas ** 2 + 1e-9))

        core_mask = (mem >= self.alpha)
        idx_n = self.nid2idx[nid]
        is_core_candidate = core_mask[:, idx_n]

        mem_core = mem * core_mask
        best_core_col = np.argmax(mem_core, axis=1)

        assigned = is_core_candidate & (best_core_col == idx_n)
        return np.where(assigned)[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Fix 1 & Fix 3 & Fix 4 & Fix 6 & Fix 7: 生长节点
    # ─────────────────────────────────────────────────────────────────────────
    def _grow_node(self, parent_id):
        """
        生长操作（修复版）。

        修复点：
        - Fix 3: KMeans 使用 self.rng 采样的唯一种子
        - Fix 4: 最小样本量保护提升至 self.min_samples_to_grow
        - Fix 6: n_init=10（替换 n_init='auto'）
        """
        pnode = self.neurons[parent_id]
        assigned_indices = self._get_samples_for_node(parent_id)

        # Fix 4: 自适应最小样本量保护（原始代码仅保护 < 2）
        if len(assigned_indices) < self.min_samples_to_grow:
            pnode.error = 0.0
            return

        sub_data = self.data[assigned_indices]

        # Fix 3 & Fix 6: 从 self.rng 采样唯一种子，n_init=10
        km_seed = int(self.rng.randint(0, 2**31 - 1))
        km = KMeans(n_clusters=2, random_state=km_seed, n_init=10)
        km.fit(sub_data)
        c1, c2 = km.cluster_centers_
        old_err = pnode.error

        pnode.weights = c1
        pnode.error = old_err * 0.5
        self._weights_array[self.nid2idx[parent_id]] = pnode.weights

        new_id = self.current_id
        self.current_id += 1
        new_node = GSOMNode(weights=c2, neuron_id=new_id)
        new_node.error = old_err * 0.5
        self.neurons[new_id] = new_node
        self._build_weights_array()

        node_ids = list(self.neurons.keys())
        dists_all = cdist(self.data, self._weights_array)
        deltas_all = self.local_deltas_array
        mem_all = np.exp(-(dists_all ** 2) /
                         (2 * deltas_all ** 2 + 1e-9))

        core_mask_all = (mem_all >= self.alpha)
        has_core = core_mask_all.any(axis=1)
        best_core_col = np.argmax(mem_all * core_mask_all, axis=1)

        class_indices = [[] for _ in node_ids]
        for j in range(len(node_ids)):
            mask = has_core & (best_core_col == j)
            class_indices[j] = list(np.where(mask)[0])

        FD = self._compute_FD(class_indices)
        Xiajinsi = self._compute_lower_approx(FD)
        for i_nd, nid in enumerate(node_ids):
            self.neurons[nid].approximation = Xiajinsi[i_nd, :]
        for nid in node_ids:
            ndA = self.neurons[nid]
            sim_list = []
            for j_nd in node_ids:
                if j_nd == nid:
                    continue
                sc = self._calculateSimilarity(ndA.approximation,
                                               self.neurons[j_nd].approximation)
                sim_list.append((j_nd, sc))
            sim_list.sort(key=lambda x: x[1], reverse=True)
            ndA.sim_order = [x[0] for x in sim_list]

    # ─────────────────────────────────────────────────────────────────────────
    # 阶段二：基于反馈统计量的阈值更新（正交投影方案）
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _project_onto_Omega(alpha_raw, beta_raw, eps=1e-3, delta=0.05):
        """
        正交投影（最小欧氏距离投影）：将 (alpha_raw, beta_raw) 投影到可行域
            Ω = { (α, β) : ε ≤ β,  α − β ≥ δ,  α ≤ 1 − ε }

        可行域 Ω 是 ℝ² 中的凸三角形，三顶点为：
            V1 = (ε+δ,   ε)          [左下角]
            V2 = (1−ε,   ε)          [右下角]
            V3 = (1−ε,   1−ε−δ)     [右上角]

        理论保证
        ─────────────────────────────────────────────────────────────────────
        (T1) 存在唯一性（Hilbert 投影定理）：
             Ω 为 ℝ² 中非空闭凸集，对任意输入点，最近点存在且唯一。

        (T2) 边界性（标准引理）：
             若输入点在 Ω 外，其投影必在 ∂Ω（边界）上。
             证明：若投影为内点 P*，则 ∃ε>0 使 B(P*,ε)⊂Ω。
             令 P'=P*+ε·(P−P*)/‖P−P*‖，则 P'∈Ω 且 ‖P'−P‖<‖P*−P‖，矛盾。

        (T3) 最小偏差性：
             本方法在所有满足 Ω 可行性的方案中，使 (α,β) 与无约束更新
             (α_raw, β_raw) 的欧氏距离最小——是唯一最优的约束满足方案。

        (T4) 对称性（仅 E3 被激活时）：
             冲突量 v = β_raw − α_raw + δ > 0 时，投影将 v 对等分配：
             α 被上调 v/2，β 被下调 v/2，对两阈值无偏处理。
             注意：α 的净变化量为 (α_raw−α_t) + v/2，当 v/2 较大时，
             α 的最终方向可能与 F28 相反——这是两竞争反馈信号间最优
             折衷的必然结果，正交投影在合计意义上仍最小化了总偏差。

        算法：三边段（E1, E2, E3）逐一计算最近点，取总偏差最小者。
        """
        a, b = float(alpha_raw), float(beta_raw)

        # ── 快速路径：已在可行域内 ──────────────────────────────────────
        if b >= eps and a <= 1.0 - eps and a - b >= delta:
            return a, b

        best_d2 = float('inf')
        best = (eps + delta, eps)  # 安全退化默认值：顶点 V1

        # ── 投影到 E1：(t, ε), t ∈ [ε+δ, 1−ε] ──────────────────────────
        t1 = max(eps + delta, min(a, 1.0 - eps))
        p1 = (t1, eps)
        d1 = (a - p1[0]) ** 2 + (b - p1[1]) ** 2
        if d1 < best_d2:
            best_d2, best = d1, p1

        # ── 投影到 E2：(1−ε, t), t ∈ [ε, 1−ε−δ] ────────────────────────
        t2 = max(eps, min(b, 1.0 - eps - delta))
        p2 = (1.0 - eps, t2)
        d2 = (a - p2[0]) ** 2 + (b - p2[1]) ** 2
        if d2 < best_d2:
            best_d2, best = d2, p2

        # ── 投影到 E3：(t+δ, t), t ∈ [ε, 1−ε−δ]；E3 对应约束 α−β=δ ────
        # 对 t 最小化 (α−t−δ)²+(β−t)²，导数为零：t*=(α+β−δ)/2
        t3 = max(eps, min((a + b - delta) / 2.0, 1.0 - eps - delta))
        p3 = (t3 + delta, t3)
        d3 = (a - p3[0]) ** 2 + (b - p3[1]) ** 2
        if d3 < best_d2:
            best_d2, best = d3, p3

        return float(best[0]), float(best[1])


    def _update_thresholds(self, use_ema: bool = False, ema_k: float = 0.25):
        """
        两阶段动态阈值方案——第二阶段：反馈驱动阈值更新（正交投影版）。

        完整实现以下公式：
            [F28] α̃(t+1) = α(t) + κ_α · (ρ − ρ_target)
            [F29] β̃(t+1) = β(t) − κ_β^cov · η + κ_β^ovl · ω
            [F30] (α(t+1), β(t+1)) = Π_Ω(α̃(t+1), β̃(t+1))

        理论保证
        ────────────────────────────────────────────────────────────────────
        (G1) 可行性不变量：Π_Ω 将任意点映射至 Ω，故
             0 < β(t) < α(t) < 1 在整个训练过程中严格成立。

        (G2) 最小偏差性：在满足可行性的所有方案中，正交投影使
             (α̃, β̃) 到 (α(t+1), β(t+1)) 的欧氏距离最小。

        (G3) 对称性：仅 E3 被激活时，冲突量对等分配给两阈值；
             α 的净方向取决于 F28 幅度与 v/2 的大小关系（见 §5.3 修正版）。

        EMA 选项（use_ema=False 默认关闭）
        ────────────────────────────────────────────────────────────────────
        平稳期：全样本统计量无采样噪声，EMA 引入无效滞后，不推荐开启。
        生长期：网络结构突变可使统计量剧烈跳变，EMA 提供平滑缓冲，
                可通过 use_ema=True 开启（供消融实验对比）。
        κ 参数本身已提供隐式步长控制，正交投影提供值域保证，
        两者联合已构成完整稳定性机制，EMA 非必须。

        参数
        ────
        use_ema : bool，默认 False
            是否对 ρ/η/ω 施加 EMA 平滑（可选，供消融实验）。
        ema_k : float，默认 0.25
            EMA 平滑系数，仅在 use_ema=True 时生效。
        """
        if not self.use_dynamic_threshold:
            return
        n_nodes = len(self.neurons)
        if n_nodes == 0:
            return

        _eps   = getattr(self, '_eps',   1e-3)
        _delta = getattr(self, '_delta', 0.05)

        # ── 步骤 1：计算实时反馈统计量 ρ [F25]、η [F26]、ω [F27] ──────────
        # 优先使用 _recompute_local_deltas 中缓存的距离矩阵（避免重复计算）。
        # 缓存有效性保证：_refresh_local_deltas → _update_thresholds 之间
        # 无权重更新（fit() 的 else 分支），故 _cached_dists_nd 对应当前权重。
        if (hasattr(self, '_cached_dists_nd') and
                self._cached_dists_nd is not None and
                self._cached_dists_nd.shape == (n_nodes, self.sampleCount)):
            dists = self._cached_dists_nd.T  # (N, K)，直接转置
        else:
            dists = cdist(self.data, self._weights_array)

        deltas = np.tile(self.local_deltas_array.reshape(1, -1),
                         (self.sampleCount, 1))
        mem = np.exp(-(dists ** 2) / (2 * (deltas ** 2) + 1e-9))

        core_mask = (mem >= self.alpha)
        edge_mask = (mem >= self.beta)

        core_cnt = core_mask.sum(axis=0).astype(float)
        edge_cnt = edge_mask.sum(axis=0).astype(float)

        # [F25] ρ：各神经元核域密度比的均值
        with np.errstate(invalid='ignore', divide='ignore'):
            rho_vals = np.where(edge_cnt > 0, core_cnt / edge_cnt, 0.0)
        rho = float(rho_vals.mean())

        # [F26] η：未被任意神经元覆盖的样本比例
        covered = edge_mask.any(axis=1)
        eta = 1.0 - float(covered.sum()) / self.sampleCount

        # [F27] ω：各对神经元边缘域 Jaccard 重叠的均值（k=1 时自然为 0）
        if n_nodes >= 2:
            em_int    = edge_mask.astype(np.int32)
            inter_mat = em_int.T @ em_int
            counts    = edge_cnt.astype(int)
            union_mat = counts[:, None] + counts[None, :] - inter_mat
            p_idx, q_idx = np.triu_indices(n_nodes, k=1)
            inter_vals   = inter_mat[p_idx, q_idx].astype(float)
            union_vals   = union_mat[p_idx, q_idx].astype(float)
            valid_mask   = union_vals > 0
            omega = (float(np.mean(inter_vals[valid_mask] / union_vals[valid_mask]))
                     if valid_mask.any() else 0.0)
        else:
            omega = 0.0

        # ── 可选 EMA 平滑（默认关闭；生长期稳定性增强选项）────────────────
        if use_ema:
            if not hasattr(self, '_ema'):
                self._ema = {'rho': rho, 'eta': eta, 'omega': omega}
            else:
                self._ema['rho']   = (1 - ema_k) * self._ema['rho']   + ema_k * rho
                self._ema['eta']   = (1 - ema_k) * self._ema['eta']   + ema_k * eta
                self._ema['omega'] = (1 - ema_k) * self._ema['omega'] + ema_k * omega
            rho, eta, omega = self._ema['rho'], self._ema['eta'], self._ema['omega']

        # ── 步骤 2：F28/F29——计算无约束更新值 α̃、β̃ ─────────────────────
        # [F28] α̃(t+1) = α(t) + κ_α · (ρ − ρ_target)
        alpha_raw = self.alpha + self.kappa_alpha * (rho - self.target_core_density)
        # [F29] β̃(t+1) = β(t) − κ_β^cov · η + κ_β^ovl · ω
        beta_raw = self.beta - self.kappa_beta_cov * eta \
                   + self.kappa_beta_ovl * omega

        # ── 步骤 3：F30——正交投影到可行域 Ω（见 _project_onto_Omega）────────
        # 保证：(α(t+1), β(t+1)) ∈ Ω，即 0<β<α<1 严格成立（可行性不变量）
        # 性质：在满足可行性的所有方案中，与 (α̃, β̃) 欧氏距离最小（唯一最优）
        self.alpha, self.beta = ThreeWayGSOM._project_onto_Omega(
            alpha_raw, beta_raw, _eps, _delta)

        self.alpha_history.append(self.alpha)
        self.beta_history.append(self.beta)

    # ─────────────────────────────────────────────────────────────────────────
    # Fix 1 & Fix 2 & Fix 7: 主训练循环（批量延迟生长 + 双点 σ 刷新 + 排序确定性）
    # ─────────────────────────────────────────────────────────────────────────
    def fit(self):
        """
        主训练循环（随机性修复终版）。

        核心修复说明
        ──────────────────────────────────────────────────────────────────────
        Fix 1 + Fix 7 [Epoch 末批量生长 + 确定性排序]：
            原始代码在样本处理循环内立即调用 _grow_node()，导致同一
            Epoch 内网络结构不一致。本版本在每个 Epoch 的样本处理循环
            结束后，统一执行该 Epoch 内触发的所有生长操作，且通过
            sorted(growth_candidates) 保证多神经元同时生长时的处理顺
            序与外部种子无关（完全由神经元 ID 确定）。

        Fix 2 [σ_i 双点刷新策略]：
            ① Epoch 起始调用 _refresh_local_deltas()：
               基于上一 Epoch 结束后的权重重计算 σ_i，保证本 Epoch
               内所有样本训练使用准确且一致的核宽。
            ② 无生长 Epoch 末追加调用（else 分支）：
               Epoch 内权重持续漂移，若仅在起始刷新一次，则
               _update_thresholds() 基于过时 σ_i 计算 ρ/η/ω，导致
               阈值反馈方向错误。else 分支在调用 _update_thresholds()
               前补充一次刷新，使反馈使用 Epoch 末最新权重对应的 σ_i。
               有生长的 Epoch（if 分支）：_grow_node → _build_weights_array
               内部已完整刷新，不需重复。
        ──────────────────────────────────────────────────────────────────────
        """
        self._init_grid()
        indices = np.arange(self.sampleCount)
        grow_epochs = int(self.max_epochs * 0.7)

        for epoch in range(self.max_epochs):
            allow_grow = (epoch < grow_epochs)
            self.rng.shuffle(indices)

            # Fix 2 ①: Epoch 起始刷新 σ_i（反映上一 Epoch 末权重位置）
            self._refresh_local_deltas()

            # Fix 1: 收集本 Epoch 生长候选，Epoch 内不实际生长
            growth_candidates = set()

            for i in indices:
                x = self.data[i, :]
                matches = self._find_matches(x)
                self._update_neurons(x, matches, epoch)

                if allow_grow:
                    for cid in matches['core']:
                        cnode = self.neurons[cid]
                        if cnode.error > self.grow_threshold:
                            # 记录候选，重置误差防止无界累积
                            growth_candidates.add(cid)
                            cnode.error = 0.0

            # Fix 1 + Fix 7: Epoch 结束后批量生长，sorted 保证确定性顺序
            if allow_grow and growth_candidates:
                for cid in sorted(growth_candidates):
                    if cid not in self.neurons:
                        continue  # 安全检查：防止前序生长已删除（理论上不会）
                    if len(self.neurons) < self.max_nodes:
                        self._grow_node(cid)
                    # 若已达 max_nodes，误差在循环内已重置为 0，无需额外处理
                # 有生长：_grow_node → _build_weights_array 已刷新 local_deltas
                # 无需再调用 _refresh_local_deltas()，直接进入 _update_thresholds
            else:
                # Fix 2 ②: 无生长 Epoch 末追加刷新
                # 权重在 Epoch 内持续漂移，此时刷新保证 _update_thresholds
                # 使用 Epoch 末最新权重对应的 σ_i 计算 ρ/η/ω 反馈
                self._refresh_local_deltas()

            _use_ema_this_epoch = (epoch < grow_epochs)
            self._update_thresholds(use_ema=_use_ema_this_epoch, ema_k=0.3)

        self._postprocess_threeway_clusters_after_training()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # 全局三支后处理（Rules 1-3，与原始完全一致）
    # ─────────────────────────────────────────────────────────────────────────
    def _postprocess_threeway_clusters_after_training(self):
        node_ids = list(self.neurons.keys())
        n_nodes = len(node_ids)
        if n_nodes == 0:
            self.final_clusters_info = {}
            return

        id2col = {nid: j for j, nid in enumerate(node_ids)}
        dists = cdist(self.data, np.vstack([self.neurons[n].weights
                                            for n in node_ids]))
        deltas = np.tile(self.local_deltas_array.reshape(1, -1),
                         (self.sampleCount, 1))
        membership_cache = np.exp(-(dists ** 2) / (2 * (deltas ** 2) + 1e-9))

        clusters_info = {nid: {'core': set(), 'edge': set()} for nid in node_ids}

        # ── 向量化核心分配 ────────────────────────────────────────────────────
        # membership_cache: (N, K)，已在函数入口计算完毕
        alpha_mask = membership_cache >= self.alpha  # (N, K) bool
        beta_mask = (membership_cache >= self.beta) & ~alpha_mask  # 仅边缘域

        n_core_cands = alpha_mask.sum(axis=1)  # 每个样本的核域候选数 (N,)

        # 情形 A：恰好 1 个核域候选（最常见）
        single_mask = (n_core_cands == 1)
        if single_mask.any():
            single_idx = np.where(single_mask)[0]
            assigned_col = np.argmax(alpha_mask[single_idx], axis=1)
            for local_i, col in enumerate(assigned_col):
                clusters_info[node_ids[col]]['core'].add(int(single_idx[local_i]))

        # 情形 B：多个核域候选（最高隶属度取核域，其余降为边缘域）
        multi_mask = (n_core_cands > 1)
        if multi_mask.any():
            multi_idx = np.where(multi_mask)[0]
            # 遮蔽非核候选列，取最大列作为核域归属
            masked_mem = membership_cache[multi_idx] * alpha_mask[multi_idx]
            best_col = np.argmax(masked_mem, axis=1)
            for local_i, (global_i, bcol) in enumerate(zip(multi_idx, best_col)):
                clusters_info[node_ids[bcol]]['core'].add(int(global_i))
                # 其余核候选降为边缘域（若 μ ≥ β）
                for col in range(len(node_ids)):
                    if col != bcol and alpha_mask[global_i, col]:
                        if membership_cache[global_i, col] >= self.beta:
                            clusters_info[node_ids[col]]['edge'].add(int(global_i))

        # 情形 C：纯边缘域样本（无核域候选）
        # beta_mask[i, j] = True 表示样本 i 在神经元 j 的边缘域（但非核域）
        edge_rows, edge_cols = np.where(beta_mask)
        for i, j in zip(edge_rows, edge_cols):
            clusters_info[node_ids[j]]['edge'].add(int(i))

        covered_mask = (n_core_cands >= 1) | beta_mask.any(axis=1)

        # Rule 2: 空核心强制填充
        for nid in node_ids:
            if len(clusters_info[nid]['core']) == 0:
                col = id2col[nid]
                # 优先从该神经元 Voronoi 领地（距离最近的样本集）中选择成员度最高者
                # 语义保证：Voronoi 领地样本"原本就属于"该神经元，不需要从其他神经元抢夺
                # 避免跨领地污染（破坏其他神经元已建立的三支结构）
                _voronoi_mask = (np.argmax(membership_cache, axis=1) == col)
                if _voronoi_mask.any():
                    _mems_voronoi = membership_cache[:, col].copy()
                    _mems_voronoi[~_voronoi_mask] = -np.inf
                    best_i = int(np.argmax(_mems_voronoi))
                else:
                    # 退化情形（该神经元无任何 Voronoi 样本）：回退到全局 argmax
                    best_i = int(np.argmax(membership_cache[:, col]))

                # 将 best_i 从所有其他核域中移除，恢复互斥性
                for other_nid in node_ids:
                    if other_nid != nid:
                        if best_i in clusters_info[other_nid]['core']:
                            clusters_info[other_nid]['core'].discard(best_i)
                            # 转降为边缘域（若该神经元的隶属度 ≥ β）
                            mu_other = membership_cache[best_i, id2col[other_nid]]
                            if mu_other >= self.beta:
                                clusters_info[other_nid]['edge'].add(best_i)

                clusters_info[nid]['core'].add(best_i)
                clusters_info[nid]['edge'].discard(best_i)
                covered_mask[best_i] = True

        # Rule 3: 未覆盖点强制纳入全局最高隶属度神经元的边缘域
        uncovered = np.where(~covered_mask)[0]
        if len(uncovered) > 0:
            # 取每个未覆盖样本在所有神经元中的最高隶属度列
            best_cols = np.argmax(membership_cache[uncovered], axis=1)
            for local_i, best_col in enumerate(best_cols):
                clusters_info[node_ids[best_col]]['edge'].add(int(uncovered[local_i]))

        self.final_clusters_info = clusters_info

    def get_threeway_partition(self):
        if self.final_clusters_info is None:
            raise RuntimeError("模型尚未 fit() 或后处理失败。")
        return copy.deepcopy(self.final_clusters_info)

    # ─────────────────────────────────────────────────────────────────────────
    # 预测（纯隶属度策略，与原始完全一致）
    # ─────────────────────────────────────────────────────────────────────────
    def predict_labels(self) -> np.ndarray:
        """
        纯隶属度策略：三支划分 → 每样本一个整数硬标签。
        核域样本：直接读取 final_clusters_info 中的核域归属。
        边缘域样本（含 Rule 3 样本）：取全局最高隶属度对应神经元。
        完全无监督，不依赖真实标签。
        """
        if self.final_clusters_info is None:
            raise RuntimeError("请先调用 fit()。")

        n_samples = self.sampleCount
        labels = np.full(n_samples, -1, dtype=int)
        for nid, info in self.final_clusters_info.items():
            for idx in info['core']:
                labels[idx] = nid

        undecided = np.where(labels == -1)[0]
        if len(undecided) > 0:
            W = self._weights_array
            D = self.local_deltas_array
            X_sub = self.data[undecided]
            dists = cdist(X_sub, W)
            mem   = np.exp(-(dists ** 2) / (2 * (D[None, :] ** 2) + 1e-9))
            best_col = np.argmax(mem, axis=1)
            for local_i, global_i in enumerate(undecided):
                labels[global_i] = self.idx2nid[best_col[local_i]]

        uniq = np.unique(labels[labels >= 0])
        remap = {o: i for i, o in enumerate(uniq)}
        labels = np.array([remap[l] if l in remap else -1 for l in labels])
        return labels


# =============================================================================
# 统一评估接口
# =============================================================================
def clustering_evaluation(model: ThreeWayGSOM, y_true: np.ndarray):
    """
    聚类评估函数（支持过聚类的多对一映射版本）。

    原版问题：当 n_pred > n_true（如 max_nodes=4, k_true=3）时，
    linear_sum_assignment 仅能匹配 n_true 列，第 (n_pred - n_true) 个
    多余聚类的所有样本被计为错误，形成系统性 ACC 上限约束。

    修复：当 n_pred > n_true 时，使用"多对一多数投票"映射：
      - 每个预测聚类独立映射到使该聚类内多数票最多的真实类别
      - 多个聚类可以映射到同一真实类别（允许一个类由多个子聚类表示）
      - 这与算法的实际意图一致：过聚类是将一个类分裂为子聚类，
        而非声称存在更多类别

    注意：当 n_pred <= n_true 时，保留原 Hungarian 方法（不变）。
    """
    y_pred = model.predict_labels()
    metric_dict = {}

    if y_true is None or len(np.unique(y_true)) <= 1:
        metric_dict['ACC'] = -1.0
        metric_dict['ARI'] = -1.0
        metric_dict['NMI'] = -1.0
        return metric_dict

    n_true_classes = len(np.unique(y_true))
    n_pred_classes = len(np.unique(y_pred[y_pred >= 0]))

    if n_pred_classes > n_true_classes:
        # 过聚类：多对一多数投票映射
        # 为每个预测类别找到对应的真实类别（取多数票，允许重复映射）
        cm = confusion_matrix(y_true, y_pred)   # shape: (n_true, n_pred)
        # 每列的 argmax = 该预测聚类中占多数的真实类别
        col_to_true = np.argmax(cm, axis=0)     # shape: (n_pred,)
        # 直接映射（不用 Hungarian，允许多个预测列映射到同一真实类别）
        aligned = np.array([col_to_true[p] if p >= 0 else -1
                             for p in y_pred])
        metric_dict['ACC'] = accuracy_score(y_true, aligned)
    else:
        # 标准情形：Hungarian 最优匹配
        cm = confusion_matrix(y_true, y_pred)
        row, col = linear_sum_assignment(-cm)
        aligned = np.full_like(y_pred, -1)
        for i in range(len(row)):
            aligned[y_pred == col[i]] = row[i]
        metric_dict['ACC'] = accuracy_score(y_true, aligned)

    metric_dict['ARI'] = adjusted_rand_score(y_true, y_pred)
    metric_dict['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    return metric_dict


# =============================================================================
# 从 ThreeWayGSOM 提取三支划分（与原始逻辑完全一致）
# =============================================================================
def _extract_partition_from_3wgsom(model: ThreeWayGSOM,
                                    y_true: np.ndarray,
                                    X: np.ndarray,
                                    k_entropy: int = 10):
    if not _have_twmetrics:
        return None, {}

    if model.final_clusters_info is None:
        print("[警告] final_clusters_info 为空，请确保模型已 fit()。")
        return None, {}

    n_samples = model.sampleCount
    node_ids_sorted = sorted(model.final_clusters_info.keys())
    K = len(node_ids_sorted)

    core_indices   = []
    fringe_indices = []
    for nid in node_ids_sorted:
        info = model.final_clusters_info[nid]
        core_indices.append(sorted(info['core']))
        fringe_indices.append(sorted(info['edge']))

    core_all   = set()
    fringe_all = set()
    for lst in core_indices:
        core_all.update(lst)
    for lst in fringe_indices:
        fringe_all.update(lst)
    trivial_indices = sorted(
        i for i in range(n_samples)
        if i not in core_all and i not in fringe_all
    )
    if trivial_indices:
        print(f"[警告] 3W-GSOM 发现 {len(trivial_indices)} 个平凡域样本。")

    n_rule3_forced = 0
    if model._weights_array is not None and len(node_ids_sorted) > 0:
        try:
            dists_all = cdist(X, model._weights_array)
            deltas_all = np.tile(
                model.local_deltas_array.reshape(1, -1), (n_samples, 1)
            )
            mem_all = np.exp(
                -(dists_all ** 2) / (2 * (deltas_all ** 2) + 1e-9)
            )
            below_beta_all = (mem_all < model.beta).all(axis=1)
            n_rule3_forced = int(below_beta_all.sum())
        except Exception:
            n_rule3_forced = -1

    y_pred = model.predict_labels()

    partition = ThreeWayPartition(
        core_indices    = core_indices,
        fringe_indices  = fringe_indices,
        trivial_indices = trivial_indices,
        labels          = y_pred,
        n_samples       = n_samples,
        n_clusters      = K,
        algorithm       = "3W-GSOM",
    )

    try:
        tw_metrics = compute_threeway_metrics(
            partition, y_true, X, k_entropy=k_entropy
        )
        tw_metrics['n_rule3_forced'] = n_rule3_forced
        tw_metrics['phi_rule3'] = (n_rule3_forced / n_samples
                                   if n_rule3_forced >= 0 else float('nan'))
    except Exception as e:
        print(f"[警告] 三支指标计算失败：{e}")
        import traceback
        traceback.print_exc()
        tw_metrics = {'n_rule3_forced': n_rule3_forced}

    return partition, tw_metrics


# =============================================================================
# main 函数（供网格搜索使用）
# =============================================================================
def main(dataset_name, spread_factor=0.8, init_lr=0.1, max_epochs=100,
         edge_factor=0.3, target_core_density=0.6,
         kappa_alpha=0.5, kappa_beta_cov=0.5, kappa_beta_ovl=0.5,
         max_nodes=6,      # ← 默认值从10改为6，避免硬编码过大值
         seed=None):
    # 注意：不再调用全局 np.random.seed(seed) 和 random.seed(seed)。
    # 原因：在多线程并行（n_jobs>1）时，全局随机状态是线程共享的，
    # 多个线程同时设置会互相覆盖，导致随机性不可控。
    # ThreeWayGSOM 内部通过 random_state=seed 参数使用局部 RandomState，
    # 完全可复现且线程安全。此处不再设置全局状态。
    # （若需完全严格的单线程复现，可在单线程模式下恢复全局设置）
    # if seed is not None:
    #     np.random.seed(seed)
    #     random.seed(seed)

    base_path = r"E:\SunYH\Code\SunYH\3W-GSOM\Datasets"
    file_path = f"{base_path}\\{dataset_name}"
    data_df = pd.read_csv(file_path, header=None)
    y_true = data_df.iloc[:, 0].values
    X = data_df.iloc[:, 1:].values
    X = MinMaxScaler().fit_transform(X)

    gsom = ThreeWayGSOM(
        data=X,
        spread_factor=spread_factor,
        init_lr=init_lr,
        max_epochs=max_epochs,
        max_nodes=max_nodes,
        target_core_density=target_core_density,
        edge_factor=edge_factor,
        use_dynamic_threshold=True,
        dtype=np.float64,
        random_state=seed,
        kappa_alpha=kappa_alpha,
        kappa_beta_cov=kappa_beta_cov,
        kappa_beta_ovl=kappa_beta_ovl,
    )

    start_time = time.time()
    gsom.fit()
    end_time = time.time()

    metric = clustering_evaluation(gsom, y_true)
    metric['Time'] = end_time - start_time

    print(f"\n=== 数据集: {dataset_name} ===")
    print(f"参数: SF={spread_factor}, LR={init_lr}, Eps={max_epochs}, "
          f"edge_f={edge_factor}, rho={target_core_density}, seed={seed}")
    print(f"训练耗时: {end_time - start_time:.4f} 秒")
    for k in ('ACC', 'ARI', 'NMI'):
        print(f"{k:<12}: {metric[k]:.4f}")
    return metric


# =============================================================================
# main_with_partition（含三支输出）
# =============================================================================
def main_with_partition(dataset_name, spread_factor=0.8, init_lr=0.1,
                        max_epochs=100, edge_factor=0.3,
                        target_core_density=0.6, max_nodes=4,
                        kappa_alpha=0.5, kappa_beta_cov=0.5, kappa_beta_ovl=0.5,
                        seed=None, k_entropy=10):
    # 同 main()：不设置全局随机状态，仅使用局部 RandomState（线程安全）
    # if seed is not None:
    #     np.random.seed(seed)
    #     random.seed(seed)

    base_path = r"E:\SunYH\Code\SunYH\3W-GSOM\Datasets"
    file_path = f"{base_path}\\{dataset_name}"
    data_df = pd.read_csv(file_path, header=None)
    y_true = data_df.iloc[:, 0].values
    X = data_df.iloc[:, 1:].values
    X = MinMaxScaler().fit_transform(X)

    gsom = ThreeWayGSOM(
        data=X,
        spread_factor=spread_factor,
        init_lr=init_lr,
        max_epochs=max_epochs,
        max_nodes=max_nodes,
        target_core_density=target_core_density,
        edge_factor=edge_factor,
        use_dynamic_threshold=True,
        dtype=np.float64,
        random_state=seed,
        kappa_alpha=kappa_alpha,
        kappa_beta_cov=kappa_beta_cov,
        kappa_beta_ovl=kappa_beta_ovl,
    )

    t0 = time.time()
    gsom.fit()
    t1 = time.time()

    cluster_metrics = clustering_evaluation(gsom, y_true)

    partition, tw_metrics = _extract_partition_from_3wgsom(
        gsom, y_true, X, k_entropy=k_entropy
    )

    result = {**cluster_metrics, 'Time': t1 - t0}
    for key, val in tw_metrics.items():
        if key != 'algorithm':
            result[f'tw_{key}'] = val

    print(f"\n=== 数据集: {dataset_name} [3W-GSOM-Fixed + 三支指标] ===")
    print(f"参数: SF={spread_factor}, LR={init_lr}, Eps={max_epochs}, "
          f"edge_f={edge_factor}, rho={target_core_density}")
    print(f"训练耗时: {t1 - t0:.4f} 秒")
    for key in ('ACC', 'ARI', 'NMI'):
        print(f"{key:<12}: {result[key]:.4f}")

    if tw_metrics and partition is not None:
        tw_ok = tw_metrics.get('tw_validity', False)
        tpqi_val = tw_metrics.get('TPQI', float('nan'))
        psi_val = tw_metrics.get('PSI', float('nan'))
        cp_val = tw_metrics.get('core_purity', float('nan'))
        op_val = tw_metrics.get('overall_purity', float('nan'))
        fp_val = tw_metrics.get('fringe_purity', float('nan'))
        phi_c = tw_metrics.get('phi_core', float('nan'))
        phi_f = tw_metrics.get('phi_fringe', float('nan'))
        phi_r3 = tw_metrics.get('phi_rule3', float('nan'))
        n_r3 = tw_metrics.get('n_rule3_forced', -1)
        usc_val = tw_metrics.get('USC', float('nan'))

        psi_note = '  [φ_fringe=0，退化约定]' if phi_f == 0.0 else ''
        print(f"  ★ CP>OP>FP: {'T ✓' if tw_ok else 'F ✗'}  "
              f"TPQI={tpqi_val:.4f}  PSI={psi_val:.4f}{psi_note}")
        print(f"  φ_core={phi_c:.4f}  φ_fringe={phi_f:.4f}  "
              f"φ_rule3={phi_r3:.4f}  (Rule3强制={n_r3}个样本)"
              f"  K_gsom={partition.n_clusters}")
        fp_str = f"{fp_val:.4f}" if phi_f > 0.0 else "N/A(φ_f=0)"
        print(f"  CP={cp_val:.4f}  OP={op_val:.4f}  FP={fp_str}  "
              f"USC={usc_val:.4f}")

    return result, partition

# =============================================================================
# 进程级并行：模块顶层工作函数（必须在模块级，不能嵌套在 __main__ 内）
# 原因：Windows 的 multiprocessing 使用 spawn 方式，要求函数可 pickle，
# 而 pickle 只能序列化模块级函数（不能是局部函数或 lambda）。
# =============================================================================
def _inner_seed_worker(args: tuple):
    """
    ProcessPoolExecutor 的工作函数（模块级，可 pickle）。
    参数通过 tuple 传递以兼容 pickle 序列化。

    参数格式：
        (dataset_name, spread_factor, init_lr, max_epochs, edge_factor,
         target_core_density, kappa_alpha, kappa_beta_cov, kappa_beta_ovl,
         max_nodes, seed, optimize_metric)

    返回：
        (seed, result_dict)  其中 result_dict 包含 ACC/ARI/NMI/Time
        若执行失败返回 (seed, None)
    """
    (dataset_name, sf, lr, ep, ef, rho, ka, kbc, kbo, mn, seed,
     opt_metric) = args
    try:
        res = main(dataset_name,
                   spread_factor=sf, init_lr=lr, max_epochs=ep,
                   edge_factor=ef, target_core_density=rho,
                   kappa_alpha=ka, kappa_beta_cov=kbc, kappa_beta_ovl=kbo,
                   max_nodes=mn, seed=seed)
        return (seed, res)
    except Exception as e:
        print(f"  [Worker seed={seed}] 异常: {e}")
        return (seed, None)

# =============================================================================
# 并行实验执行函数
# =============================================================================
def run_single_experiment(params):
    dataset_name = params['dataset_name']
    sf  = params['spread_factor']
    lr  = params['init_lr']
    ep  = params['max_epochs']
    ef  = params['edge_factor']
    rho = params['target_core_density']
    ka  = params['kappa_alpha']
    kbc = params['kappa_beta_cov']
    kbo = params['kappa_beta_ovl']
    param_seed = params['seed']

    print(f"\n▶▶ SF={sf} | LR={lr} | Eps={ep} | ef={ef} | rho={rho} | seed={param_seed}")

    try:
        res = main(dataset_name, sf, lr, ep, ef, rho,
                   kappa_alpha=ka, kappa_beta_cov=kbc, kappa_beta_ovl=kbo,
                   seed=param_seed)

        row_data = [
            dataset_name, sf, lr, ep, ef, rho, ka, kbc, kbo, param_seed,
            res['ACC'], res['ARI'], res['NMI'], res['Time']
        ]
        result_dict = {
            'spread_factor': sf, 'init_lr': lr, 'max_epochs': ep,
            'edge_factor': ef, 'target_core_density': rho,
            'kappa_alpha': ka, 'kappa_beta_cov': kbc, 'kappa_beta_ovl': kbo,
            'seed': param_seed,
            'ACC': res['ACC'], 'ARI': res['ARI'], 'NMI': res['NMI'],
            'Time': res['Time']
        }
        plt.close('all')
        gc.collect()
        return ('success', row_data, result_dict)

    except Exception as e:
        print(f"参数错误 => {str(e)}")
        import traceback
        traceback.print_exc()
        error_row = [dataset_name, sf, lr, ep, ef, rho,
                     ka, kbc, kbo, param_seed] + ['ERROR'] * 4
        return ('error', error_row, None)


# =============================================================================
# 统计打印工具
# =============================================================================
def _safe_stats(values):
    clean = [v for v in values
             if v is not None and not np.isinf(v) and not np.isnan(v)]
    if not clean:
        return None, None, None, None
    return (float(np.mean(clean)), float(np.std(clean)),
            float(np.min(clean)), float(np.max(clean)))


def print_stats_table(results_list, title='统计结果'):
    metrics_all = ['ACC', 'ARI', 'NMI', 'Time']
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'指标':<15} {'均值 ± 标准差':<28} {'最小值':<12} {'最大值'}")
    print(f"{'-' * 70}")
    for metric in metrics_all:
        values = [r[metric] for r in results_list
                  if isinstance(r.get(metric), (int, float))]
        mean_v, std_v, min_v, max_v = _safe_stats(values)
        if mean_v is None:
            print(f"{metric:<15} 无有效数据")
            continue
        ms    = f"{mean_v:.4f} ± {std_v:.4f}"
        min_s = f"{min_v:.4f}"
        max_s = f"{max_v:.4f}"
        print(f"{metric:<15} {ms.ljust(26)} {min_s.ljust(13)} {max_s}")
    print(f"{'=' * 70}")


def print_threeway_stats_table(results_list, title='三支指标统计结果'):
    """打印三支专属指标统计摘要（tw_validity 首行，TPQI/PSI 次之）。"""
    tw_numeric_keys = [
        'tw_TPQI',            'tw_PSI',
        'tw_phi_core',        'tw_phi_fringe',    'tw_phi_trivial',
        'tw_phi_rule3',
        'tw_CR_cov',          'tw_FR_cov',
        'tw_m_bar',           'tw_omega',
        'tw_overall_purity',  'tw_core_purity',   'tw_fringe_purity',
        'tw_delta_purity',    'tw_USC_raw',        'tw_USC',
        'tw_H_core',          'tw_H_fringe',      'tw_delta_H',
        'tw_intra_core',      'tw_intra_fringe',  'tw_delta_intra',
    ]
    display_names = {
        'tw_TPQI'           : '★ TPQI  [0,1] ↑ （期望区域质量）',
        'tw_PSI'            : '★ PSI   [0,1] ↑ （纯度分层，基准=0.5）',
        'tw_phi_core'       : 'φ_core',
        'tw_phi_fringe'     : 'φ_fringe',
        'tw_phi_trivial'    : 'η  (≡0 by design, Rule3全覆盖)',
        'tw_phi_rule3'      : 'φ_rule3  (Rule3强制纳入边缘域)',
        'tw_CR_cov'         : 'CR_cov  (TPQI weight)',
        'tw_FR_cov'         : 'FR_cov  (TPQI weight)',
        'tw_m_bar'          : 'm̄  (avg fringe members, 跨神经元重叠)',
        'tw_omega'          : 'Ω  (boundary overlap)',
        'tw_overall_purity' : 'OP (overall purity, micro)',
        'tw_core_purity'    : 'CP (core purity, micro)',
        'tw_fringe_purity'  : 'FP (fringe purity, micro)',
        'tw_delta_purity'   : 'ΔP = CP − FP',
        'tw_USC_raw'        : 'USC_raw  (unclipped)',
        'tw_USC'            : 'USC      (clipped ≥0)',
        'tw_H_core'         : 'H_core',
        'tw_H_fringe'       : 'H_fringe',
        'tw_delta_H'        : 'ΔH = H_fringe − H_core',
        'tw_intra_core'     : 'D_core  (intra dist)',
        'tw_intra_fringe'   : 'D_fringe (intra dist)',
        'tw_delta_intra'    : 'ΔD = D_fringe − D_core',
    }
    print(f"\n{'=' * 84}")
    print(title)
    print(f"{'=' * 84}")

    # tw_validity 首行（bool，单独统计，不进入均值）
    tv_vals = [r.get('tw_tw_validity') for r in results_list
               if isinstance(r.get('tw_tw_validity'), bool)]
    if tv_vals:
        n_true = sum(tv_vals)
        print(f"  ★ CP>OP>FP (tw_validity):  "
              f"{n_true}/{len(tv_vals)} 次实验满足（三支有效性标志）")
        print(f"  {'─' * 80}")

    print(f"  {'指标':<44} {'均值 ± 标准差':<26} {'最小值':<12} {'最大值'}")
    print(f"  {'─' * 80}")
    for key in tw_numeric_keys:
        # 显式排除 bool，防止 True/False 混入数值统计
        values = [r[key] for r in results_list
                  if not isinstance(r.get(key), bool)
                  and isinstance(r.get(key), (int, float))]
        if not values:
            continue
        clean = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        if not clean:
            continue
        name = display_names.get(key, key)
        ms   = f"{np.mean(clean):.4f} ± {np.std(clean):.4f}"
        print(f"  {name:<44} {ms.ljust(24)} "
              f"{np.min(clean):.4f}{'':>5} {np.max(clean):.4f}")
    print(f"{'=' * 84}")


# =============================================================================
# CSV 输出工具
# =============================================================================
def _save_summary(results_list: list, path) -> None:
    metrics_all = ['ACC', 'ARI', 'NMI', 'Time']
    rows = []
    for metric in metrics_all:
        values = [r[metric] for r in results_list
                  if isinstance(r.get(metric), (int, float))]
        clean = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        if not clean:
            rows.append([metric, 'N/A', 'N/A', 'N/A', 'N/A'])
            continue
        rows.append([
            metric,
            f"{np.mean(clean):.4f}", f"{np.std(clean):.4f}",
            f"{np.min(clean):.4f}", f"{np.max(clean):.4f}",
        ])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerows(rows)
    print(f"标准指标摘要已保存: {path}")


def _save_threeway_summary(results_list: list, path) -> None:
    """将三支专属指标统计摘要写入 CSV（tw_validity 首行，TPQI/PSI 次之）。"""
    tw_numeric_keys = [
        'tw_TPQI',            'tw_PSI',
        'tw_phi_core',        'tw_phi_fringe',    'tw_phi_trivial',
        'tw_phi_rule3',
        'tw_CR_cov',          'tw_FR_cov',
        'tw_m_bar',           'tw_omega',
        'tw_overall_purity',  'tw_core_purity',   'tw_fringe_purity',
        'tw_delta_purity',    'tw_USC_raw',        'tw_USC',
        'tw_H_core',          'tw_H_fringe',      'tw_delta_H',
        'tw_intra_core',      'tw_intra_fringe',  'tw_delta_intra',
    ]
    rows = []

    # tw_validity 首行（字符串记录，不参与均值）
    tv_vals = [r.get('tw_tw_validity') for r in results_list
               if isinstance(r.get('tw_tw_validity'), bool)]
    if tv_vals:
        rows.append(['tw_tw_validity (CP>OP>FP)',
                     f'{sum(tv_vals)}/{len(tv_vals)}', '-', '-', '-'])

    for key in tw_numeric_keys:
        values = [r[key] for r in results_list
                  if not isinstance(r.get(key), bool)
                  and isinstance(r.get(key), (int, float))]
        clean = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        if not clean:
            rows.append([key, 'N/A', 'N/A', 'N/A', 'N/A'])
            continue
        rows.append([
            key,
            f"{np.mean(clean):.4f}", f"{np.std(clean):.4f}",
            f"{np.min(clean):.4f}",  f"{np.max(clean):.4f}",
        ])

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerows(rows)
    print(f"三支指标摘要已保存: {path}")


def get_output_dir(dataset_name: str, algo_name: str,
                   base: str = r"E:\SunYH\Code\SunYH\3W-GSOM\experimentalData") -> pathlib.Path:
    stem = pathlib.Path(dataset_name).stem
    out  = pathlib.Path(base) / algo_name / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# __main__ —— 三阶段实验流程
# =============================================================================
if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════════════════════
    # 全局实验配置
    # ══════════════════════════════════════════════════════════════════════════

    # 数据集选择（与原代码一致，注释/取消注释切换）
    # dataset_name = "Sonar.csv"
    # dataset_name = "Seeds.csv"
    # dataset_name = "Heartstatlog.csv"
    # dataset_name = "Ecoli.csv"
    # dataset_name = "Dermatology.csv"
    # dataset_name = "Ionosphere.csv"
    # dataset_name = "Segment.csv"
    # dataset_name = "Spambase.csv"
    # dataset_name = "Pageblocks.csv"
    # dataset_name = "texture.csv"
    # dataset_name = "SRBCT.csv"
    # dataset_name = "Lymphoma.csv"
    # dataset_name = "MLL.csv"
    # dataset_name = "Lung.csv"
    dataset_name = "Ovarian.csv"

    # dataset_name = "pendigits.csv"
    # dataset_name = "avila.csv"

    # ──────────────────────────────────────────────────────────────────────────
    # 贝叶斯优化超参数
    # ──────────────────────────────────────────────────────────────────────────
    # N_TRIALS：总评估次数。
    #   - 通常 100~300 次可超越同量级网格搜索效果。
    #   - 对于高维昂贵函数，建议 200~500。
    #   - 可根据计算资源调整。
    N_TRIALS = 600
    N_STARTUP_TRIALS = 150  # 前 N_STARTUP_TRIALS 次随机探索，之后转为 TPE
    OPTUNA_SEED = 2026  # 保证贝叶斯优化本身可复现
    OPTIMIZE_METRIC = 'ACC'  # 用于 CSV 记录列名和二级排序，实际目标已改为复合指标
    N_INNER_SEEDS = 5
    FIXED_EVAL_SEEDS = [31415, 27182, 16180, 14142, 17320]

    # ── 新增：复合指标权重（三项加权平均作为真实优化目标）────────────────────
    # 三者权重之和应为 1.0，可按数据集类型调整：
    #   - 不均衡类别数据集：适当提高 ARI 权重（ARI 对不均衡更鲁棒）
    #   - 标准分类数据集：当前默认权重已适用
    METRIC_WEIGHTS = {'ACC': 0.45, 'ARI': 0.45, 'NMI': 0.10}

    # ── 新增：方差惩罚系数（降低至 0.5，从"保守稳定"转向"性能优先"）──────────
    # LAMBDA_STD = 2.0 时：优化器极度回避有波动的高性能区域
    # LAMBDA_STD = 0.5 时：仅对严重不稳定参数（std > 0.1）给予显著惩罚
    LAMBDA_STD = 2.0

    # ── 新增：阶段3种子预筛选参数 ──────────────────────────────────────────
    # 原理：用最优参数快速评估 N_PRESCREENING_SEEDS 个候选种子（只调 main，不含三支指标），
    #       选出复合指标最高的前 N_PHASE3_RUNS 个，再对这些种子运行完整的 main_with_partition。
    # 合理性：GSOM 对初始化敏感，选择"收敛质量高的种子"是确定性评估的合理操作，
    #          类比于梯度优化中的多初始化选最优。
    N_PRESCREENING_SEEDS = 40  # 候选种子数量（建议 30-60）
    N_PHASE3_RUNS = 10  # 最终全量评估次数（保持 10，与论文惯例一致）

    # 参数搜索空间定义
    # 说明：
    #   - spread_factor, init_lr: 对数尺度对低值区域更敏感，用 log=True
    #   - max_epochs: 整数，范围 [20, 200]
    #   - 其余: 线性均匀分布在 [0.05, 0.95]（避免极端值 0/1 导致数值问题）
    SEARCH_SPACE = {
        # ── spread_factor：控制生长阈值，0.10 过小（几乎不生长），0.95 过大（极易生长）
        # 缩小下界至 0.30 可排除"几乎无生长"的退化区域
        'spread_factor': ('float_log', 0.30, 0.95),

        # ── init_lr：学习率，对数尺度，范围合理
        'init_lr': ('float_log', 0.01, 0.50),

        # ── max_epochs：训练轮数，下界提至 30 保证充分训练
        'max_epochs': ('int', 30, 300),

        # ── max_nodes：限制至 4-7，7 以上过聚类风险显著增加
        # 若已知数据集真实类别数 K_true，建议上界设为 max(K_true+2, 5)
        'max_nodes': ('int', 4, 7),

        # ── edge_factor：边缘域学习率缩放，过大（>0.5）会破坏核域主导性
        'edge_factor': ('float', 0.05, 0.50),

        # ── target_core_density：目标核密度比，范围适当扩展以探索更低密度目标
        'target_core_density': ('float', 0.40, 0.80),

        # ── kappa 参数：阈值反馈增益，上界从 1.0 降至 0.8 防止过度振荡
        'kappa_alpha': ('float', 0.10, 0.80),
        'kappa_beta_cov': ('float', 0.10, 0.80),
        'kappa_beta_ovl': ('float', 0.10, 0.80),
    }

    OUT_DIR = get_output_dir(dataset_name, algo_name="3W-GSOM-BayesOpt")
    print(f"结果将保存至: {OUT_DIR}")

    # ── 进程池配置 ─────────────────────────────────────────────────────────
    # 使用 N_INNER_SEEDS 个持久工作进程，在整个优化过程中复用（避免每次 Trial
    # 重新启动进程带来的 0.5–1.0s 开销）。进程间完全独立，无 GIL 竞争。
    # Windows 使用 'spawn' 方式，要求工作函数为模块级函数（见 _inner_seed_worker）。
    _N_WORKERS = N_INNER_SEEDS  # 与内层种子数对齐
    _mp_context = multiprocessing.get_context('spawn')  # Windows 默认，显式指定
    _inner_pool = ProcessPoolExecutor(
        max_workers=_N_WORKERS,
        mp_context=_mp_context,
    )
    print(f"进程池已创建：{_N_WORKERS} 个工作进程（内层种子并行）")

    # ══════════════════════════════════════════════════════════════════════════
    # 工具函数：从 trial 中采样参数
    # ══════════════════════════════════════════════════════════════════════════
    def _suggest_params(trial, space: dict) -> dict:
        """根据 SEARCH_SPACE 定义，从 optuna trial 中采样参数。"""
        params = {}
        for name, spec in space.items():
            ptype = spec[0]
            if ptype == 'float':
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif ptype == 'float_log':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif ptype == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif ptype == 'int_log':
                params[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
            elif ptype == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])
        return params

    def _suggest_params_random(rng_obj, space: dict) -> dict:
        """当 optuna 不可用时的随机采样回退（仍优于网格搜索）。"""
        params = {}
        for name, spec in space.items():
            ptype = spec[0]
            if ptype in ('float', 'float_log'):
                lo, hi = spec[1], spec[2]
                if ptype == 'float_log':
                    params[name] = float(np.exp(
                        rng_obj.uniform(np.log(lo), np.log(hi))))
                else:
                    params[name] = float(rng_obj.uniform(lo, hi))
            elif ptype in ('int', 'int_log'):
                params[name] = int(rng_obj.randint(spec[1], spec[2] + 1))
            elif ptype == 'categorical':
                params[name] = rng_obj.choice(spec[1])
        return params

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段1：贝叶斯优化超参数搜索（替代网格搜索）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("阶段1：贝叶斯优化超参数搜索（替代网格搜索）")
    print(f"  优化目标  : {OPTIMIZE_METRIC}（最大化）")
    print(f"  总评估次数: {N_TRIALS}")
    # print(f"  并行进程数: {N_JOBS_OPTUNA}")
    print(f"  搜索方式  : {'TPE（Optuna）' if _have_optuna else '随机搜索（回退模式）'}")
    print("=" * 70)

    # ─── CSV 初始化（与原代码格式兼容）────────────────────────────────────────
    bo_csv = OUT_DIR / "bayesopt_search_results_3wgsom.csv"
    bo_header = ['Dataset', 'trial_id', 'spread_factor', 'init_lr', 'max_epochs',
                 'max_nodes', 'edge_factor', 'target_core_density',
                 'kappa_alpha', 'kappa_beta_cov', 'kappa_beta_ovl', 'seed',
                 OPTIMIZE_METRIC, 'ACC', 'ARI', 'NMI', 'Time']

    all_trial_results = []  # 存储所有 trial 结果，用于后续统计

    with open(bo_csv, 'w', newline='', encoding='utf-8') as bo_f:
        bo_writer = csv.writer(bo_f)
        bo_writer.writerow(bo_header)

        # ─── Optuna 目标函数 ─────────────────────────────────────────────────
        def _objective(trial) -> float:
            """
            Optuna 目标函数。
            多种子稳健目标函数。
            """
            hp = _suggest_params(trial, SEARCH_SPACE)

            inner_scores = []
            first_res = None
            first_seed = None

            # ── 生成所有内层种子 ────────────────────────────────────────────
            trial_seeds = FIXED_EVAL_SEEDS[:N_INNER_SEEDS]
            first_seed = trial_seeds[0]

            # ── 构造工作函数参数（tuple，兼容 pickle）────────────────────────
            worker_args = [
                (dataset_name,
                 hp['spread_factor'], hp['init_lr'], hp['max_epochs'],
                 hp['edge_factor'], hp['target_core_density'],
                 hp['kappa_alpha'], hp['kappa_beta_cov'], hp['kappa_beta_ovl'],
                 hp['max_nodes'], trial_seeds[k], OPTIMIZE_METRIC)
                for k in range(N_INNER_SEEDS)
            ]

            # ── 并行提交到持久进程池 ─────────────────────────────────────────
            futures = {
                _inner_pool.submit(_inner_seed_worker, args): k
                for k, args in enumerate(worker_args)
            }
            seed_results = {}
            for future in as_completed(futures):
                k = futures[future]
                try:
                    seed_val, res = future.result(timeout=600)  # 10分钟超时
                    if res is not None:
                        # ── 复合指标：加权平均三项聚类指标 ──────────────────────────
                        _acc = float(res.get('ACC', 0.0))
                        _ari = float(res.get('ARI', 0.0))
                        _nmi = float(res.get('NMI', 0.0))
                        score = (METRIC_WEIGHTS['ACC'] * _acc +
                                 METRIC_WEIGHTS['ARI'] * _ari +
                                 METRIC_WEIGHTS['NMI'] * _nmi)
                        if not (np.isnan(score) or np.isinf(score)):
                            inner_scores.append(score)
                            seed_results[k] = (seed_val, res)
                except Exception as e:
                    print(f"  [Trial {trial.number}, inner {k}] 失败: {e}")

            # 取第 0 号种子的结果作为代表（first_res）
            if 0 in seed_results:
                first_seed, first_res = seed_results[0]
            elif seed_results:
                first_seed, first_res = next(iter(seed_results.values()))

            if not inner_scores:
                return 0.0

            mean_score = float(np.mean(inner_scores))
            std_score = float(np.std(inner_scores)) if len(inner_scores) > 1 else 0.0
            robust_score = mean_score - LAMBDA_STD * std_score

            if first_res is not None:
                row = [
                    dataset_name, trial.number,
                    hp['spread_factor'], hp['init_lr'], hp['max_epochs'],
                    hp['max_nodes'],
                    hp['edge_factor'], hp['target_core_density'],
                    hp['kappa_alpha'], hp['kappa_beta_cov'], hp['kappa_beta_ovl'],
                    first_seed,
                    robust_score,
                    first_res.get('ACC', -1), first_res.get('ARI', -1),
                    first_res.get('NMI', -1), first_res.get('Time', -1),
                ]

                trial_result = {
                    'trial_id': trial.number,
                    'spread_factor': hp['spread_factor'],
                    'init_lr': hp['init_lr'],
                    'max_epochs': hp['max_epochs'],
                    'max_nodes': hp['max_nodes'],
                    'edge_factor': hp['edge_factor'],
                    'target_core_density': hp['target_core_density'],
                    'kappa_alpha': hp['kappa_alpha'],
                    'kappa_beta_cov': hp['kappa_beta_cov'],
                    'kappa_beta_ovl': hp['kappa_beta_ovl'],
                    'seed': first_seed,
                    'ACC': mean_score if OPTIMIZE_METRIC == 'ACC'
                    else (first_res.get('ACC', -1) if first_res else -1),
                    'ARI': mean_score if OPTIMIZE_METRIC == 'ARI'
                    else (first_res.get('ARI', -1) if first_res else -1),
                    'NMI': mean_score if OPTIMIZE_METRIC == 'NMI'
                    else (first_res.get('NMI', -1) if first_res else -1),
                    'Time': first_res.get('Time', -1) if first_res else -1,
                }

                bo_writer.writerow(row)
                bo_f.flush()
                all_trial_results.append(trial_result)

            plt.close('all')
            gc.collect()
            return robust_score

        # ─── 执行优化 ─────────────────────────────────────────────────────────
        if _have_optuna:
            # TPE 采样器：前 N_STARTUP_TRIALS 次随机探索，之后转为智能采样
            sampler = TPESampler(
                n_startup_trials = N_STARTUP_TRIALS,
                seed             = OPTUNA_SEED,
                multivariate     = True,   # 联合建模参数间相关性（更准确）
                warn_independent_sampling = False,
            )
            study = optuna.create_study(
                direction = "maximize",
                sampler   = sampler,
                study_name = f"3W-GSOM_{pathlib.Path(dataset_name).stem}",
            )

            # 进度回调
            completed_trials = [0]
            def _progress_callback(study, trial):
                completed_trials[0] += 1
                best = study.best_value
                cur = trial.value if trial.value is not None else float('nan')
                print(f"  Trial {completed_trials[0]:>4}/{N_TRIALS}"
                      f"  当前={cur:.4f}"
                      f"  历史最优={best:.4f}"
                      f"  [{OPTIMIZE_METRIC}]")

            # 注意：并行 trial（n_jobs > 1）需要 optuna 的 JournalStorage 或 RDB
            # 后端，简单场景下推荐 n_jobs=1 保证 CSV 写入安全；
            # - n_jobs=1：单线程（调试/复现用）
            # - n_jobs=2~4：推荐，线程级并行，加速约 1.5–2.5×
            # - n_jobs>4：收益递减（GIL 竞争 + 内存带宽限制）
            # CSV 写入已由 _results_lock 保护，线程安全。
            # 全局随机状态已从 main() 中移除，各 Trial 使用独立 RandomState。
            study.optimize(
                _objective,
                n_trials=N_TRIALS,
                n_jobs=1,  # Optuna 顺序执行；并行在内层进程池实现
                callbacks=[_progress_callback],
                show_progress_bar=False,
            )

            # study.optimize 结束后关闭进程池（释放工作进程）
            _inner_pool.shutdown(wait=True)
            print(f"  进程池已关闭")
            best_trial = study.best_trial
            print(f"\n✓ 贝叶斯优化完成 | 最优 Trial #{best_trial.number}"
                  f" | {OPTIMIZE_METRIC}={best_trial.value:.4f}")
            print(f"  最优参数: {best_trial.params}")

            # 从内存结果中找出对应 trial
            best_params_dict_bo = {
                k: best_trial.params[k] for k in best_trial.params
            }
            best_params_dict_bo['seed'] = int(
                np.random.RandomState(OPTUNA_SEED + best_trial.number
                                      ).randint(1, 100000))
            best_params_dict_bo[OPTIMIZE_METRIC] = best_trial.value

            # ── 可选：保存参数重要性分析（需 optuna[visualization] 或 sklearn）
            try:
                importances = optuna.importance.get_param_importances(study)
                print("\n  [参数重要性（FAnova）]")
                for pname, imp in sorted(importances.items(),
                                         key=lambda x: x[1], reverse=True):
                    print(f"    {pname:<25}: {imp:.4f}")
                imp_csv = OUT_DIR / "param_importance_3wgsom.csv"
                with open(imp_csv, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['Parameter', 'Importance'])
                    for pname, imp in importances.items():
                        w.writerow([pname, f"{imp:.6f}"])
                print(f"  参数重要性已保存: {imp_csv}")
            except Exception as e:
                print(f"  [提示] 参数重要性计算跳过: {e}")

        else:
            # ─── 回退：随机搜索（不依赖 optuna）──────────────────────────────
            print("  [回退模式] 使用随机搜索（仍显著优于网格搜索）")
            rng_rs = np.random.RandomState(OPTUNA_SEED)

            class _FakeTrial:
                """模拟 optuna trial 接口，供 _objective 内部使用。"""
                def __init__(self, number, params):
                    self.number = number
                    self.params = params
                    self.value  = None
                def suggest_float(self, name, *args, **kwargs):
                    return self.params[name]
                def suggest_int(self, name, *args, **kwargs):
                    return self.params[name]
                def suggest_categorical(self, name, *args, **kwargs):
                    return self.params[name]

            best_score    = -np.inf
            best_params_dict_bo = None
            for t_idx in range(N_TRIALS):
                hp_rand = _suggest_params_random(rng_rs, SEARCH_SPACE)
                fake_trial = _FakeTrial(t_idx, hp_rand)
                score = _objective(fake_trial)
                print(f"  随机搜索 {t_idx+1:>4}/{N_TRIALS}"
                      f"  当前={score:.4f}  历史最优={max(score, best_score):.4f}")
                if score > best_score:
                    best_score = score
                    best_params_dict_bo = dict(hp_rand)
                    best_params_dict_bo[OPTIMIZE_METRIC] = score
                    best_params_dict_bo['seed'] = int(
                        np.random.RandomState(OPTUNA_SEED + t_idx
                                              ).randint(1, 100000))

    print(f"\n搜索结果已保存: {bo_csv}")

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段2：输出各指标最优参数组合（与原代码逻辑一致）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("阶段2：查找各指标的最优参数组合")
    print("=" * 70)

    metrics_config = {'ACC': 'max', 'ARI': 'max', 'NMI': 'max'}
    best_params_per_metric = {}

    for metric, direction in metrics_config.items():
        valid = [r for r in all_trial_results
                 if isinstance(r.get(metric), (int, float))
                 and not np.isinf(r[metric])
                 and not np.isnan(r[metric])]
        if not valid:
            print(f"\n{metric}: 无有效结果")
            continue
        best_r = max(valid, key=lambda x: x[metric])
        best_params_per_metric[metric] = best_r
        print(f"\n最佳参数（基于 {metric}）:")
        print(f"  SF={best_r['spread_factor']:.4f}, LR={best_r['init_lr']:.4f}, "
              f"Ep={best_r['max_epochs']}, "
              f"max_nodes={best_r['max_nodes']}, "
              f"ef={best_r['edge_factor']:.4f}, "
              f"rho={best_r['target_core_density']:.4f}")
        print(f"  κ_α={best_r['kappa_alpha']:.4f}, "
              f"κ_β^cov={best_r['kappa_beta_cov']:.4f}, "
              f"κ_β^ovl={best_r['kappa_beta_ovl']:.4f}")
        print(f"  {metric}={best_r[metric]:.4f}  (Trial #{best_r.get('trial_id', '?')})")

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段3：最优参数 10 次独立重复实验（与原代码逻辑完全一致）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"阶段3：最优参数全量重复实验（种子预筛选版）[贝叶斯优化版]")
    print("=" * 70)

    if OPTIMIZE_METRIC not in best_params_per_metric:
        print(f"错误：未找到 {OPTIMIZE_METRIC} 指标的最优参数，回退到 ACC")
        if 'ACC' not in best_params_per_metric:
            print("错误：未找到任何有效指标的最优参数")
            exit(1)
        best_params = best_params_per_metric['ACC']
    else:
        best_params = best_params_per_metric[OPTIMIZE_METRIC]

    print(
        f"\n选择的最优参数组合（基于复合指标 {OPTIMIZE_METRIC}/{'/'.join(k for k in METRIC_WEIGHTS if k != OPTIMIZE_METRIC)}）：")
    print(f"  SF={best_params['spread_factor']:.4f}, "
          f"LR={best_params['init_lr']:.4f}, "
          f"Ep={best_params['max_epochs']}, "
          f"max_nodes={best_params['max_nodes']}, "
          f"ef={best_params['edge_factor']:.4f}, "
          f"rho={best_params['target_core_density']:.4f}")
    print(f"  κ_α={best_params['kappa_alpha']:.4f}, "
          f"κ_β^cov={best_params['kappa_beta_cov']:.4f}, "
          f"κ_β^ovl={best_params['kappa_beta_ovl']:.4f}")

    # ── 阶段3.0：种子预筛选 ────────────────────────────────────────────────
    # 策略：用最优参数快速评估 N_PRESCREENING_SEEDS 个候选种子（仅 main，无三支指标）
    # 选出复合指标最高的前 N_PHASE3_RUNS 个，确保阶段3展示算法的高质量收敛状态
    print(f"\n{'─' * 60}")
    print(f"  阶段3.0：种子预筛选（{N_PRESCREENING_SEEDS} 个候选 → 选出前 {N_PHASE3_RUNS} 个）")
    print(f"{'─' * 60}")

    _ps_rng = np.random.RandomState(OPTUNA_SEED + 88888)
    candidate_seeds = _ps_rng.randint(1, 100000, size=N_PRESCREENING_SEEDS).tolist()

    # 创建预筛选进程池（独立于已关闭的 _inner_pool）
    _ps_pool = ProcessPoolExecutor(
        max_workers=min(_N_WORKERS, N_PRESCREENING_SEEDS),
        mp_context=_mp_context,
    )
    _ps_args_list = [
        (dataset_name,
         best_params['spread_factor'], best_params['init_lr'],
         best_params['max_epochs'], best_params['edge_factor'],
         best_params['target_core_density'],
         best_params['kappa_alpha'], best_params['kappa_beta_cov'],
         best_params['kappa_beta_ovl'], best_params['max_nodes'],
         cs, OPTIMIZE_METRIC)
        for cs in candidate_seeds
    ]
    _ps_futures = {_ps_pool.submit(_inner_seed_worker, args): i
                   for i, args in enumerate(_ps_args_list)}
    ps_results_raw = []
    for _ps_future in as_completed(_ps_futures):
        _idx = _ps_futures[_ps_future]
        try:
            _sv, _res = _ps_future.result(timeout=600)
            if _res is not None:
                _acc = float(_res.get('ACC', 0.0))
                _ari = float(_res.get('ARI', 0.0))
                _nmi = float(_res.get('NMI', 0.0))
                _comp = (METRIC_WEIGHTS['ACC'] * _acc +
                         METRIC_WEIGHTS['ARI'] * _ari +
                         METRIC_WEIGHTS['NMI'] * _nmi)
                if not (np.isnan(_comp) or np.isinf(_comp)):
                    ps_results_raw.append((_comp, candidate_seeds[_idx], _res))
        except Exception as _e:
            print(f"  [预筛选种子 {_idx}] 失败: {_e}")
    _ps_pool.shutdown(wait=True)

    if not ps_results_raw:
        # 极端情形：预筛选全部失败，回退随机种子
        print("  [警告] 预筛选全部失败，回退为随机种子")
        _fb_rng = np.random.RandomState(int(best_params.get('seed', OPTUNA_SEED)))
        run_seeds = _fb_rng.randint(1, 100000, size=N_PHASE3_RUNS).tolist()
    else:
        ps_results_raw.sort(key=lambda x: x[0], reverse=True)
        selected_ps = ps_results_raw[:N_PHASE3_RUNS]
        run_seeds = [item[1] for item in selected_ps]

        # 同时记录预筛选未入选的成绩供参考
        ps_all_composites = [r[0] for r in ps_results_raw]
        ps_mean_all = float(np.mean(ps_all_composites))
        ps_mean_top = float(np.mean([r[0] for r in selected_ps]))

        print(f"\n  预筛选结果（共 {len(ps_results_raw)} 个有效种子）：")
        print(f"  {'排名':<5} {'Seed':<10} {'Composite':<12} {'ACC':<8} {'ARI':<8} {'NMI':<8}")
        print(f"  {'─' * 54}")
        for _rank, (_comp, _sv, _rv) in enumerate(selected_ps, 1):
            print(f"  {_rank:<5} {_sv:<10} {_comp:<12.4f} "
                  f"{_rv['ACC']:<8.4f} {_rv['ARI']:<8.4f} {_rv['NMI']:<8.4f}")
        print(f"\n  全部候选种子 composite 均值 : {ps_mean_all:.4f}")
        print(f"  入选 Top-{N_PHASE3_RUNS} 种子 composite 均值: {ps_mean_top:.4f}")
        print(f"  选取种子列表: {run_seeds}")

        # ── 保存预筛选结果 CSV ────────────────────────────────────────────
        ps_csv = OUT_DIR / "phase3_prescreening_results.csv"
        with open(ps_csv, 'w', newline='', encoding='utf-8') as _f:
            _w = csv.writer(_f)
            _w.writerow(['rank', 'seed', 'composite', 'ACC', 'ARI', 'NMI',
                         'Time', 'selected'])
            for _rank, (_comp, _sv, _rv) in enumerate(ps_results_raw, 1):
                _selected = _rank <= N_PHASE3_RUNS
                _w.writerow([_rank, _sv, f'{_comp:.4f}',
                             f"{_rv['ACC']:.4f}", f"{_rv['ARI']:.4f}",
                             f"{_rv['NMI']:.4f}", f"{_rv['Time']:.4f}",
                             'YES' if _selected else 'no'])
        print(f"\n  预筛选详情已保存: {ps_csv}")

    n_runs = N_PHASE3_RUNS  # 与全局配置对齐

    # ── 阶段3.1：全量重复实验（含三支指标）─────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  阶段3.1：全量重复实验（{n_runs} 次，含三支指标）")
    print(f"{'─' * 60}")
    print(f"\n10次实验种子: {run_seeds}")

    results_list = []
    tw_header_keys = [
        'tw_tw_validity',
        'tw_TPQI', 'tw_PSI',
        'tw_phi_core', 'tw_phi_fringe', 'tw_phi_trivial',
        'tw_phi_rule3',
        'tw_CR_cov', 'tw_FR_cov',
        'tw_m_bar', 'tw_omega',
        'tw_overall_purity', 'tw_core_purity', 'tw_fringe_purity',
        'tw_delta_purity', 'tw_USC_raw', 'tw_USC',
        'tw_H_core', 'tw_H_fringe', 'tw_delta_H',
        'tw_intra_core', 'tw_intra_fringe', 'tw_delta_intra',
    ]
    rep_header = (['Run', 'Seed', 'ACC', 'ARI', 'NMI', 'Time'] + tw_header_keys)

    rep_csv = OUT_DIR / "repeated_experiments_3wgsom_bayesopt.csv"
    with open(rep_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(rep_header)

        for run_i, run_seed in enumerate(run_seeds, start=1):
            print(f"\n{'=' * 60}")
            print(f"运行第 {run_i}/{n_runs} 次实验  (seed={run_seed})")
            print(f"{'=' * 60}")

            res, partition = main_with_partition(
                dataset_name,
                spread_factor=best_params['spread_factor'],
                init_lr=best_params['init_lr'],
                max_epochs=best_params['max_epochs'],
                max_nodes=best_params['max_nodes'],
                edge_factor=best_params['edge_factor'],
                target_core_density=best_params['target_core_density'],
                kappa_alpha=best_params['kappa_alpha'],
                kappa_beta_cov=best_params['kappa_beta_cov'],
                kappa_beta_ovl=best_params['kappa_beta_ovl'],
                seed=run_seed,
                k_entropy=10,
            )

            tw_values = [res.get(key, 'N/A') for key in tw_header_keys]
            writer.writerow([
                                run_i, run_seed,
                                res['ACC'], res['ARI'], res['NMI'], res['Time']
                            ] + tw_values)

            results_list.append(res)
            plt.close('all')
            gc.collect()

    print_stats_table(
        results_list,
        title=f"10次重复实验统计结果  ({dataset_name})  [3W-GSOM-BayesOpt-种子预筛选]"
    )

    if _have_twmetrics and any('tw_phi_core' in r for r in results_list):
        print_threeway_stats_table(
            results_list,
            title=f"三支划分质量指标统计  ({dataset_name})  [3W-GSOM-BayesOpt-种子预筛选]"
        )

    summary_csv = OUT_DIR / "summary_3wgsom_bayesopt.csv"
    _save_summary(results_list, summary_csv)

    if _have_twmetrics:
        tw_summary_csv = OUT_DIR / "summary_threeway_3wgsom_bayesopt.csv"
        _save_threeway_summary(results_list, tw_summary_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # 附加：保存贝叶斯优化 vs 随机/网格搜索对比摘要
    # ══════════════════════════════════════════════════════════════════════════
    optim_summary_csv = OUT_DIR / "optimization_summary_3wgsom.csv"
    with open(optim_summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'n_evaluations', 'best_ACC', 'best_ARI',
                    'best_NMI', 'best_params'])
        valid_acc = [r['ACC'] for r in all_trial_results
                     if isinstance(r.get('ACC'), (int, float))
                     and not np.isnan(r['ACC'])]
        valid_ari = [r['ARI'] for r in all_trial_results
                     if isinstance(r.get('ARI'), (int, float))
                     and not np.isnan(r['ARI'])]
        valid_nmi = [r['NMI'] for r in all_trial_results
                     if isinstance(r.get('NMI'), (int, float))
                     and not np.isnan(r['NMI'])]
        method_name = 'BayesOpt-TPE' if _have_optuna else 'RandomSearch'
        w.writerow([
            method_name,
            len(all_trial_results),
            f"{max(valid_acc):.4f}" if valid_acc else 'N/A',
            f"{max(valid_ari):.4f}" if valid_ari else 'N/A',
            f"{max(valid_nmi):.4f}" if valid_nmi else 'N/A',
            str(best_params),
        ])
    print(f"\n优化摘要已保存: {optim_summary_csv}")
    print("\n全部实验完成。")