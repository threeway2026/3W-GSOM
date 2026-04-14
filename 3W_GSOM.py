import os, gc, importlib, time, csv, copy
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
np.set_printoptions(suppress=True)

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

class GSOMNode:
    def __init__(self, weights, neuron_id):
        self.weights = weights
        self.error = 0.0
        self.id = neuron_id
        self.approximation = None
        self.sim_order = []

class ThreeWayGSOM:

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
        if self.random_state is not None:
            self.rng = np.random.RandomState(self.random_state)
        else:
            self.rng = np.random.RandomState()
        self.grow_threshold = -self.input_dim * np.log(max(self.spread_factor, 1e-9))
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

    def _compute_similarity_matrix(self, X):
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

    def _build_weights_array(self):
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
        n = self._weights_array.shape[0]
        if n >= 2:
            dist_nn = cdist(self._weights_array, self._weights_array).astype(self.dtype)
            np.fill_diagonal(dist_nn, np.inf)
            self.local_deltas_array = dist_nn.min(axis=1).astype(self.dtype)
        else:
            fallback = float(self.delta) if self.delta is not None else 1.0
            self.local_deltas_array = np.array([fallback], dtype=self.dtype)

    def _refresh_local_deltas(self):
        if self._weights_array is None or self._weights_array.shape[0] == 0:
            return
        self._recompute_local_deltas()

    def _init_grid(self):
        self.sim_matrix = self._compute_similarity_matrix(self.data)
        N_INIT_BALLS = 5

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

                dists_to_centers = cdist(self.data, cand_centers)
                labels_cand = np.argmin(dists_to_centers, axis=1)

                unique_labels = np.unique(labels_cand)
                if len(unique_labels) < 2:
                    continue

                a_vals = dists_to_centers[np.arange(len(self.data)), labels_cand]

                sorted_dists = np.sort(dists_to_centers, axis=1)
                b_vals = sorted_dists[:, 1]
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

        mu_core_list, mu_edge_list = [], []
        for binfo in ball_info:
            idx_b = binfo['indices']
            if len(idx_b) == 0:
                continue
            dists_b = np.linalg.norm(self.data[idx_b] - binfo['center'], axis=1)
            r_plus = float(np.median(dists_b))
            r_minus = float(np.max(dists_b))
            mu_core_list.append(dist2mu_local(r_plus))
            mu_edge_list.append(dist2mu_local(r_minus))

        _tau_alpha = 0.75
        _tau_beta = 0.25
        alpha0 = float(np.quantile(mu_core_list, _tau_alpha))
        beta0 = float(np.quantile(mu_edge_list, _tau_beta))

        if len(ball_info) > 1:
            c_all = np.array([b['center'] for b in ball_info])
            cdist_ = np.linalg.norm(c_all[:, None] - c_all[None, :], axis=2)
            np.fill_diagonal(cdist_, np.inf)
            d_min = cdist_.min()
            mu_gap = dist2mu_local(d_min)
            beta0 = min(beta0, mu_gap)

        _eps = 1e-3
        _delta = 0.05
        self._eps = _eps
        self._delta = _delta
        self.alpha = float(np.clip(alpha0, _eps + _delta, 1.0 - _eps))
        beta_upper = self.alpha - _delta
        self.beta = float(np.clip(beta0, _eps, beta_upper))

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

        self._build_weights_array()

    def _find_matches_raw(self, x):
        if self._weights_array.shape[0] == 0:
            return [], [], []
        dists = np.linalg.norm(self._weights_array - x, axis=1)
        deltas = (self.local_deltas_array if self.local_deltas_array is not None
                  else np.full_like(dists, self.delta))
        membership_arr = np.exp(-(dists ** 2) / (2 * (deltas ** 2) + 1e-9))

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

    def _get_learning_rate(self, epoch):
        return self.init_lr * np.exp(-epoch / self.max_epochs)

    def _get_neighbor_range(self, epoch):
        theta_0 = 3
        nr = theta_0 * np.exp(-epoch / self.max_epochs)
        return max(1, round(nr))

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

    def _grow_node(self, parent_id):

        pnode = self.neurons[parent_id]
        assigned_indices = self._get_samples_for_node(parent_id)

        if len(assigned_indices) < self.min_samples_to_grow:
            pnode.error = 0.0
            return

        sub_data = self.data[assigned_indices]

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

    @staticmethod
    def _project_onto_Omega(alpha_raw, beta_raw, eps=1e-3, delta=0.05):

        a, b = float(alpha_raw), float(beta_raw)

        if b >= eps and a <= 1.0 - eps and a - b >= delta:
            return a, b

        best_d2 = float('inf')
        best = (eps + delta, eps)

        t1 = max(eps + delta, min(a, 1.0 - eps))
        p1 = (t1, eps)
        d1 = (a - p1[0]) ** 2 + (b - p1[1]) ** 2
        if d1 < best_d2:
            best_d2, best = d1, p1

        t2 = max(eps, min(b, 1.0 - eps - delta))
        p2 = (1.0 - eps, t2)
        d2 = (a - p2[0]) ** 2 + (b - p2[1]) ** 2
        if d2 < best_d2:
            best_d2, best = d2, p2

        t3 = max(eps, min((a + b - delta) / 2.0, 1.0 - eps - delta))
        p3 = (t3 + delta, t3)
        d3 = (a - p3[0]) ** 2 + (b - p3[1]) ** 2
        if d3 < best_d2:
            best_d2, best = d3, p3

        return float(best[0]), float(best[1])


    def _update_thresholds(self, use_ema: bool = False, ema_k: float = 0.25):

        if not self.use_dynamic_threshold:
            return
        n_nodes = len(self.neurons)
        if n_nodes == 0:
            return

        _eps   = getattr(self, '_eps',   1e-3)
        _delta = getattr(self, '_delta', 0.05)

        if (hasattr(self, '_cached_dists_nd') and
                self._cached_dists_nd is not None and
                self._cached_dists_nd.shape == (n_nodes, self.sampleCount)):
            dists = self._cached_dists_nd.T
        else:
            dists = cdist(self.data, self._weights_array)

        deltas = np.tile(self.local_deltas_array.reshape(1, -1),
                         (self.sampleCount, 1))
        mem = np.exp(-(dists ** 2) / (2 * (deltas ** 2) + 1e-9))

        core_mask = (mem >= self.alpha)
        support_mask = (mem >= self.beta)
        fringe_mask = support_mask & ~core_mask

        core_cnt = core_mask.sum(axis=0).astype(float)
        support_cnt = support_mask.sum(axis=0).astype(float)

        with np.errstate(invalid='ignore', divide='ignore'):
            rho_vals = np.where(support_cnt > 0, core_cnt / support_cnt, 0.0)
        rho = float(rho_vals.mean())

        covered = support_mask.any(axis=1)
        eta = 1.0 - float(covered.sum()) / self.sampleCount

        if n_nodes >= 2:
            fm_int = fringe_mask.astype(np.int32)
            inter_mat = fm_int.T @ fm_int
            fringe_cnt = fringe_mask.sum(axis=0).astype(float)
            union_mat = fringe_cnt[:, None] + fringe_cnt[None, :] - inter_mat
            p_idx, q_idx = np.triu_indices(n_nodes, k=1)
            inter_vals = inter_mat[p_idx, q_idx].astype(float)
            union_vals = union_mat[p_idx, q_idx].astype(float)
            valid_mask = union_vals > 0
            omega = (float(np.mean(inter_vals[valid_mask] / union_vals[valid_mask]))
                     if valid_mask.any() else 0.0)
        else:
            omega = 0.0

        if use_ema:
            if not hasattr(self, '_ema'):
                self._ema = {'rho': rho, 'eta': eta, 'omega': omega}
            else:
                self._ema['rho']   = (1 - ema_k) * self._ema['rho']   + ema_k * rho
                self._ema['eta']   = (1 - ema_k) * self._ema['eta']   + ema_k * eta
                self._ema['omega'] = (1 - ema_k) * self._ema['omega'] + ema_k * omega
            rho, eta, omega = self._ema['rho'], self._ema['eta'], self._ema['omega']

        alpha_raw = self.alpha + self.kappa_alpha * (rho - self.target_core_density)
        beta_raw = self.beta - self.kappa_beta_cov * eta \
                   + self.kappa_beta_ovl * omega

        self.alpha, self.beta = ThreeWayGSOM._project_onto_Omega(
            alpha_raw, beta_raw, _eps, _delta)

        self.alpha_history.append(self.alpha)
        self.beta_history.append(self.beta)


    def fit(self):
        self._init_grid()
        indices = np.arange(self.sampleCount)
        grow_epochs = int(self.max_epochs * 0.7)

        for epoch in range(self.max_epochs):
            allow_grow = (epoch < grow_epochs)
            self.rng.shuffle(indices)
            self._refresh_local_deltas()
            growth_candidates = set()

            for i in indices:
                x = self.data[i, :]
                matches = self._find_matches(x)
                self._update_neurons(x, matches, epoch)

                if allow_grow:
                    for cid in matches['core']:
                        cnode = self.neurons[cid]
                        if cnode.error > self.grow_threshold:
                            growth_candidates.add(cid)
                            cnode.error = 0.0

            if allow_grow and growth_candidates:
                for cid in sorted(growth_candidates):
                    if cid not in self.neurons:
                        continue
                    if len(self.neurons) < self.max_nodes:
                        self._grow_node(cid)
            else:
                self._refresh_local_deltas()

            _use_ema_this_epoch = (epoch < grow_epochs)
            self._update_thresholds(use_ema=_use_ema_this_epoch, ema_k=0.3)

        self._postprocess_threeway_clusters_after_training()
        return self

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

        alpha_mask = membership_cache >= self.alpha
        beta_mask = (membership_cache >= self.beta) & ~alpha_mask

        n_core_cands = alpha_mask.sum(axis=1)

        single_mask = (n_core_cands == 1)
        if single_mask.any():
            single_idx = np.where(single_mask)[0]
            assigned_col = np.argmax(alpha_mask[single_idx], axis=1)
            for local_i, col in enumerate(assigned_col):
                clusters_info[node_ids[col]]['core'].add(int(single_idx[local_i]))

        multi_mask = (n_core_cands > 1)
        if multi_mask.any():
            multi_idx = np.where(multi_mask)[0]
            masked_mem = membership_cache[multi_idx] * alpha_mask[multi_idx]
            best_col = np.argmax(masked_mem, axis=1)
            for local_i, (global_i, bcol) in enumerate(zip(multi_idx, best_col)):
                clusters_info[node_ids[bcol]]['core'].add(int(global_i))
                for col in range(len(node_ids)):
                    if col != bcol and alpha_mask[global_i, col]:
                        if membership_cache[global_i, col] >= self.beta:
                            clusters_info[node_ids[col]]['edge'].add(int(global_i))

        edge_rows, edge_cols = np.where(beta_mask)
        for i, j in zip(edge_rows, edge_cols):
            clusters_info[node_ids[j]]['edge'].add(int(i))

        covered_mask = (n_core_cands >= 1) | beta_mask.any(axis=1)

        for nid in node_ids:
            if len(clusters_info[nid]['core']) == 0:
                col = id2col[nid]
                _voronoi_mask = (np.argmax(membership_cache, axis=1) == col)
                if _voronoi_mask.any():
                    _mems_voronoi = membership_cache[:, col].copy()
                    _mems_voronoi[~_voronoi_mask] = -np.inf
                    best_i = int(np.argmax(_mems_voronoi))
                else:
                    best_i = int(np.argmax(membership_cache[:, col]))

                for other_nid in node_ids:
                    if other_nid != nid:
                        if best_i in clusters_info[other_nid]['core']:
                            clusters_info[other_nid]['core'].discard(best_i)
                            mu_other = membership_cache[best_i, id2col[other_nid]]
                            if mu_other >= self.beta:
                                clusters_info[other_nid]['edge'].add(best_i)

                clusters_info[nid]['core'].add(best_i)
                clusters_info[nid]['edge'].discard(best_i)
                covered_mask[best_i] = True

        uncovered = np.where(~covered_mask)[0]
        if len(uncovered) > 0:
            best_cols = np.argmax(membership_cache[uncovered], axis=1)
            for local_i, best_col in enumerate(best_cols):
                clusters_info[node_ids[best_col]]['edge'].add(int(uncovered[local_i]))

        self.final_clusters_info = clusters_info

    def get_threeway_partition(self):
        if self.final_clusters_info is None:
            raise RuntimeError("模型尚未 fit() 或后处理失败。")
        return copy.deepcopy(self.final_clusters_info)

    def predict_labels(self) -> np.ndarray:

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


