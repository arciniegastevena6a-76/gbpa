import numpy as np

class FGS_Module:
    """
    Fuzzy Gap Statistic
    """
    def __init__(self, config):
        self.config = config

    def run_fgs(self, data):
        """
        执行 FGS 流程确定最佳 K
        """
        K_max = self.config['fcm']['max_clusters']
        B = self.config['fgs']['B_monte_carlo']
        m = self.config['fcm']['m_exponent']
        
        N, D = data.shape
        log_J = np.zeros(K_max)
        log_J_star = np.zeros((K_max, B))
        
        print("   > Starting FGS calculation...")
        
        # 1. 计算原始数据的 log(J)
        print("   > Processing Original Data: ", end='', flush=True)
        for k in range(1, K_max + 1):
            # 索引 k-1
            _, _, J_val = self.my_fcm(data, k, m, 100, 1e-5)
            log_J[k-1] = np.log(max(J_val, 1e-10))
            if k % 5 == 0: print('.', end='', flush=True)
        print(" Done.")
        
        # 2. 蒙特卡洛模拟
        print(f"   > Running Monte Carlo (B={B}): ", end='', flush=True)
        min_v = np.min(data, axis=0)
        max_v = np.max(data, axis=0)
        
        for b in range(B):
            # 生成均匀分布的随机数据
            rand_data = min_v + (max_v - min_v) * np.random.rand(N, D)
            
            for k in range(1, K_max + 1):
                _, _, J_val_b = self.my_fcm(rand_data, k, m, 100, 1e-5)
                log_J_star[k-1, b] = np.log(max(J_val_b, 1e-10))
            
            if (b + 1) % 10 == 0: print(f"{int((b+1)/B*100)}%...", end='', flush=True)
        print(" Done.")
        
        # 3. 计算统计量
        mean_log_J_star = np.mean(log_J_star, axis=1) # E[log(J*)]
        gap = mean_log_J_star - log_J                 # Gap(k)
        sd_k = np.sqrt(np.mean((log_J_star - mean_log_J_star[:, None])**2, axis=1))
        sk = sd_k * np.sqrt(1 + 1/B)
        
        # 4. 寻找最佳 K
        optimal_k = 1
        # Python 索引 0 对应 k=1
        for k_idx in range(K_max - 1):
            # k_idx 对应 k=k_idx+1
            # 比较 k 与 k+1
            if gap[k_idx] >= gap[k_idx+1] - sk[k_idx+1]:
                optimal_k = k_idx + 1
                break
        
        # 备选策略：如果没找到，找峰值
        if optimal_k == 1 and len(gap) > 1 and gap[0] < gap[1]:
            optimal_k = np.argmax(gap) + 1
            
        return optimal_k, gap, sk

    def my_fcm(self, data, cluster_n, m, max_iter, min_improv):
        """
        FCM 算法
        """
        N, D = data.shape
        if cluster_n == 1:
            center = np.mean(data, axis=0)
            dists = np.sum((data - center)**2, axis=1)
            J_final = np.sum(dists)
            return np.array([center]), np.ones((1, N)), J_final

        # 从数据中随机选择初始中心，避免随机隶属度导致的 NaN
        rand_indices = np.random.choice(N, cluster_n, replace=False)
        centers = data[rand_indices, :]
        
        # 初始化隶属度
        U = np.zeros((cluster_n, N))
        
        # 预计算距离以初始化 U
        # cdist 稍微快一点
        dists = np.zeros((cluster_n, N))
        for c in range(cluster_n):
            dists[c, :] = np.sum((data - centers[c, :])**2, axis=1)
        
        dists[dists < 1e-10] = 1e-10
        inv_dists = dists ** (-1 / (m - 1))
        sum_inv = np.sum(inv_dists, axis=0)
        U = inv_dists / (sum_inv + 1e-10)
        
        J_old = float('inf')
        J_final = 0
        
        for i in range(max_iter):
            # 1. 更新中心 (Eq. 14 的变体)
            Um = U ** m
            sum_Um = np.sum(Um, axis=1)
            sum_Um[sum_Um == 0] = 1e-10 # 保护
            
            centers = (Um @ data) / sum_Um[:, None]
            
            # 2. 计算距离
            dists = np.zeros((cluster_n, N))
            for c in range(cluster_n):
                dists[c, :] = np.sum((data - centers[c, :])**2, axis=1)
            
            # 3. 计算目标函数
            J_final = np.sum(Um * dists)
            
            # NaN 检查
            if np.isnan(J_final):
                J_final = J_old
                break
            
            if abs(J_final - J_old) < min_improv:
                break
            J_old = J_final
            
            # 4. 更新隶属度
            dists[dists < 1e-10] = 1e-10
            inv_dists = dists ** (-1 / (m - 1))
            sum_inv = np.sum(inv_dists, axis=0)
            U = inv_dists / (sum_inv + 1e-10)
            
        return centers, U, J_final