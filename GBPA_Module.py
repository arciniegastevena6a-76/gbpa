import numpy as np

class GBPA_Module:
    """
    广义基本概率指派生成模块 
    严格对应论文 Eq. 19-20
    """
    def train_models(self, train_data, train_labels):
        classes = np.unique(train_labels)
        models = {}
        
        for cls in classes:
            # 提取当前类的所有样本
            cls_data = train_data[train_labels == cls]
            
            models[cls] = {
                'min': np.min(cls_data, axis=0),
                'mean': np.mean(cls_data, axis=0),
                'max': np.max(cls_data, axis=0)
            }
        return models

    def calculate_gbpa(self, test_data, models):
        num_samples, num_attrs = test_data.shape
        class_names = list(models.keys())
        num_classes = len(class_names)
        
        # 最后一列是 m(empty)
        gbpa_matrix = np.zeros((num_samples, num_classes + 1))
        
        for i in range(num_samples):
            sample = test_data[i, :]
            memberships = np.zeros(num_classes)
            
            for c, cls in enumerate(class_names):
                params = models[cls]
                attr_m = np.zeros(num_attrs)
                
                for a in range(num_attrs):
                    x = sample[a]
                    p_min, p_mean, p_max = params['min'][a], params['mean'][a], params['max'][a]
                    
                    # 三角隶属度函数 (Eq. 19)
                    if x <= p_min or x >= p_max:
                        attr_m[a] = 0
                    elif x == p_mean:
                        attr_m[a] = 1
                    elif x < p_mean:
                        denom = p_mean - p_min
                        attr_m[a] = (x - p_min) / (denom + 1e-10)
                    else:
                        denom = p_max - p_mean
                        attr_m[a] = (p_max - x) / (denom + 1e-10)
                
                memberships[c] = np.mean(attr_m)
            
            total_mass = np.sum(memberships)
            
            # 归一化与冲突计算 (Eq. 20)
            if total_mass >= 1:
                gbpa_matrix[i, :num_classes] = memberships / total_mass
                gbpa_matrix[i, -1] = 0
            else:
                gbpa_matrix[i, :num_classes] = memberships
                gbpa_matrix[i, -1] = 1.0 - total_mass
                
        m_empty_mean = np.mean(gbpa_matrix[:, -1])
        return gbpa_matrix, m_empty_mean