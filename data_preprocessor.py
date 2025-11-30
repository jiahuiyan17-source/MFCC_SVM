import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class MFCCDataPreprocessor:
    """
    MFCC数据预处理器 - 改进版本
    确保无数据泄露，支持不平衡数据处理
    """
    
    def __init__(self,
                 normalize_features: bool = True,
                 feature_scaling_method: str = 'standard',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 fixed_length: Optional[int] = None):
        """
        初始化MFCC数据预处理器
        """
        self.normalize_features = normalize_features
        self.feature_scaling_method = feature_scaling_method
        self.test_size = test_size
        self.random_state = random_state
        self.fixed_length = fixed_length
        
        # 初始化标准化器和编码器
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 存储处理后的数据和统计信息
        self.processed_data = {}
        self.feature_stats = {}
        self.dataset_info = {}
        self.all_labels = set()
        
        # 标记标准化器是否已拟合
        self.scaler_fitted = False
        self.label_encoder_fitted = False
    
    def preprocess_features(self, datasets: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        预处理MFCC特征
        """
        print("开始预处理MFCC特征...")
        
        # 首先收集所有可能的标签
        self._collect_all_labels(datasets)
        
        processed_datasets = {}
        
        for dataset_type, dataset in datasets.items():
            print(f"\n处理 {dataset_type} 数据集...")
            
            # 预处理特征
            processed_dataset = self._preprocess_dataset(dataset, dataset_type)
            processed_datasets[dataset_type] = processed_dataset
            
            # 计算特征统计信息
            self.feature_stats[dataset_type] = self._calculate_feature_statistics(processed_dataset)
            
            # 存储数据集信息
            self.dataset_info[dataset_type] = {
                'samples': len(processed_dataset),
                'feature_dimension': processed_dataset[0]['processed_features'].shape[0] if processed_dataset else 0,
                'label_distribution': self._get_label_distribution(processed_dataset)
            }
            
            print(f"  {dataset_type} 数据集: {len(processed_dataset)} 个样本")
            print(f"  特征维度: {self.dataset_info[dataset_type]['feature_dimension']}")
            print(f"  标签分布: {self.dataset_info[dataset_type]['label_distribution']}")
        
        self.processed_data = processed_datasets
        return processed_datasets
    
    def _collect_all_labels(self, datasets: Dict[str, List[Dict]]):
        """
        收集所有数据集中出现的标签
        """
        all_labels = set()
        for dataset_type, dataset in datasets.items():
            for sample in dataset:
                all_labels.add(sample['label'])
        
        self.all_labels = all_labels
        print(f"发现的所有标签: {self.all_labels}")
    
    def _preprocess_dataset(self, dataset: List[Dict], dataset_type: str) -> List[Dict]:
        """
        预处理单个数据集
        """
        processed_dataset = []
        
        for sample in dataset:
            try:
                processed_sample = sample.copy()
                
                # 提取特征
                features = sample['features']['full_features']
                
                # 转换为固定长度特征
                fixed_features = self._convert_to_fixed_length(features)
                
                # 存储处理后的特征
                processed_sample['processed_features'] = fixed_features
                processed_sample['original_feature_shape'] = features.shape
                processed_sample['processed_feature_shape'] = fixed_features.shape
                
                processed_dataset.append(processed_sample)
                
            except Exception as e:
                print(f"处理样本 {sample.get('file_name', 'unknown')} 时出错: {str(e)}")
                continue
        
        return processed_dataset
    
    def _convert_to_fixed_length(self, features: np.ndarray) -> np.ndarray:
        """
        将变长MFCC特征转换为固定长度
        """
        if self.fixed_length is not None:
            # 如果指定了固定长度，使用截断或填充
            n_frames = features.shape[1]
            if n_frames >= self.fixed_length:
                # 截断
                return features[:, :self.fixed_length].flatten()
            else:
                # 填充
                padded = np.zeros((features.shape[0], self.fixed_length))
                padded[:, :n_frames] = features
                return padded.flatten()
        else:
            # 使用统计特征作为固定长度表示
            mean_features = np.mean(features, axis=1)
            std_features = np.std(features, axis=1)
            max_features = np.max(features, axis=1)
            min_features = np.min(features, axis=1)
            median_features = np.median(features, axis=1)
            
            # 拼接所有统计特征
            statistical_features = np.concatenate([
                mean_features, 
                std_features, 
                max_features, 
                min_features,
                median_features
            ])
            
            return statistical_features
    
    def _calculate_feature_statistics(self, dataset: List[Dict]) -> Dict:
        """
        计算特征统计信息
        """
        if not dataset:
            return {}
        
        all_features = np.array([sample['processed_features'] for sample in dataset])
        
        stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0),
            'min': np.min(all_features, axis=0),
            'max': np.max(all_features, axis=0),
            'feature_dimension': all_features.shape[1],
            'sample_count': len(dataset)
        }
        
        return stats
    
    def _get_label_distribution(self, dataset: List[Dict]) -> Dict:
        """
        获取标签分布
        """
        labels = [sample['label'] for sample in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))
    
    def prepare_training_data(self, 
                            use_datasets: List[str] = ['train'],
                            balance_classes: bool = False) -> Tuple:
        """
        准备训练数据 - 关键修改：只在训练集上拟合标准化器
        """
        print("\n准备训练数据...")
        
        # 收集所有指定数据集的数据
        all_features = []
        all_labels = []
        
        for dataset_type in use_datasets:
            if dataset_type in self.processed_data:
                for sample in self.processed_data[dataset_type]:
                    all_features.append(sample['processed_features'])
                    all_labels.append(sample['label'])
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"训练数据形状: {X.shape}")
        
        # 关键修改：只在训练集上拟合标签编码器
        if len(y) > 0:
            self.label_encoder.fit(y)
            self.label_encoder_fitted = True
            y_encoded = self.label_encoder.transform(y)
            print(f"标签编码: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        else:
            print("警告: 训练数据为空!")
            y_encoded = np.array([])
        
        # 关键修改：只在训练集上拟合标准化器
        if self.normalize_features and len(X) > 0:
            print("正在拟合标准化器...")
            self.scaler.fit(X)  # 只在训练集上拟合
            self.scaler_fitted = True
            X = self.scaler.transform(X)
            print("特征已标准化（仅在训练集上拟合标准化器）")
        elif self.normalize_features:
            print("警告: 无法拟合标准化器，数据为空!")
        
        # 平衡类别（如果需要）
        if balance_classes and len(X) > 0:
            X, y_encoded = self._balance_classes(X, y_encoded)
        
        # 分割训练集和验证集
        if len(X) > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y_encoded
            )
            
            print(f"训练集形状: {X_train.shape}")
            print(f"验证集形状: {X_val.shape}")
            print(f"训练集标签分布: {dict(zip(self.label_encoder.classes_, np.bincount(y_train, minlength=len(self.label_encoder.classes_))))}")
            print(f"验证集标签分布: {dict(zip(self.label_encoder.classes_, np.bincount(y_val, minlength=len(self.label_encoder.classes_))))}")
        else:
            X_train, X_val, y_train, y_val = np.array([]), np.array([]), np.array([]), np.array([])
            print("警告: 训练数据为空，无法分割!")
        
        return X_train, X_val, y_train, y_val
    
    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡类别分布
        """
        from sklearn.utils import resample
        
        # 获取每个类别的样本
        unique_classes = np.unique(y)
        class_counts = [np.sum(y == c) for c in unique_classes]
        max_count = max(class_counts)
        
        X_balanced = []
        y_balanced = []
        
        for c in unique_classes:
            X_class = X[y == c]
            y_class = y[y == c]
            
            # 上采样到最大类别数量
            X_upsampled = resample(X_class, replace=True, n_samples=max_count, random_state=self.random_state)
            y_upsampled = np.full(max_count, c)
            
            X_balanced.append(X_upsampled)
            y_balanced.append(y_upsampled)
        
        # 合并所有类别的数据
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        
        # 打乱数据
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        print(f"类别平衡完成，每个类别 {max_count} 个样本")
        
        return X_balanced, y_balanced
    
    def prepare_evaluation_data(self, dataset_type: str = 'dev') -> Tuple:
        """
        准备评估数据（开发集/测试集）- 关键修改：使用训练集的标准化参数
        """
        print(f"\n准备 {dataset_type} 评估数据...")
        
        if dataset_type not in self.processed_data:
            raise ValueError(f"数据集 {dataset_type} 未找到")
        
        # 检查标准化器和编码器是否已拟合
        if self.normalize_features and not self.scaler_fitted:
            raise ValueError("标准化器尚未拟合，请先调用 prepare_training_data 方法")
        if not self.label_encoder_fitted:
            raise ValueError("标签编码器尚未拟合，请先调用 prepare_training_data 方法")
        
        eval_features = []
        eval_labels = []
        
        for sample in self.processed_data[dataset_type]:
            eval_features.append(sample['processed_features'])
            eval_labels.append(sample['label'])
        
        # 转换为numpy数组
        X_eval = np.array(eval_features)
        y_eval = np.array(eval_labels)
        
        print(f"原始 {dataset_type} 数据形状: {X_eval.shape}")
        
        # 检查评估集中是否有未知标签
        unique_eval_labels = set(y_eval)
        unknown_labels = unique_eval_labels - set(self.label_encoder.classes_)
        
        if unknown_labels:
            print(f"警告: {dataset_type} 数据集中发现未知标签: {unknown_labels}")
            # 可以选择过滤这些样本
            # 这里我们选择跳过这些样本
            valid_indices = [i for i, label in enumerate(y_eval) if label not in unknown_labels]
            X_eval = X_eval[valid_indices]
            y_eval = y_eval[valid_indices]
            print(f"已过滤 {len(unknown_labels)} 个未知标签的样本")
        
        # 编码标签（使用训练集拟合的编码器）
        y_eval_encoded = self.label_encoder.transform(y_eval)
        
        # 关键修改：只转换，不拟合（使用训练集的参数）
        if self.normalize_features:
            X_eval = self.scaler.transform(X_eval)
        
        print(f"{dataset_type} 集形状: {X_eval.shape}")
        print(f"{dataset_type} 集标签分布: {dict(zip(self.label_encoder.classes_, np.bincount(y_eval_encoded, minlength=len(self.label_encoder.classes_))))}")
        
        return X_eval, y_eval_encoded
    
    # 为了向后兼容，保留 prepare_test_data 方法
    def prepare_test_data(self, dataset_type: str = 'eval') -> Tuple:
        """准备测试数据（包装方法）"""
        return self.prepare_evaluation_data(dataset_type)
    
    def get_preprocessing_summary(self) -> Dict:
        """
        获取预处理摘要信息
        """
        summary = {
            'normalize_features': self.normalize_features,
            'feature_scaling_method': self.feature_scaling_method,
            'fixed_length': self.fixed_length,
            'test_size': self.test_size,
            'scaler_fitted': self.scaler_fitted,
            'label_encoder_fitted': self.label_encoder_fitted,
            'label_encoding': dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))) if self.label_encoder_fitted else {},
            'dataset_info': self.dataset_info,
            'feature_stats': {}
        }
        
        for dataset_type, stats in self.feature_stats.items():
            summary['feature_stats'][dataset_type] = {
                'feature_dimension': stats['feature_dimension'],
                'sample_count': stats['sample_count'],
                'global_mean': np.mean(stats['mean']),
                'global_std': np.mean(stats['std'])
            }
        
        return summary
    
    def save_preprocessor(self, file_path: str):
        """
        保存预处理器状态
        """
        save_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'normalize_features': self.normalize_features,
            'feature_scaling_method': self.feature_scaling_method,
            'fixed_length': self.fixed_length,
            'dataset_info': self.dataset_info,
            'all_labels': list(self.all_labels),
            'scaler_fitted': self.scaler_fitted,
            'label_encoder_fitted': self.label_encoder_fitted
        }
        
        joblib.dump(save_data, file_path)
        print(f"预处理器已保存到: {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """
        加载预处理器状态
        """
        import os
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return
        
        load_data = joblib.load(file_path)
        self.scaler = load_data['scaler']
        self.label_encoder = load_data['label_encoder']
        self.normalize_features = load_data['normalize_features']
        self.feature_scaling_method = load_data['feature_scaling_method']
        self.fixed_length = load_data['fixed_length']
        self.dataset_info = load_data['dataset_info']
        self.all_labels = set(load_data['all_labels'])
        self.scaler_fitted = load_data.get('scaler_fitted', False)
        self.label_encoder_fitted = load_data.get('label_encoder_fitted', False)
        
        print(f"预处理器已从 {file_path} 加载")