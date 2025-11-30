import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class SVMTuner:
    """
    SVM模型参数调优器 - 改进版本
    防止过拟合，优化参数搜索策略，支持多种评估指标
    """
    
    def __init__(self, random_state: int = 42, tuning_mode: str = "balanced",
                 handle_imbalance: bool = True, scoring: str = "balanced_accuracy",
                 use_stratified_cv: bool = True):
        """
        初始化SVM调优器
        
        参数:
            random_state: 随机种子
            tuning_mode: 调优模式 - "fast", "balanced", "thorough"
            handle_imbalance: 是否处理不平衡数据
            scoring: 评分标准 - "accuracy", "f1", "roc_auc", "balanced_accuracy"
            use_stratified_cv: 是否使用分层交叉验证
        """
        self.random_state = random_state
        self.tuning_mode = tuning_mode
        self.handle_imbalance = handle_imbalance
        self.scoring = scoring
        self.use_stratified_cv = use_stratified_cv
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self.search_results = None
        self.cv_results = None
        self.evaluation_history = {}
        self.class_weights = None
        self.class_distribution = None
        
        # 根据模式设置配置
        self._set_tuning_config()
    
    def _set_tuning_config(self):
        """根据调优模式设置配置参数"""
        if self.tuning_mode == "fast":
            self.cv_folds_default = 3
            self.n_iter_default = 20
            self.param_density = "sparse"
        elif self.tuning_mode == "balanced":
            self.cv_folds_default = 5
            self.n_iter_default = 50
            self.param_density = "medium"
        else:  # thorough
            self.cv_folds_default = 5
            self.n_iter_default = 100
            self.param_density = "dense"
    
    def analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """
        分析类别分布
        
        参数:
            y: 标签数组
            
        返回:
            distribution_info: 类别分布信息
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        proportions = counts / total_samples
        
        distribution_info = {
            'classes': unique_classes,
            'counts': counts,
            'proportions': proportions,
            'total_samples': total_samples,
            'imbalance_ratio': max(proportions) / min(proportions) if len(proportions) > 1 else 1.0
        }
        
        print("\n=== 类别分布分析 ===")
        for i, cls in enumerate(unique_classes):
            print(f"类别 {cls}: {counts[i]} 个样本 ({proportions[i]:.2%})")
        
        print(f"不平衡比例: {distribution_info['imbalance_ratio']:.2f}")
        
        if distribution_info['imbalance_ratio'] > 2.0:
            print("警告: 数据集存在明显的不平衡问题!")
        
        self.class_distribution = distribution_info
        return distribution_info
    
    def compute_class_weights(self, y: np.ndarray, method: str = "balanced") -> Dict[int, float]:
        """
        计算类别权重
        
        参数:
            y: 标签数组
            method: 权重计算方法 - "balanced", "inverse", "custom"
            
        返回:
            class_weights: 类别权重字典
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        if method == "balanced":
            # 使用sklearn的平衡权重
            weights = compute_class_weight('balanced', classes=unique_classes, y=y)
            class_weights = dict(zip(unique_classes, weights))
        elif method == "inverse":
            # 基于频率的逆权重
            weights = total_samples / (len(unique_classes) * counts)
            class_weights = dict(zip(unique_classes, weights))
        else:
            # 自定义权重
            class_weights = 'balanced'
        
        print(f"\n类别权重 ({method}方法):")
        for cls, weight in class_weights.items():
            print(f"  类别 {cls}: {weight:.4f}")
        
        self.class_weights = class_weights
        return class_weights
    
    def comprehensive_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                           cv_folds: Optional[int] = None, 
                           n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        全面参数调优 - 改进版本，防止过拟合
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            cv_folds: 交叉验证折数
            n_iter: 随机搜索迭代次数
            
        返回:
            tuning_results: 调优结果字典
        """
        if cv_folds is None:
            cv_folds = self.cv_folds_default
        if n_iter is None:
            n_iter = self.n_iter_default
            
        print("开始SVM全面参数调优...")
        print(f"训练数据形状: {X_train.shape}")
        print(f"调优模式: {self.tuning_mode}")
        print(f"处理不平衡数据: {self.handle_imbalance}")
        print(f"评分标准: {self.scoring}")
        print(f"使用分层交叉验证: {self.use_stratified_cv}")
        
        # 分析类别分布
        dist_info = self.analyze_class_distribution(y_train)
        
        # 计算类别权重（如果需要）
        if self.handle_imbalance:
            self.compute_class_weights(y_train)
        
        # 使用改进的调优策略
        print("\n=== 使用改进的调优策略 ===")
        start_time = time.time()
        
        # 方法1: 智能分层调优 - 针对不平衡数据优化
        smart_results = self._improved_smart_tuning(X_train, y_train, cv_folds)
        smart_time = time.time() - start_time
        
        # 方法2: 随机搜索验证 - 针对不平衡数据优化
        start_time = time.time()
        random_results = self._improved_random_search(X_train, y_train, cv_folds, n_iter//2)
        random_time = time.time() - start_time
        
        # 方法3: 集成调优 - 结合前两种方法的结果
        start_time = time.time()
        ensemble_results = self._ensemble_tuning(X_train, y_train, cv_folds, 
                                               smart_results, random_results)
        ensemble_time = time.time() - start_time
        
        # 比较结果并选择最佳模型
        all_results = {
            'smart_tuning': smart_results,
            'random_search': random_results,
            'ensemble_tuning': ensemble_results
        }
        
        self._select_best_model(all_results)
        
        # 最终验证 - 使用指定的评分标准
        if self.scoring == "balanced_accuracy":
            scoring_func = 'balanced_accuracy'
        elif self.scoring == "f1":
            scoring_func = 'f1_macro'
        elif self.scoring == "roc_auc":
            scoring_func = 'roc_auc_ovo'
        else:
            scoring_func = 'accuracy'
        
        # 使用分层交叉验证进行最终评估
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=self.random_state)
        else:
            cv = min(cv_folds, 5)
            
        cv_scores = cross_val_score(self.best_model, X_train, y_train, 
                                   cv=cv, scoring=scoring_func)
        
        total_time = smart_time + random_time + ensemble_time
        
        # 存储调优结果
        self.tuning_results = {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_cv_score': cv_scores.mean(),
            'cv_scores': cv_scores,
            'all_results': all_results,
            'timing': {'total': total_time, 'smart': smart_time, 
                      'random': random_time, 'ensemble': ensemble_time},
            'class_distribution': dist_info,
            'class_weights': self.class_weights
        }
        
        print(f"\n=== 调优完成 ===")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"智能调优: {smart_time:.2f}秒, 随机搜索: {random_time:.2f}秒, 集成调优: {ensemble_time:.2f}秒")
        print(f"最佳参数: {self.best_params}")
        print(f"交叉验证{self.scoring}分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # 检查是否过拟合
        self._check_overfitting(X_train, y_train)
        
        return self.tuning_results
    
    def _improved_smart_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              cv_folds: int) -> Dict[str, Any]:
        """
        改进的智能分层调优 - 防止过拟合
        """
        print("执行改进的智能分层调优...")
        
        # 第一步：使用更稳健的核函数筛选
        print("步骤1: 稳健核函数筛选")
        best_kernel = self._robust_kernel_selection(X_train, y_train, cv_folds=3)
        print(f"最佳核函数: {best_kernel}")
        
        # 第二步：针对最佳核函数进行稳健调优
        print(f"步骤2: {best_kernel}核函数稳健调优")
        robust_results = self._robust_tune_single_kernel(best_kernel, X_train, y_train, cv_folds)
        
        return {
            'best_kernel': best_kernel,
            'best_estimator': robust_results['best_estimator'],
            'best_params': robust_results['best_params'],
            'best_score': robust_results['best_score']
        }
    
    def _robust_kernel_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                                cv_folds: int) -> str:
        """
        稳健核函数选择 - 使用多种评估指标
        """
        kernels = ['linear', 'rbf', 'poly']
        kernel_scores = {}
        
        # 使用分层交叉验证
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=min(cv_folds, 3), shuffle=True, random_state=self.random_state)
        else:
            cv = min(cv_folds, 3)
        
        for kernel in kernels:
            # 使用优化后的默认参数
            if kernel == 'linear':
                params = {'C': 1.0, 'kernel': 'linear'}
            elif kernel == 'rbf':
                params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            else:  # poly
                params = {'C': 1.0, 'gamma': 'scale', 'degree': 2, 'kernel': 'poly'}
            
            # 添加类别权重（如果需要）
            if self.handle_imbalance:
                params['class_weight'] = self.class_weights
            
            model = SVC(**params, random_state=self.random_state, probability=True)
            
            # 使用多种评分标准进行综合评估
            scores = {}
            
            # 主要评分标准
            if self.scoring == "balanced_accuracy":
                scoring_func = 'balanced_accuracy'
            elif self.scoring == "f1":
                scoring_func = 'f1_macro'
            elif self.scoring == "roc_auc":
                scoring_func = 'roc_auc_ovo'
            else:
                scoring_func = 'accuracy'
            
            main_score = cross_val_score(model, X_train, y_train, 
                                       cv=cv, scoring=scoring_func, n_jobs=-1).mean()
            
            # 辅助评分标准 - 准确率
            accuracy_score_val = cross_val_score(model, X_train, y_train, 
                                               cv=cv, scoring='accuracy', n_jobs=-1).mean()
            
            # 综合评分（主要评分占70%，准确率占30%）
            combined_score = 0.7 * main_score + 0.3 * accuracy_score_val
            
            kernel_scores[kernel] = {
                'main_score': main_score,
                'accuracy': accuracy_score_val,
                'combined_score': combined_score
            }
            
            print(f"  {kernel}核 - 主要分数: {main_score:.4f}, 准确率: {accuracy_score_val:.4f}, 综合分数: {combined_score:.4f}")
        
        # 选择综合分数最高的核函数
        best_kernel = max(kernel_scores.keys(), key=lambda k: kernel_scores[k]['combined_score'])
        return best_kernel
    
    def _robust_tune_single_kernel(self, kernel: str, X_train: np.ndarray, 
                                  y_train: np.ndarray, cv_folds: int) -> Dict[str, Any]:
        """
        对单个核函数进行稳健调优 - 防止过拟合
        """
        # 获取优化后的参数网格
        param_grid = self._get_robust_param_grid(kernel)
        
        print(f"  参数网格大小: {self._count_param_combinations(param_grid)}")
        
        # 设置评分标准
        if self.scoring == "balanced_accuracy":
            scoring_func = 'balanced_accuracy'
        elif self.scoring == "f1":
            scoring_func = 'f1_macro'
        elif self.scoring == "roc_auc":
            scoring_func = 'roc_auc_ovo'
        else:
            scoring_func = 'accuracy'
        
        # 使用分层交叉验证
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = cv_folds
        
        grid_search = GridSearchCV(
            SVC(kernel=kernel, random_state=self.random_state, probability=True),
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def _get_robust_param_grid(self, kernel: str) -> Dict[str, List]:
        """
        获取稳健的参数网格 - 防止过拟合
        """
        # 基础参数网格
        if kernel == 'linear':
            base_grid = {
                'C': np.logspace(-3, 3, 13),  # 扩大搜索范围
                'class_weight': [None, 'balanced'] if self.handle_imbalance else [None]
            }
        elif kernel == 'rbf':
            base_grid = {
                'C': np.logspace(-3, 3, 13),  # 扩大搜索范围
                'gamma': np.logspace(-4, 2, 13),  # 扩大搜索范围
                'class_weight': [None, 'balanced'] if self.handle_imbalance else [None]
            }
        else:  # poly
            base_grid = {
                'C': np.logspace(-2, 2, 9),
                'gamma': np.logspace(-4, 1, 10),
                'degree': [2, 3],
                'coef0': [0.0, 0.5, 1.0],
                'class_weight': [None, 'balanced'] if self.handle_imbalance else [None]
            }
        
        # 如果处理不平衡数据，添加自定义权重选项
        if self.handle_imbalance and self.class_weights is not None:
            base_grid['class_weight'].append(self.class_weights)
        
        return base_grid
    
    def _improved_random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                               cv_folds: int, n_iter: int) -> Dict[str, Any]:
        """
        改进的随机搜索 - 防止过拟合
        """
        print("执行改进的随机搜索...")
        
        # 优化参数分布
        param_dist = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': np.logspace(-3, 3, 20),  # 扩大搜索范围
            'gamma': np.logspace(-4, 2, 20),  # 扩大搜索范围
            'degree': [2, 3],
            'coef0': [0.0, 0.5, 1.0],
        }
        
        # 添加类别权重选项
        if self.handle_imbalance:
            class_weight_options = [None, 'balanced']
            if self.class_weights is not None:
                class_weight_options.append(self.class_weights)
            param_dist['class_weight'] = class_weight_options
        
        # 设置评分标准
        if self.scoring == "balanced_accuracy":
            scoring_func = 'balanced_accuracy'
        elif self.scoring == "f1":
            scoring_func = 'f1_macro'
        elif self.scoring == "roc_auc":
            scoring_func = 'roc_auc_ovo'
        else:
            scoring_func = 'accuracy'
        
        # 使用分层交叉验证
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = cv_folds

        random_search = RandomizedSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"随机搜索完成，耗时: {end_time - start_time:.2f}秒")
        print(f"最佳参数: {random_search.best_params_}")
        print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")
        
        return {
            'best_estimator': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def _ensemble_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv_folds: int, smart_results: Dict, random_results: Dict) -> Dict[str, Any]:
        """
        集成调优 - 结合多种调优方法的结果
        """
        print("执行集成调优...")
        
        # 收集所有候选模型
        candidates = [
            ('smart', smart_results['best_estimator'], smart_results['best_score']),
            ('random', random_results['best_estimator'], random_results['best_score'])
        ]
        
        # 评估每个候选模型的稳健性
        best_score = -1
        best_estimator = None
        best_method = None
        
        for method, estimator, score in candidates:
            # 使用分层交叉验证进行稳健性评估
            if self.use_stratified_cv:
                cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=self.random_state)
            else:
                cv = min(cv_folds, 5)
            
            # 使用多种评分标准
            scores = []
            
            # 主要评分标准
            if self.scoring == "balanced_accuracy":
                scoring_func = 'balanced_accuracy'
            elif self.scoring == "f1":
                scoring_func = 'f1_macro'
            elif self.scoring == "roc_auc":
                scoring_func = 'roc_auc_ovo'
            else:
                scoring_func = 'accuracy'
            
            main_scores = cross_val_score(estimator, X_train, y_train, 
                                        cv=cv, scoring=scoring_func)
            
            # 准确率评分
            accuracy_scores = cross_val_score(estimator, X_train, y_train, 
                                            cv=cv, scoring='accuracy')
            
            # 综合评分（考虑均值和方差）
            main_mean = main_scores.mean()
            main_std = main_scores.std()
            accuracy_mean = accuracy_scores.mean()
            
            # 稳健性评分（均值高，方差低）
            robustness_score = main_mean / (1 + main_std) + 0.2 * accuracy_mean
            
            print(f"  {method}方法 - 主要分数: {main_mean:.4f} (±{main_std:.4f}), 稳健性评分: {robustness_score:.4f}")
            
            if robustness_score > best_score:
                best_score = robustness_score
                best_estimator = estimator
                best_method = method
        
        # 获取最佳参数
        if best_method == 'smart':
            best_params = smart_results['best_params']
        else:
            best_params = random_results['best_params']
        
        print(f"集成调优选择: {best_method}方法")
        
        return {
            'best_estimator': best_estimator,
            'best_params': best_params,
            'best_score': best_score,
            'selected_method': best_method
        }
    
    def _count_param_combinations(self, param_grid: Dict[str, List]) -> int:
        """计算参数组合数量"""
        from functools import reduce
        import operator
        return reduce(operator.mul, [len(v) for v in param_grid.values()], 1)
    
    def _select_best_model(self, all_results: Dict[str, Any]):
        """
        从所有调优结果中选择最佳模型
        """
        best_score = 0
        best_method = None
        
        for method, results in all_results.items():
            if results['best_score'] > best_score:
                best_score = results['best_score']
                best_method = method
                self.best_model = results['best_estimator']
                self.best_params = results['best_params']
        
        self.best_score = best_score
        print(f"\n最终选择方法: {best_method}")
        print(f"最佳交叉验证分数: {best_score:.4f}")
    
    def _check_overfitting(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        检查是否过拟合
        """
        print("\n=== 过拟合检查 ===")
        
        # 计算训练集分数
        train_score = self.best_model.score(X_train, y_train)
        
        # 计算交叉验证分数
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            cv = 5
            
        if self.scoring == "balanced_accuracy":
            scoring_func = 'balanced_accuracy'
        elif self.scoring == "f1":
            scoring_func = 'f1_macro'
        elif self.scoring == "roc_auc":
            scoring_func = 'roc_auc_ovo'
        else:
            scoring_func = 'accuracy'
            
        cv_scores = cross_val_score(self.best_model, X_train, y_train, 
                                   cv=cv, scoring=scoring_func)
        cv_score = cv_scores.mean()
        
        # 计算过拟合程度
        overfitting_gap = train_score - cv_score
        
        print(f"训练集分数: {train_score:.4f}")
        print(f"交叉验证分数: {cv_score:.4f}")
        print(f"过拟合差距: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            print("警告: 可能存在过拟合!")
        elif overfitting_gap < 0.01:
            print("模型泛化能力良好")
        else:
            print("模型表现正常")
    
    def quick_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                    cv_folds: int = 3) -> Dict[str, Any]:
        """
        快速调优 - 改进版本，防止过拟合
        """
        print("执行快速调优...")
        
        # 分析类别分布
        self.analyze_class_distribution(y_train)
        
        # 计算类别权重（如果需要）
        if self.handle_imbalance:
            self.compute_class_weights(y_train)
        
        # 只使用随机搜索
        param_dist = {
            'kernel': ['linear', 'rbf'],
            'C': np.logspace(-2, 2, 10),
            'gamma': np.logspace(-3, 1, 10),
        }
        
        # 添加类别权重选项
        if self.handle_imbalance:
            class_weight_options = [None, 'balanced']
            if self.class_weights is not None:
                class_weight_options.append(self.class_weights)
            param_dist['class_weight'] = class_weight_options
        
        # 设置评分标准
        if self.scoring == "balanced_accuracy":
            scoring_func = 'balanced_accuracy'
        elif self.scoring == "f1":
            scoring_func = 'f1_macro'
        elif self.scoring == "roc_auc":
            scoring_func = 'roc_auc_ovo'
        else:
            scoring_func = 'accuracy'
        
        # 使用分层交叉验证
        if self.use_stratified_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = cv_folds
        
        random_search = RandomizedSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_dist,
            n_iter=15,  # 增加迭代次数
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        end_time = time.time()
        
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        print(f"快速调优完成，耗时: {end_time - start_time:.2f}秒")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳分数: {self.best_score:.4f}")
        
        # 检查过拟合
        self._check_overfitting(X_train, y_train)
        
        return {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_score': self.best_score
        }
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray, 
                        class_names: Optional[List[str]] = None,
                        dataset_name: str = "测试集") -> Dict[str, Any]:
        """
        在测试集上评估最佳模型 - 改进版本，提供更多指标
        """
        if self.best_model is None:
            raise ValueError("请先进行参数调优")
        
        print(f"\n=== {dataset_name}评估 ===")
        
        # 预测
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # 计算基础指标
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        
        # 计算不平衡数据相关指标
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # 计算AUC（如果是二分类）
        if len(np.unique(y_test)) == 2:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
        
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # 打印结果
        print(f"{dataset_name}准确率: {accuracy:.4f}")
        print(f"{dataset_name}平衡准确率: {balanced_accuracy:.4f}")
        print(f"{dataset_name}F1分数: {f1:.4f}")
        print(f"{dataset_name}精确率: {precision:.4f}")
        print(f"{dataset_name}召回率: {recall:.4f}")
        print(f"{dataset_name}AUC: {auc_score:.4f}")
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # 可视化混淆矩阵
        self._plot_confusion_matrix(cm, class_names, dataset_name)
        
        # 存储评估结果
        eval_results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.evaluation_history[dataset_name] = eval_results
        
        return eval_results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: Optional[List[str]],
                             dataset_name: str):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(8, 6))
        
        # 处理class_names为None的情况
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始数值
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title(f'{dataset_name}混淆矩阵 - 原始数值')
        ax1.set_ylabel('真实标签')
        ax1.set_xlabel('预测标签')
        
        # 百分比
        sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title(f'{dataset_name}混淆矩阵 - 百分比')
        ax2.set_ylabel('真实标签')
        ax2.set_xlabel('预测标签')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, y_train: np.ndarray, y_test: np.ndarray = None):
        """
        绘制类别分布图
        """
        fig, axes = plt.subplots(1, 2 if y_test is not None else 1, figsize=(12, 5))
        
        if y_test is not None:
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None
        
        # 训练集分布
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        ax1.bar(train_unique, train_counts, color='skyblue', alpha=0.7)
        ax1.set_title('训练集类别分布')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('样本数量')
        
        # 添加数值标签
        for i, v in enumerate(train_counts):
            ax1.text(i, v + max(train_counts)*0.01, str(v), ha='center')
        
        # 测试集分布（如果提供）
        if y_test is not None:
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            ax2.bar(test_unique, test_counts, color='lightcoral', alpha=0.7)
            ax2.set_title('测试集类别分布')
            ax2.set_xlabel('类别')
            ax2.set_ylabel('样本数量')
            
            # 添加数值标签
            for i, v in enumerate(test_counts):
                ax2.text(i, v + max(test_counts)*0.01, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_importance(self, all_results: Dict[str, Any]):
        """
        可视化参数重要性 - 新增方法
        """
        try:
            # 从随机搜索结果中提取参数重要性
            if 'random_search' in all_results and 'cv_results' in all_results['random_search']:
                cv_results = all_results['random_search']['cv_results']
                
                # 创建参数重要性图
                param_names = [key for key in cv_results.keys() if key.startswith('param_')]
                mean_scores = cv_results['mean_test_score']
                
                # 分析参数对性能的影响
                param_importance = {}
                for param in param_names:
                    unique_values = list(set(cv_results[param]))
                    if len(unique_values) > 1:  # 只分析有变化的参数
                        scores_by_param = {}
                        for i, value in enumerate(cv_results[param]):
                            if value not in scores_by_param:
                                scores_by_param[value] = []
                            scores_by_param[value].append(mean_scores[i])
                        
                        # 计算参数值的范围
                        max_score = max([np.mean(scores) for scores in scores_by_param.values()])
                        min_score = min([np.mean(scores) for scores in scores_by_param.values()])
                        param_importance[param.replace('param_', '')] = max_score - min_score
                
                if param_importance:
                    plt.figure(figsize=(10, 6))
                    params = list(param_importance.keys())
                    importance = list(param_importance.values())
                    
                    plt.barh(params, importance)
                    plt.xlabel('参数重要性 (评分范围)')
                    plt.title('SVM参数重要性分析')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("无法计算参数重要性：参数值变化不足")
            else:
                print("随机搜索结果不可用，无法绘制参数重要性图")
                
        except Exception as e:
            print(f"绘制参数重要性图时出错: {e}")
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """
        获取调优摘要信息
        """
        if not hasattr(self, 'tuning_results'):
            return {"error": "尚未进行调优"}
        
        summary = {
            'tuning_mode': self.tuning_mode,
            'handle_imbalance': self.handle_imbalance,
            'scoring': self.scoring,
            'use_stratified_cv': self.use_stratified_cv,
            'best_params': self.best_params,
            'best_cv_score': self.tuning_results.get('best_cv_score', 0),
            'timing': self.tuning_results.get('timing', {}),
            'class_distribution': self.class_distribution,
            'class_weights': self.class_weights,
            'evaluation_history': self.evaluation_history
        }
        
        return summary
    
    def save_tuned_model(self, file_path: str):
        """
        保存调优后的模型和相关信息
        """
        if self.best_model is None:
            raise ValueError("没有可保存的模型")
        
        save_data = {
            'model': self.best_model,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuning_mode': self.tuning_mode,
            'handle_imbalance': self.handle_imbalance,
            'scoring': self.scoring,
            'use_stratified_cv': self.use_stratified_cv,
            'random_state': self.random_state,
            'tuning_results': getattr(self, 'tuning_results', None),
            'class_distribution': self.class_distribution,
            'class_weights': self.class_weights,
            'evaluation_history': self.evaluation_history
        }
        
        joblib.dump(save_data, file_path)
        print(f"调优后的模型和相关信息已保存到: {file_path}")
    
    def load_tuned_model(self, file_path: str):
        """
        加载调优后的模型和相关信息
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")
        
        load_data = joblib.load(file_path)
        self.best_model = load_data['model']
        self.best_params = load_data['best_params']
        self.best_score = load_data['best_score']
        self.tuning_mode = load_data.get('tuning_mode', 'balanced')
        self.handle_imbalance = load_data.get('handle_imbalance', True)
        self.scoring = load_data.get('scoring', 'balanced_accuracy')
        self.use_stratified_cv = load_data.get('use_stratified_cv', True)
        self.random_state = load_data.get('random_state', 42)
        self.tuning_results = load_data.get('tuning_results')
        self.class_distribution = load_data.get('class_distribution')
        self.class_weights = load_data.get('class_weights')
        self.evaluation_history = load_data.get('evaluation_history', {})
        
        print(f"调优后的模型和相关信息已从 {file_path} 加载")
    
    def compare_with_baseline(self, baseline_model: Any, 
                            X_test: np.ndarray, y_test: np.ndarray,
                            baseline_name: str = "基线模型") -> Dict[str, Any]:
        """
        与基线模型进行比较 - 改进版本，提供更多指标
        """
        if self.best_model is None:
            raise ValueError("请先进行参数调优")
        
        print(f"\n=== 模型比较: {baseline_name} vs 调优后模型 ===")
        
        # 评估基线模型
        baseline_pred = baseline_model.predict(X_test)
        baseline_pred_proba = baseline_model.predict_proba(X_test) if hasattr(baseline_model, 'predict_proba') else None
        
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_balanced_accuracy = balanced_accuracy_score(y_test, baseline_pred)
        baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
        
        # 评估调优模型
        tuned_pred = self.best_model.predict(X_test)
        tuned_pred_proba = self.best_model.predict_proba(X_test)
        
        tuned_accuracy = accuracy_score(y_test, tuned_pred)
        tuned_balanced_accuracy = balanced_accuracy_score(y_test, tuned_pred)
        tuned_f1 = f1_score(y_test, tuned_pred, average='weighted')
        
        # 计算AUC（如果可能）
        if baseline_pred_proba is not None and len(np.unique(y_test)) == 2:
            baseline_auc = roc_auc_score(y_test, baseline_pred_proba[:, 1])
            tuned_auc = roc_auc_score(y_test, tuned_pred_proba[:, 1])
        else:
            baseline_auc = None
            tuned_auc = None
        
        accuracy_improvement = tuned_accuracy - baseline_accuracy
        balanced_accuracy_improvement = tuned_balanced_accuracy - baseline_balanced_accuracy
        f1_improvement = tuned_f1 - baseline_f1
        
        print(f"{baseline_name}准确率: {baseline_accuracy:.4f}")
        print(f"调优后模型准确率: {tuned_accuracy:.4f}")
        print(f"准确率提升: {accuracy_improvement:+.4f}")
        
        print(f"{baseline_name}平衡准确率: {baseline_balanced_accuracy:.4f}")
        print(f"调优后模型平衡准确率: {tuned_balanced_accuracy:.4f}")
        print(f"平衡准确率提升: {balanced_accuracy_improvement:+.4f}")
        
        print(f"{baseline_name}F1分数: {baseline_f1:.4f}")
        print(f"调优后模型F1分数: {tuned_f1:.4f}")
        print(f"F1分数提升: {f1_improvement:+.4f}")
        
        if baseline_auc is not None:
            print(f"{baseline_name}AUC: {baseline_auc:.4f}")
            print(f"调优后模型AUC: {tuned_auc:.4f}")
            print(f"AUC提升: {tuned_auc - baseline_auc:+.4f}")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'tuned_accuracy': tuned_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'baseline_balanced_accuracy': baseline_balanced_accuracy,
            'tuned_balanced_accuracy': tuned_balanced_accuracy,
            'balanced_accuracy_improvement': balanced_accuracy_improvement,
            'baseline_f1': baseline_f1,
            'tuned_f1': tuned_f1,
            'f1_improvement': f1_improvement,
            'baseline_auc': baseline_auc,
            'tuned_auc': tuned_auc
        }


# 使用示例
def example_usage():
    """
    改进后类的使用示例 - 演示防止过拟合
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成不平衡的示例数据
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_classes=2, weights=[0.9, 0.1],
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 不同模式的速度对比
    modes = ["fast", "balanced"]
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"调优模式: {mode}")
        print(f"{'='*50}")
        
        # 创建调优器，启用不平衡数据处理和分层交叉验证
        tuner = SVMTuner(random_state=42, tuning_mode=mode,
                        handle_imbalance=True, scoring="balanced_accuracy",
                        use_stratified_cv=True)
        
        # 绘制类别分布
        tuner.plot_class_distribution(y_train, y_test)
        
        start_time = time.time()
        results = tuner.comprehensive_tuning(X_train, y_train)
        total_time = time.time() - start_time
        
        print(f"{mode}模式总耗时: {total_time:.2f}秒")
        print(f"最佳平衡准确率: {results['best_cv_score']:.4f}")
        
        # 测试集评估
        test_results = tuner.evaluate_on_test(X_test, y_test, 
                                            class_names=['Class0', 'Class1'],
                                            dataset_name="测试集")
        
        # 获取调优摘要
        summary = tuner.get_tuning_summary()
        print(f"调优摘要: {summary}")

if __name__ == "__main__":
    example_usage()