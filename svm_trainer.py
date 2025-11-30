import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class SVMTrainer:
    """
    SVM模型训练器
    用于训练和评估支持向量机模型
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 random_state: int = 42,
                 use_grid_search: bool = False):
        """
        初始化SVM训练器
        
        参数:
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数
            gamma: 核函数系数 ('scale', 'auto' 或数值)
            random_state: 随机种子
            use_grid_search: 是否使用网格搜索优化超参数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.use_grid_search = use_grid_search
        
        # 初始化模型和参数
        self.model = None
        self.best_params_ = None  # 确保总是初始化
        
        if not use_grid_search:
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                random_state=random_state,
                probability=True  # 启用概率预测
            )
        
        # 存储训练历史
        self.training_history = {}
        self.cv_scores = None
        
        # 新增：存储评估指标
        self.evaluation_results = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              cv_folds: int = 5) -> SVC:
        """
        训练SVM模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)
            cv_folds: 交叉验证折数
            
        返回:
            trained_model: 训练好的SVM模型
        """
        print("开始训练SVM模型...")
        
        if self.use_grid_search:
            # 使用网格搜索寻找最佳参数
            self._grid_search(X_train, y_train, cv_folds)
        else:
            # 直接训练模型
            print(f"使用参数训练SVM: kernel={self.kernel}, C={self.C}, gamma={self.gamma}")
            self.model.fit(X_train, y_train)
            print("模型训练完成!")
        
        # 计算训练集准确率
        train_accuracy = self.evaluate(X_train, y_train, "训练集")
        self.training_history['train_accuracy'] = train_accuracy
        
        # 如果有验证集，计算验证集准确率
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val, "验证集")
            self.training_history['val_accuracy'] = val_accuracy
        
        # 进行交叉验证
        self._cross_validate(X_train, y_train, cv_folds)
        
        return self.model
    
    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5):
        """
        使用网格搜索优化超参数
        """
        print("开始网格搜索优化超参数...")
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        # 创建网格搜索对象
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 保存最佳模型和参数
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        print(f"网格搜索完成! 最佳参数: {self.best_params_}")
        print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """
        执行交叉验证
        """
        print(f"执行{cv_folds}折交叉验证...")
        
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        self.cv_scores = cv_scores
        print(f"交叉验证准确率: {cv_scores}")
        print(f"平均交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "测试集") -> float:
        """
        评估模型性能
        
        参数:
            X: 特征数据
            y: 真实标签
            dataset_name: 数据集名称
            
        返回:
            accuracy: 准确率
        """
        # 预测
        y_pred = self.model.predict(X)
        
        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        
        print(f"{dataset_name}准确率: {accuracy:.4f}")
        
        return accuracy
    
    def detailed_evaluation(self, X: np.ndarray, y: np.ndarray, 
                          dataset_name: str = "测试集",
                          class_names: Optional[list] = None) -> Dict[str, Any]:
        """
        详细评估模型性能
        
        参数:
            X: 特征数据
            y: 真实标签
            dataset_name: 数据集名称
            class_names: 类别名称
            
        返回:
            metrics: 评估指标字典
        """
        print(f"\n{dataset_name}详细评估:")
        
        # 预测
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # 计算各种指标
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        # 打印结果
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y, y_pred, target_names=class_names))
        
        # 可视化混淆矩阵
        self._plot_confusion_matrix(cm, class_names, dataset_name)
        
        # 返回评估指标
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # 存储评估结果
        self.evaluation_results[dataset_name] = metrics
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[list], 
                             dataset_name: str):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{dataset_name}混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
            X: 特征数据
            
        返回:
            predictions: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        获取预测概率
        
        参数:
            X: 特征数据
            
        返回:
            probabilities: 预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.model.predict_proba(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        返回:
            info: 模型信息字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        info = {
            'model_type': 'SVM',
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'classes': self.model.classes_.tolist(),
            'training_history': self.training_history,
            'cv_scores': self.cv_scores.tolist() if self.cv_scores is not None else None,
            'evaluation_results': self.evaluation_results
        }
        
        # 安全地添加最佳参数（如果存在）
        if self.best_params_ is not None:
            info['best_params'] = self.best_params_
        
        return info
    
    def save_model(self, file_path: str):
        """
        保存训练好的模型
        
        参数:
            file_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, file_path)
        print(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path: str):
        """
        加载训练好的模型
        
        参数:
            file_path: 模型文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")
        
        self.model = joblib.load(file_path)
        print(f"模型已从 {file_path} 加载")
    
    # 新增方法：获取模型参数用于调优比较
    def get_model_params(self) -> Dict[str, Any]:
        """
        获取当前模型参数
        
        返回:
            params: 模型参数字典
        """
        if self.model is None:
            return {
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma
            }
        
        return {
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma
        }
    
    # 新增方法：快速评估（不绘制混淆矩阵）
    def quick_evaluate(self, X: np.ndarray, y: np.ndarray, 
                      dataset_name: str = "测试集") -> Dict[str, Any]:
        """
        快速评估模型性能（不绘制图形）
        
        参数:
            X: 特征数据
            y: 真实标签
            dataset_name: 数据集名称
            
        返回:
            metrics: 评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 预测
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # 计算各种指标
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        print(f"{dataset_name}快速评估 - 准确率: {accuracy:.4f}")
        
        # 返回评估指标
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # 存储评估结果
        self.evaluation_results[f"{dataset_name}_quick"] = metrics
        
        return metrics