import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import det_curve, DetCurveDisplay, accuracy_score, f1_score
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import joblib
import os

# 设置全局字体为仿宋
plt.rcParams['font.sans-serif'] = ['KaiTi']  
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ModelEvaluator:
    """
    模型评估器
    用于计算EER和绘制ROC曲线等评估指标
    """
    
    def __init__(self, model: Any, preprocessor: Any):
        """
        初始化模型评估器
        
        参数:
            model: 训练好的模型
            preprocessor: 数据预处理器
        """
        self.model = model
        self.preprocessor = preprocessor
        self.evaluation_results = {}
        self.comparison_results = {}  # 新增：存储比较结果
    
    def compute_eer(self, y_true: np.ndarray, y_scores: np.ndarray, 
                   positive_class: int = 1) -> Tuple[float, float, np.ndarray]:
        """
        计算等错误率 (EER)
        
        参数:
            y_true: 真实标签
            y_scores: 预测分数（通常是正类的概率）
            positive_class: 正类标签
            
        返回:
            eer: 等错误率
            eer_threshold: EER对应的阈值
            fpr: 假正率数组
        """
        # 输入验证
        if y_true is None or y_scores is None:
            raise ValueError("y_true 和 y_scores 不能为 None")
        
        if len(y_true) != len(y_scores):
            raise ValueError("y_true 和 y_scores 长度不一致")
        
        # 确保y_scores是正类的概率
        if y_scores.ndim > 1:
            if positive_class >= y_scores.shape[1]:
                raise ValueError(f"positive_class {positive_class} 超出范围")
            y_scores = y_scores[:, positive_class]
        
        # 计算FPR和TPR
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=positive_class)
        
        # 寻找EER点（FPR = 1 - TPR）
        fnr = 1 - tpr  # 假负率
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        eer_threshold = thresholds[eer_index]


        
        return eer, eer_threshold, fpr
    
    

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      positive_class: int = 1, title: str = "ROC曲线",
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制ROC曲线并计算AUC
        
        参数:
            y_true: 真实标签
            y_scores: 预测分数
            positive_class: 正类标签
            title: 图表标题
            save_path: 图片保存路径（可选）
            
        返回:
            roc_metrics: ROC相关指标
        """
        # 确保y_scores是正类的概率
        if y_scores.ndim > 1:
            y_scores = y_scores[:, positive_class]
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=positive_class)
        roc_auc = auc(fpr, tpr)
        
        # 计算EER
        eer, eer_threshold, _ = self.compute_eer(y_true, y_scores, positive_class)
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
        
        # 标记EER点
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        plt.plot(fpr[eer_index], tpr[eer_index], 'ro', markersize=8, label=f'EER点 (EER = {eer:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        
        plt.show()
        
        # 返回ROC指标
        roc_metrics = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold
        }
        
        self.evaluation_results['roc'] = roc_metrics
        
        return roc_metrics
    
    def plot_det_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      positive_class: int = 1, title: str = "DET曲线",
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制DET曲线（Detection Error Tradeoff）
        
        参数:
            y_true: 真实标签
            y_scores: 预测分数
            positive_class: 正类标签
            title: 图表标题
            save_path: 图片保存路径（可选）
            
        返回:
            det_metrics: DET相关指标
        """
        # 确保y_scores是正类的概率
        if y_scores.ndim > 1:
            y_scores = y_scores[:, positive_class]
        
        # 计算DET曲线
        fpr, fnr, thresholds = det_curve(y_true, y_scores, pos_label=positive_class)
        
        # 绘制DET曲线
        plt.figure(figsize=(10, 8))
        
        display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="SVM")
        display.plot()
        plt.title(title)
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('假负率 (False Negative Rate)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DET曲线已保存到: {save_path}")
        
        plt.show()
        
        # 返回DET指标
        det_metrics = {
            'fpr': fpr,
            'fnr': fnr,
            'thresholds': thresholds
        }
        
        self.evaluation_results['det'] = det_metrics
        return det_metrics
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                  positive_class: int = 1, title: str = "精确率-召回率曲线",
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制精确率-召回率曲线
        
        参数:
            y_true: 真实标签
            y_scores: 预测分数
            positive_class: 正类标签
            title: 图表标题
            save_path: 图片保存路径（可选）
            
        返回:
            pr_metrics: PR相关指标
        """
        # 确保y_scores是正类的概率
        if y_scores.ndim > 1:
            y_scores = y_scores[:, positive_class]
        
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=positive_class)
        average_precision = average_precision_score(y_true, y_scores, pos_label=positive_class)
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR曲线 (AP = {average_precision:.4f})')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存到: {save_path}")
        
        plt.show()
        
        # 返回PR指标
        pr_metrics = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': average_precision
        }
        
        self.evaluation_results['pr'] = pr_metrics
        return pr_metrics
    
    def plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray,
                              positive_class: int = 1, title: str = "得分分布",
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制正负类得分分布
        
        参数:
            y_true: 真实标签
            y_scores: 预测分数
            positive_class: 正类标签
            title: 图表标题
            save_path: 图片保存路径（可选）
            
        返回:
            distribution_stats: 分布统计信息
        """
        # 确保y_scores是正类的概率
        if y_scores.ndim > 1:
            y_scores = y_scores[:, positive_class]
        
        # 分离正负类得分
        positive_scores = y_scores[y_true == positive_class]
        negative_scores = y_scores[y_true != positive_class]
        
        # 绘制分布图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(positive_scores, bins=50, alpha=0.7, color='green', label='正类')
        plt.hist(negative_scores, bins=50, alpha=0.7, color='red', label='负类')
        plt.xlabel('得分')
        plt.ylabel('频数')
        plt.title(f'{title} - 直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        sns.kdeplot(positive_scores, fill=True, color='green', label='正类', alpha=0.7)
        sns.kdeplot(negative_scores, fill=True, color='red', label='负类', alpha=0.7)
        plt.xlabel('得分')
        plt.ylabel('密度')
        plt.title(f'{title} - 密度图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"得分分布图已保存到: {save_path}")
        
        plt.show()
        
        # 计算统计信息
        pos_mean = np.mean(positive_scores)
        pos_std = np.std(positive_scores)
        neg_mean = np.mean(negative_scores)
        neg_std = np.std(negative_scores)
        
        # 打印统计信息
        print(f"正类得分统计: 均值={pos_mean:.4f}, 标准差={pos_std:.4f}")
        print(f"负类得分统计: 均值={neg_mean:.4f}, 标准差={neg_std:.4f}")
        
        distribution_stats = {
            'positive_mean': pos_mean,
            'positive_std': pos_std,
            'negative_mean': neg_mean,
            'negative_std': neg_std,
            'separation': abs(pos_mean - neg_mean) / ((pos_std + neg_std) / 2)  # 分离度
        }
        
        self.evaluation_results['score_distribution'] = distribution_stats
        return distribution_stats
    
    def comprehensive_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                              positive_class: int = 1, dataset_name: str = "测试集",
                              save_plots: bool = False, plot_prefix: str = "") -> Dict[str, Any]:
        """
        综合评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            positive_class: 正类标签
            dataset_name: 数据集名称
            save_plots: 是否保存图表
            plot_prefix: 图表文件名前缀
            
        返回:
            evaluation_results: 综合评估结果
        """
        print(f"\n=== {dataset_name}综合评估 ===")
        
        # 预测概率
        y_scores = self.model.predict_proba(X_test)
        
        # 设置保存路径
        save_paths = {}
        if save_plots:
            os.makedirs('evaluation_plots', exist_ok=True)
            prefix = f"evaluation_plots/{plot_prefix}_{dataset_name}" if plot_prefix else f"evaluation_plots/{dataset_name}"
            save_paths = {
                'roc': f"{prefix}_roc.png",
                'det': f"{prefix}_det.png",
                'pr': f"{prefix}_pr.png",
                'distribution': f"{prefix}_distribution.png"
            }
        
        # 计算各种评估指标
        roc_metrics = self.plot_roc_curve(y_test, y_scores, positive_class, 
                                         f"{dataset_name} ROC曲线",
                                         save_paths.get('roc'))
        
        det_metrics = self.plot_det_curve(y_test, y_scores, positive_class,
                                        f"{dataset_name} DET曲线",
                                        save_paths.get('det'))
        
        pr_metrics = self.plot_precision_recall_curve(y_test, y_scores, positive_class,
                                                    f"{dataset_name} 精确率-召回率曲线",
                                                    save_paths.get('pr'))
        
        # 绘制得分分布
        distribution_stats = self.plot_score_distribution(y_test, y_scores, positive_class,
                                                        f"{dataset_name} 得分分布",
                                                        save_paths.get('distribution'))
        
        # 打印关键指标
        print(f"\n关键性能指标:")
        print(f"  AUC (ROC曲线下面积): {roc_metrics['auc']:.4f}")
        print(f"  EER (等错误率): {roc_metrics['eer']:.4f}")
        print(f"  EER阈值: {roc_metrics['eer_threshold']:.4f}")
        print(f"  AP (平均精确率): {pr_metrics['average_precision']:.4f}")
        print(f"  得分分离度: {distribution_stats['separation']:.4f}")
        
        # 计算在EER阈值下的性能
        y_pred_eer = (y_scores[:, positive_class] >= roc_metrics['eer_threshold']).astype(int)
        eer_accuracy = accuracy_score(y_test, y_pred_eer)
        eer_f1 = f1_score(y_test, y_pred_eer, pos_label=positive_class)
        
        print(f"\n在EER阈值下的性能:")
        print(f"  准确率: {eer_accuracy:.4f}")
        print(f"  F1分数: {eer_f1:.4f}")
        
        # 综合评估结果
        comprehensive_results = {
            'roc_metrics': roc_metrics,
            'det_metrics': det_metrics,
            'pr_metrics': pr_metrics,
            'distribution_stats': distribution_stats,
            'eer_accuracy': eer_accuracy,
            'eer_f1': eer_f1,
            'y_scores': y_scores,
            'dataset_name': dataset_name
        }
        
        self.evaluation_results['comprehensive'] = comprehensive_results
        return comprehensive_results
    
    def compare_models(self, models_dict: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray, positive_class: int = 1,
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        比较多个模型的ROC曲线
        
        参数:
            models_dict: 模型字典 {模型名称: 模型对象}
            X_test: 测试特征
            y_test: 测试标签
            positive_class: 正类标签
            save_path: 图片保存路径（可选）
            
        返回:
            comparison_results: 比较结果
        """
        plt.figure(figsize=(10, 8))
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            try:
                # 预测概率
                y_scores = model.predict_proba(X_test)
                if y_scores.ndim > 1:
                    y_scores = y_scores[:, positive_class]
                
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=positive_class)
                roc_auc = auc(fpr, tpr)
                
                # 计算EER
                eer, _, _ = self.compute_eer(y_test, y_scores, positive_class)
                
                # 绘制ROC曲线
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f}, EER = {eer:.4f})')
                
                # 存储结果
                comparison_results[model_name] = {
                    'auc': roc_auc,
                    'eer': eer,
                    'fpr': fpr,
                    'tpr': tpr
                }
                
            except Exception as e:
                print(f"评估模型 {model_name} 时出错: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('多模型ROC曲线比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型比较图已保存到: {save_path}")
        
        plt.show()
        
        # 打印比较结果
        print("\n模型性能比较:")
        for model_name, metrics in comparison_results.items():
            print(f"  {model_name}: AUC = {metrics['auc']:.4f}, EER = {metrics['eer']:.4f}")
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要信息
        
        返回:
            summary: 评估摘要
        """
        if 'comprehensive' not in self.evaluation_results:
            return {"error": "尚未进行综合评估"}
        
        comp = self.evaluation_results['comprehensive']
        
        summary = {
            'dataset_name': comp.get('dataset_name', '未知'),
            'auc': comp['roc_metrics']['auc'],
            'eer': comp['roc_metrics']['eer'],
            'eer_threshold': comp['roc_metrics']['eer_threshold'],
            'average_precision': comp['pr_metrics']['average_precision'],
            'eer_accuracy': comp['eer_accuracy'],
            'eer_f1': comp['eer_f1'],
            'score_separation': comp['distribution_stats']['separation']
        }
        
        return summary
    
    def save_evaluation_results(self, file_path: str):
        """
        保存评估结果
        
        参数:
            file_path: 保存路径
        """
        # 创建保存数据（避免保存大型数组）
        save_data = {
            'evaluation_summary': self.get_evaluation_summary(),
            'comprehensive_results': {
                'auc': self.evaluation_results.get('comprehensive', {}).get('roc_metrics', {}).get('auc', 0),
                'eer': self.evaluation_results.get('comprehensive', {}).get('roc_metrics', {}).get('eer', 0),
                'average_precision': self.evaluation_results.get('comprehensive', {}).get('pr_metrics', {}).get('average_precision', 0),
                'eer_accuracy': self.evaluation_results.get('comprehensive', {}).get('eer_accuracy', 0),
                'eer_f1': self.evaluation_results.get('comprehensive', {}).get('eer_f1', 0)
            },
            'comparison_results': self.comparison_results,
            'model_type': str(type(self.model)),
            'preprocessor_info': str(type(self.preprocessor))
        }
        
        joblib.dump(save_data, file_path)
        print(f"评估结果已保存到: {file_path}")
    
    def load_evaluation_results(self, file_path: str) -> Dict[str, Any]:
        """
        加载评估结果
        
        参数:
            file_path: 文件路径
            
        返回:
            loaded_data: 加载的数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        loaded_data = joblib.load(file_path)
        print(f"评估结果已从 {file_path} 加载")
        
        # 更新比较结果
        if 'comparison_results' in loaded_data:
            self.comparison_results = loaded_data['comparison_results']
        
        return loaded_data
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成评估报告
        
        参数:
            output_file: 报告输出文件路径（可选）
            
        返回:
            report: 评估报告文本
        """
        summary = self.get_evaluation_summary()
        
        report_lines = [
            "=" * 50,
            "模型评估报告",
            "=" * 50,
            f"数据集: {summary.get('dataset_name', '未知')}",
            f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "性能指标:",
            f"  AUC (ROC曲线下面积): {summary.get('auc', 0):.4f}",
            f"  EER (等错误率): {summary.get('eer', 0):.4f}",
            f"  AP (平均精确率): {summary.get('average_precision', 0):.4f}",
            f"  EER阈值下准确率: {summary.get('eer_accuracy', 0):.4f}",
            f"  EER阈值下F1分数: {summary.get('eer_f1', 0):.4f}",
            f"  得分分离度: {summary.get('score_separation', 0):.4f}",
            "",
            "模型信息:",
            f"  模型类型: {str(type(self.model))}",
            f"  预处理器: {str(type(self.preprocessor))}",
            "=" * 50
        ]
        
        report = "\n".join(report_lines)
        
        # 打印报告
        print(report)
        
        # 保存报告到文件（如果指定了路径）
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"评估报告已保存到: {output_file}")
        
        return report
    
    

# 使用示例
def example_usage():
    """
    优化后类的使用示例
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, scaler)
    
    # 综合评估
    results = evaluator.comprehensive_evaluation(X_test_scaled, y_test, 
                                               dataset_name="示例数据集",
                                               save_plots=True)
    
    # 生成报告
    report = evaluator.generate_report("evaluation_report.txt")
    
    # 保存评估结果
    evaluator.save_evaluation_results("evaluation_results.joblib")
    
    return evaluator, results

    

if __name__ == "__main__":
    evaluator, results = example_usage()