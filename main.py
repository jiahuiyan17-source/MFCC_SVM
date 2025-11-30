import sys
import os
import numpy as np

# 添加包含其他类的文件路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_extraction import ASVspoof2019LADataset
from MFCC import MFCCFeatureExtractor
from data_preprocessor import MFCCDataPreprocessor
from svm_trainer import SVMTrainer
from svm_tuner import SVMTuner
from model_evaluator import ModelEvaluator

def main():
    """
    主函数：演示完整的MFCC特征提取和数据预处理流程
    """
    # 请根据实际情况修改路径
    root_path = " "  # 数据集根目录
    
    # 创建数据加载器
    data_loader = ASVspoof2019LADataset(root_path)
    
    # 设置最大样本数（测试用）
    max_samples = {'train': 15000, 'dev': 10000, 'eval': 10000}
    
    # 加载数据集
    print("加载音频数据集...")
    datasets = data_loader.load_all_datasets(max_samples_per_set=max_samples)
    # 这里的代码可以加载所有的数据集：
    #datasets = data_loader.load_all_datasets(max_samples_per_set={})
    
    # 打印原始数据集统计信息
    data_loader.print_dataset_stats()
    
    # 创建MFCC特征提取器
    mfcc_extractor = MFCCFeatureExtractor(
        sample_rate=16000,
        n_mfcc=13,
        n_fft=512,
        hop_length=256,
        delta=True,
        delta_delta=True
    )
    
    # 提取MFCC特征
    print("\n开始提取MFCC特征...")
    feature_datasets = {}
    for dataset_type, dataset in datasets.items():
        print(f"提取 {dataset_type} 数据集的MFCC特征...")
        feature_datasets[dataset_type] = mfcc_extractor.extract_features_from_dataset(dataset)
        
        # 显示特征统计
        stats = mfcc_extractor.get_feature_statistics(feature_datasets[dataset_type])
        print(f"  {dataset_type} MFCC特征统计:")
        print(f"    特征维度: {stats['feature_dimension']}")
        print(f"    时间维度统计: {stats['time_dimension_stats']}")
    
    # 创建MFCC数据预处理器
    preprocessor = MFCCDataPreprocessor(
        normalize_features=True,
        feature_scaling_method='standard',
        test_size=0.2,
        random_state=42,
        fixed_length=None  # 使用统计特征而不是固定长度
    )
    
    # 预处理MFCC特征
    processed_datasets = preprocessor.preprocess_features(feature_datasets)
    
    # 关键修改：确保先准备训练数据（这会拟合标准化器）
    print("\n准备训练数据（仅在训练集上拟合标准化器）...")
    X_train, X_val, y_train, y_val = preprocessor.prepare_training_data(
        use_datasets=['train'],  # 关键：只用训练集，避免数据泄露
        balance_classes=False
    )
    
    # 关键修改：检查标准化器是否已拟合
    print(f"标准化器是否已拟合: {preprocessor.scaler_fitted}")
    print(f"标签编码器是否已拟合: {preprocessor.label_encoder_fitted}")
    
    # 准备开发集数据（使用训练集的标准化参数）
    print("\n准备开发集数据（使用训练集的标准化参数）...")
    X_dev, y_dev = preprocessor.prepare_evaluation_data('dev')
    
    # 准备测试数据
    X_test, y_test = preprocessor.prepare_evaluation_data('eval')
    
    # 获取预处理摘要
    summary = preprocessor.get_preprocessing_summary()
    print("\n预处理摘要:")
    print(f"特征标准化: {summary['normalize_features']}")
    print(f"标准化器已拟合: {summary['scaler_fitted']}")
    print(f"标签编码: {summary['label_encoding']}")
    
    # 创建SVM训练器 - 使用平衡准确率作为评估指标
    svm_trainer = SVMTrainer(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        use_grid_search=False  # 不使用网格搜索，让调优器来处理
    )
    
    # 训练SVM模型
    print("\n开始SVM模型训练...")
    trained_model = svm_trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,  # 使用从训练集分割的验证集
        y_val=y_val,
        cv_folds=5
    )
    
    # 在开发集上评估模型（使用训练集学到的参数）
    print("\n在开发集上评估模型性能...")
    dev_metrics = svm_trainer.detailed_evaluation(
        X=X_dev,
        y=y_dev,
        dataset_name="开发集",
        class_names=list(preprocessor.label_encoder.classes_)
    )
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型性能...")
    test_metrics = svm_trainer.detailed_evaluation(
        X=X_test,
        y=y_test,
        dataset_name="测试集",
        class_names=list(preprocessor.label_encoder.classes_)
    )
    
    # 获取模型信息
    model_info = svm_trainer.get_model_info()
    print("\n模型训练完成!")
    print(f"开发集准确率: {dev_metrics['accuracy']:.4f}")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    
    # 保存训练好的模型
    svm_trainer.save_model("trained_svm_model.joblib")
    
    # 保存预处理器（用于后续预测）
    preprocessor.save_preprocessor("mfcc_preprocessor.joblib")

    # ============ 添加：调优前模型评估（EER和ROC） ============
    print("\n" + "="*50)
    print("开始调优前模型的综合评估（EER和ROC曲线）...")
    print("="*50)
    
    # 创建模型评估器
    model_evaluator_before = ModelEvaluator(trained_model, preprocessor)
    
    # 执行综合评估（包括EER计算和ROC曲线绘制）
    evaluation_results_before = model_evaluator_before.comprehensive_evaluation(
        X_test=X_test,
        y_test=y_test,
        positive_class=1,  # 假设正类是1（bonafide）
        dataset_name="调优前模型 - 测试集"
    )
    
    # 保存评估结果
    model_evaluator_before.save_evaluation_results("evaluation_results_before_tuning.joblib")
    
    # 记录调优前的准确率用于后续比较
    accuracy_before = test_metrics['accuracy']
    eer_before = evaluation_results_before['roc_metrics']['eer']
    auc_before = evaluation_results_before['roc_metrics']['auc']
    
    # ============ 优化：使用改进的调优器 ============
    print("\n" + "="*50)
    print("开始SVM参数调优（改进版本）...")
    print("="*50)
    
    # 创建SVM调优器 - 使用改进版本，防止过拟合
    svm_tuner = SVMTuner(random_state=42, tuning_mode="balanced",
                        handle_imbalance=True, scoring="balanced_accuracy",
                        use_stratified_cv=True)
    
    # 执行改进的参数调优
    print("使用改进的调优策略...")
    tuning_results = svm_tuner.comprehensive_tuning(
        X_train=X_train,
        y_train=y_train,
        cv_folds=5,
        n_iter=30  # 增加随机搜索迭代次数
    )
    
    # 显示调优耗时信息
    timing_info = tuning_results.get('timing', {})
    if timing_info:
        print(f"\n调优时间统计:")
        print(f"  智能调优: {timing_info.get('smart', 0):.2f}秒")
        print(f"  随机搜索: {timing_info.get('random', 0):.2f}秒")
        print(f"  集成调优: {timing_info.get('ensemble', 0):.2f}秒")
        print(f"  总耗时: {timing_info.get('total', 0):.2f}秒")
    
    # 在开发集上评估调优后的模型
    print("\n评估调优后模型在开发集上的性能...")
    dev_metrics_tuned = svm_tuner.evaluate_on_test(
        X_test=X_dev,  # 使用开发集进行调优评估
        y_test=y_dev,
        class_names=list(preprocessor.label_encoder.classes_),
        dataset_name="开发集"
    )
    
    # 在测试集上评估调优后的模型
    print("\n评估调优后模型在测试集上的性能...")
    test_metrics_tuned = svm_tuner.evaluate_on_test(
        X_test=X_test,
        y_test=y_test,
        class_names=list(preprocessor.label_encoder.classes_),
        dataset_name="测试集"
    )
    
    # 分析参数重要性（可选，如果数据量很大可以跳过）
    if X_train.shape[0] < 10000:  # 只在数据量不是特别大时分析
        try:
            svm_tuner.plot_parameter_importance(tuning_results['all_results'])
        except Exception as e:
            print(f"参数重要性分析跳过: {e}")
    
    # 保存调优后的模型
    svm_tuner.save_tuned_model("tuned_svm_model.joblib")
    
    # ============ 添加：调优后模型评估（EER和ROC） ============
    print("\n" + "="*50)
    print("开始调优后模型的综合评估（EER和ROC曲线）...")
    print("="*50)
    
    # 创建模型评估器（使用调优后的模型）
    model_evaluator_after = ModelEvaluator(svm_tuner.best_model, preprocessor)
    
    # 执行综合评估（包括EER计算和ROC曲线绘制）
    evaluation_results_after = model_evaluator_after.comprehensive_evaluation(
        X_test=X_test,
        y_test=y_test,
        positive_class=1,  # 假设正类是1（bonafide）
        dataset_name="调优后模型 - 测试集"
    )
    
    # 保存评估结果
    model_evaluator_after.save_evaluation_results("evaluation_results_after_tuning.joblib")
    
    # ============ 添加：模型比较 ============
    print("\n" + "="*50)
    print("模型性能对比分析")
    print("="*50)
    
    # 比较调优前后的模型
    models_comparison = {
        '调优前模型': trained_model,
        '调优后模型': svm_tuner.best_model
    }
    
    # 使用改进的比较方法
    comparison_results = svm_tuner.compare_with_baseline(
        trained_model, X_test, y_test, "调优前模型"
    )
    
    model_evaluator_after.compare_models(
        models_dict=models_comparison,
        X_test=X_test,
        y_test=y_test,
        positive_class=1
    )
    
    # 返回所有结果
    return {
        'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val,
        'X_dev': X_dev, 'y_dev': y_dev,  # 添加开发集
        'X_test': X_test, 'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_extractor': mfcc_extractor,
        'svm_trainer': svm_trainer,
        'svm_tuner': svm_tuner,
        'tuning_results': tuning_results,
        'test_metrics': test_metrics_tuned,  # 使用调优后的测试指标
        'dev_metrics': dev_metrics_tuned,    # 添加开发集指标
        'model_info': model_info,
        'comparison_results': comparison_results,  # 添加比较结果
        # 添加评估结果
        'evaluation_results_before': evaluation_results_before,
        'evaluation_results_after': evaluation_results_after,
        'model_evaluator_before': model_evaluator_before,
        'model_evaluator_after': model_evaluator_after
    }

if __name__ == "__main__":
    results = main()
    
    if results is not None:
        print("\n=== 完整流程完成 ===")
        print(f"训练数据形状: {results['X_train'].shape}")
        print(f"开发集数据形状: {results['X_dev'].shape}")
        print(f"测试数据形状: {results['X_test'].shape}")
        
        # 显示调优前后性能对比
        print("\n" + "="*50)
        print("模型性能对比总结")
        print("="*50)
        
        # 调优前指标
        accuracy_before = results['evaluation_results_before']['eer_accuracy']
        eer_before = results['evaluation_results_before']['roc_metrics']['eer']
        auc_before = results['evaluation_results_before']['roc_metrics']['auc']
        
        # 调优后指标
        accuracy_after = results['test_metrics']['accuracy']
        balanced_accuracy_after = results['test_metrics']['balanced_accuracy']
        eer_after = results['evaluation_results_after']['roc_metrics']['eer']
        auc_after = results['evaluation_results_after']['roc_metrics']['auc']
        
        # 比较结果
        comparison = results['comparison_results']
        
        print(f"{'指标':<20} {'调优前':<10} {'调优后':<10} {'提升':<10}")
        print(f"{'-' * 55}")
        print(f"{'准确率':<20} {accuracy_before:.4f}    {accuracy_after:.4f}    {comparison['accuracy_improvement']:+.4f}")
        print(f"{'平衡准确率':<20} {'N/A':<10} {balanced_accuracy_after:.4f}    {comparison['balanced_accuracy_improvement']:+.4f}")
        print(f"{'EER':<20} {eer_before:.4f}    {eer_after:.4f}    {eer_after - eer_before:+.4f}")
        print(f"{'AUC':<20} {auc_before:.4f}    {auc_after:.4f}    {comparison['tuned_auc'] - comparison['baseline_auc'] if comparison['tuned_auc'] else 0:+.4f}")
        print(f"{'F1分数':<20} {'N/A':<10} {results['test_metrics']['f1_score']:.4f}    {comparison['f1_improvement']:+.4f}")
        
        # 显示开发集性能
        print(f"\n开发集性能:")
        print(f"  调优后开发集准确率: {results['dev_metrics']['accuracy']:.4f}")
        print(f"  调优后开发集平衡准确率: {results['dev_metrics']['balanced_accuracy']:.4f}")
        
        # 显示模型信息
        model_info = results['model_info']
        print(f"\n模型信息:")
        print(f"  模型类型: {model_info['model_type']}")
        print(f"  核函数: {model_info['kernel']}")
        if 'best_params' in model_info:
            print(f"  最佳参数: {model_info['best_params']}")
        print(f"  训练集准确率: {model_info['training_history'].get('train_accuracy', 'N/A')}")
        print(f"  验证集准确率: {model_info['training_history'].get('val_accuracy', 'N/A')}")

        print("\n=== 参数调优完成 ===")
        print(f"调优后测试准确率: {results['test_metrics']['accuracy']:.4f}")
        print(f"调优后测试平衡准确率: {results['test_metrics']['balanced_accuracy']:.4f}")
        
        # 显示最佳参数
        best_params = results['svm_tuner'].best_params
        print(f"\n最佳参数:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # 显示调优模式信息
        tuning_mode = getattr(results['svm_tuner'], 'tuning_mode', 'balanced')
        print(f"调优模式: {tuning_mode}")
        
        print("\n所有模型、预处理器和评估结果已保存，可以用于后续预测和分析。")
        print("保存的文件包括:")
        print("  - trained_svm_model.joblib (调优前模型)")
        print("  - tuned_svm_model.joblib (调优后模型)")
        print("  - mfcc_preprocessor.joblib (预处理器)")
        print("  - evaluation_results_before_tuning.joblib (调优前评估结果)")
        print("  - evaluation_results_after_tuning.joblib (调优后评估结果)")
    else:

        print("流程执行失败，请检查错误信息。")
