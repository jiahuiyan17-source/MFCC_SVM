# 基于 MFCC 和 SVM 的语音伪造检测

所用数据集 ：ASVSpoof2019

使用梅尔频率倒谱系数 (MFCC) 作为特征，支持向量机 (SVM) 作为分类器，构建一个简单但有效的语音伪造检测系统。

本质是一个分类问题，将真实与虚假语音分离出来。

## ASVspoof2019数据集结构

```
  root_path/

​           ├── ASVspoof2019_LA_train/ # 训练集音频文件

​           │  └── flac/

​           ├── ASVspoof2019_LA_dev/ # 开发集音频文件

​           │  └── flac/

​           ├── ASVspoof2019_LA_eval/ # 测试集音频文件

​           │  └── flac/

​           └── ASVspoof2019_LA_cm_protocols/

​             ├── ASVspoof2019.LA.cm.train.trn.txt # 训练集协议

​             ├── ASVspoof2019.LA.cm.dev.trl.txt # 开发集协议

​             └── ASVspoof2019.LA.cm.eval.trl.txt # 测试集协议
```



## 1.MFCC特征提取

梅尔频率倒谱系数（MFCC）在进行特征提取时要分别进行提取，不能够将三个数据集一起提取特征，一起提取可能会导致数据泄露，使得开发集与评估集准确率偏高。

## 2.数据预处理

数据预处理支持不平衡问题的处理

```
 	# 平衡类别（如果需要）

​    if balance_classes and len(X) > 0:

​      X, y_encoded = self._balance_classes(X, y_encoded)
```

## 3.SVM模型训练

SVM训练使用网格搜索进行超参数优化，这里使用了三个模型分别是linear、rbf和roly，并结合交叉验证进行训练，交叉验证折数这里选用的是5

```
    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5):
​        """
​        使用网格搜索优化超参数
​        """
​        print("开始网格搜索优化超参数...")
​        

        # 定义参数网格

​        param_grid = {
​            'C': [0.1, 1, 10, 100],
​            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
​            'kernel': ['rbf', 'linear']
​        }
​        

        # 创建网格搜索对象

​        grid_search = GridSearchCV(
​            SVC(random_state=self.random_state, probability=True),
​            param_grid,
​            cv=cv_folds,
​            scoring='accuracy',
​            n_jobs=-1,
​            verbose=1
​        )
​        

        # 执行网格搜索

​        grid_search.fit(X_train, y_train)
​        

        # 保存最佳模型和参数

​        self.model = grid_search.best_estimator_
​        self.best_params_ = grid_search.best_params_
​        
​        print(f"网格搜索完成! 最佳参数: {self.best_params_}")
​        print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
​    
​    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
​        """
​        执行交叉验证
​        """
​        print(f"执行{cv_folds}折交叉验证...")
​        
​        cv_scores = cross_val_score(
​            self.model, X, y, 
​            cv=cv_folds, 
​            scoring='accuracy',
​            n_jobs=-1
​        )
​        
​        self.cv_scores = cv_scores
​        print(f"交叉验证准确率: {cv_scores}")
​        print(f"平均交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

调优前结果如下：关键性能指标:

 AUC (ROC曲线下面积): 0.9033

 EER (等错误率): 0.1736

 EER阈值: 0.0001

 AP (平均精确率): 0.9878

 得分分离度: 2.3735

 

在EER阈值下的性能:

 准确率: 0.8262

 F1分数: 0.8947



## 4.SVM模型调优

优化采用智能分层调优、随机搜索和集成调优，其中集成调优就是将前两中调优方式结合，并使用分层交叉验证进行最终评估。

```
  def _improved_smart_tuning(self, X_train: np.ndarray, y_train: np.ndarray,

​               cv_folds: int) -> Dict[str, Any]:

​    """

​    改进的智能分层调优 - 防止过拟合

​    """

​    print("执行改进的智能分层调优...")

​    

​    \# 第一步：使用更稳健的核函数筛选

​    print("步骤1: 稳健核函数筛选")

​    best_kernel = self._robust_kernel_selection(X_train, y_train, cv_folds=3)

​    print(f"最佳核函数: {best_kernel}")

​    

​    \# 第二步：针对最佳核函数进行稳健调优

​    print(f"步骤2: {best_kernel}核函数稳健调优")

​    robust_results = self._robust_tune_single_kernel(best_kernel, X_train, y_train, cv_folds)

​    

​    return {

​      'best_kernel': best_kernel,

​      'best_estimator': robust_results['best_estimator'],

​      'best_params': robust_results['best_params'],

​      'best_score': robust_results['best_score']

​    }
```

   

```
 def _improved_random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                               cv_folds: int, n_iter: int) -> Dict[str, Any]:
        """
        改进的随机搜索 - 防止过拟合
        """
        print("执行改进的随机搜索...")
        

        # 优化参数分布

​        param_dist = {
​            'kernel': ['linear', 'rbf', 'poly'],
​            'C': np.logspace(-3, 3, 20),  # 扩大搜索范围
​            'gamma': np.logspace(-4, 2, 20),  # 扩大搜索范围
​            'degree': [2, 3],
​            'coef0': [0.0, 0.5, 1.0],
​        }
​        

        # 添加类别权重选项

​        if self.handle_imbalance:
​            class_weight_options = [None, 'balanced']
​            if self.class_weights is not None:
​                class_weight_options.append(self.class_weights)
​            param_dist['class_weight'] = class_weight_options
​        

        # 设置评分标准

​        if self.scoring == "balanced_accuracy":
​            scoring_func = 'balanced_accuracy'
​        elif self.scoring == "f1":
​            scoring_func = 'f1_macro'
​        elif self.scoring == "roc_auc":
​            scoring_func = 'roc_auc_ovo'
​        else:
​            scoring_func = 'accuracy'
​        

        # 使用分层交叉验证

​        if self.use_stratified_cv:
​            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
​        else:
​            cv = cv_folds

​        random_search = RandomizedSearchCV(
​            SVC(random_state=self.random_state, probability=True),
​            param_dist,
​            n_iter=n_iter,
​            cv=cv,
​            scoring=scoring_func,
​            n_jobs=-1,
​            random_state=self.random_state,
​            verbose=1
​        )
​        
​        start_time = time.time()
​        random_search.fit(X_train, y_train)
​        end_time = time.time()
​        
​        print(f"随机搜索完成，耗时: {end_time - start_time:.2f}秒")
​        print(f"最佳参数: {random_search.best_params_}")
​        print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")
​        
​        return {
​            'best_estimator': random_search.best_estimator_,
​            'best_params': random_search.best_params_,
​            'best_score': random_search.best_score_,
​            'cv_results': random_search.cv_results_
​        }
​    

```

  

```
  def _ensemble_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv_folds: int, smart_results: Dict, random_results: Dict) -> Dict[str, Any]:
        """
        集成调优 - 结合多种调优方法的结果
        """
        print("执行集成调优...")
        

        # 收集所有候选模型

​        candidates = [
​            ('smart', smart_results['best_estimator'], smart_results['best_score']),
​            ('random', random_results['best_estimator'], random_results['best_score'])
​        ]
​        

        # 评估每个候选模型的稳健性

​        best_score = -1
​        best_estimator = None
​        best_method = None
​        
​        for method, estimator, score in candidates:

            # 使用分层交叉验证进行稳健性评估

​            if self.use_stratified_cv:
​                cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=self.random_state)
​            else:
​                cv = min(cv_folds, 5)
​            

            # 使用多种评分标准

​            scores = []
​            

            # 主要评分标准

​            if self.scoring == "balanced_accuracy":
​                scoring_func = 'balanced_accuracy'
​            elif self.scoring == "f1":
​                scoring_func = 'f1_macro'
​            elif self.scoring == "roc_auc":
​                scoring_func = 'roc_auc_ovo'
​            else:
​                scoring_func = 'accuracy'
​            
​            main_scores = cross_val_score(estimator, X_train, y_train, 
​                                        cv=cv, scoring=scoring_func)
​            

            # 准确率评分

​            accuracy_scores = cross_val_score(estimator, X_train, y_train, 
​                                            cv=cv, scoring='accuracy')
​            

            # 综合评分（考虑均值和方差）

​            main_mean = main_scores.mean()
​            main_std = main_scores.std()
​            accuracy_mean = accuracy_scores.mean()
​            

            # 稳健性评分（均值高，方差低）

​            robustness_score = main_mean / (1 + main_std) + 0.2 * accuracy_mean
​            
​            print(f"  {method}方法 - 主要分数: {main_mean:.4f} (±{main_std:.4f}), 稳健性评分: {robustness_score:.4f}")
​            
​            if robustness_score > best_score:
​                best_score = robustness_score
​                best_estimator = estimator
​                best_method = method
​        

        # 获取最佳参数

​        if best_method == 'smart':
​            best_params = smart_results['best_params']
​        else:
​            best_params = random_results['best_params']
​        
​        print(f"集成调优选择: {best_method}方法")
​        
​        return {
​            'best_estimator': best_estimator,
​            'best_params': best_params,
​            'best_score': best_score,
​            'selected_method': best_method
​        }
```

```
执行改进的智能分层调优...

步骤1: 稳健核函数筛选

 linear核 - 主要分数: 0.9984, 准确率: 0.9988, 综合分数: 0.9985

 rbf核 - 主要分数: 0.9998, 准确率: 0.9999, 综合分数: 0.9998

 poly核 - 主要分数: 0.9960, 准确率: 0.9966, 综合分数: 0.9962

最佳核函数: rbf

步骤2: rbf核函数稳健调优

 参数网格大小: 507

Fitting 5 folds for each of 507 candidates, totalling 2535 fits

执行改进的随机搜索...

Fitting 5 folds for each of 15 candidates, totalling 75 fits

随机搜索完成，耗时: 214.42秒

最佳参数: {'kernel': 'poly', 'gamma': 0.01623776739188721, 'degree': 3, 'coef0': 0.5, 'class_weight': 'balanced', 'C': 1.438449888287663}

最佳交叉验证分数: 0.9999

执行集成调优...

 smart方法 - 主要分数: 0.9999 (±0.0002), 稳健性评分: 1.1997

 random方法 - 主要分数: 0.9999 (±0.0002), 稳健性评分: 1.1997

集成调优选择: smart方法

 

最终选择方法: ensemble_tuning

最佳交叉验证分数: 1.1997
```

最终结果如下

 AUC (ROC曲线下面积): 0.9033

 EER (等错误率): 0.1692

 EER阈值: 0.0001

 AP (平均精确率): 0.9878

 得分分离度: 2.2940

 

在EER阈值下的性能:

 准确率: 0.8312

 F1分数: 0.8979

评估结果已保存到: evaluation_results_after_tuning.joblib

## 弊端

但由于SVM并不适合大量的数据集，而且作为机器学习的方法有一定局限性，所以导致准确率和EER并没有特别优秀。
