# 基于 MFCC 和 SVM 的语音伪造检测

所用数据集 ：ASVSpoof2019

使用梅尔频率倒谱系数 (MFCC) 作为特征，支持向量机 (SVM) 作为分类器，构建一个简单但有效的语音伪造检测系统。



## 加载数据

将ASVSpoof2019数据集加载进来，便于下一步进行特征提取

ASVSpoof2019数据集结构如下：

​         

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

数据加载时，需要进行元数据加载、音频加载和数据集加载

### 元数据加载

```
def load_metadata(self, dataset_type: str) -> pd.DataFrame:
     
        #检查数据集类型是否有效，无效则抛出错误
        if dataset_type not in self.sets:
            raise ValueError(f"数据集类型必须是 {self.sets} 之一")
            
​        #获取协议文件路径，初始化元数据列表    
​        protocol_file = self.protocol_paths[dataset_type]
​        metadata = []
​        
​        with open(protocol_file, 'r') as f:
​            for line in f:
​                parts = line.strip().split()
​                if len(parts) >= 4:
​              
​                    speaker_id = parts[0]
​                    audio_file = parts[1]
​                    spoof_type = parts[3]
​                    label = parts[4] if len(parts) > 4 else parts[3]
​                 
​                    #构建完整的音频文件路径，格式为：音频目录/音频文件名.flac
​                    audio_path = os.path.join(self.audio_paths[dataset_type], f"{audio_file}.flac")
​                    
​                    #将解析出的信息组织成字典并添加到元数据列表。
​                    metadata.append({
​                        'speaker_id': speaker_id,
​                        'audio_file': audio_file,
​                        'spoof_type': spoof_type,
​                        'label': label,
​                        'file_path': audio_path,
​                        'dataset_type': dataset_type
​                    })
​        
​        #将列表转换为DataFrame，存储到类变量中并返回。
​        df = pd.DataFrame(metadata)
​        self.metadata[dataset_type] = df
​        return df
```

### 音频加载

```
def load_audio(self, file_path: str) -> tuple:

​    #使用soundfile.read()读取音频文件，包含错误处理。

​    try:

​      audio_data, sample_rate = sf.read(file_path)

​      return audio_data, sample_rate

​    except Exception as e:

​      print(f"Error loading audio file {file_path}: {str(e)}")

​      return None, None
```

### 数据集加载

```
def load_dataset(self, dataset_type: str, max_samples: Optional[int] = None, 
                    verbose: bool = True) -> List[Dict]:
        #验证数据集类型 
        if dataset_type not in self.sets:
            raise ValueError(f"数据集类型必须是 {self.sets} 之一")
            
​        metadata = self.load_metadata(dataset_type)
​        
​        #初始化数据集列表和计数器
​        dataset = []
​        count = 0
​        
​        #遍历元数据的每一行，如果设置了最大样本数且已达到，则停止加载
​        for idx, row in metadata.iterrows():
​            if max_samples and count >= max_samples:
​                break
​            #加载音频文件，如果成功则构建样本信息字典并添加到数据集    
​            audio_data, sample_rate = self.load_audio(row['file_path'])
​            
​            if audio_data is not None:
​                sample_info = {
​                    'audio': audio_data,
​                    'sample_rate': sample_rate,
​                    'label': row['label'],
​                    'spoof_type': row['spoof_type'],
​                    'speaker_id': row['speaker_id'],
​                    'file_name': row['audio_file'],
​                    'file_path': row['file_path'],
​                    'dataset_type': dataset_type
​                }
​                dataset.append(sample_info)
​                count += 1
​        
​        #存储数据集并打印加载信息，返回数据集
​        self.datasets[dataset_type] = dataset
​        
​        if verbose:
​            print(f"{dataset_type.upper()}数据集加载完成:")
​            print(f"  成功加载 {len(dataset)} 个音频样本")
​            label_counts = pd.Series([item['label'] for item in dataset]).value_counts()
​            print(f"  标签分布: {dict(label_counts)}")
​        
​        return dataset
```

由于有三个数据集（train、del、eval）所以需要将这三个数据集全部加载进来，进行后续的数据处理

```
 def load_all_datasets(self, max_samples_per_set: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict]]: 

​    \#遍历三种数据集类型，分别加载并返回所有数据集

​    if max_samples_per_set is None:

​      max_samples_per_set = {}

​      

​    for dataset_type in self.sets:

​      max_samples = max_samples_per_set.get(dataset_type, None)

​      self.load_dataset(dataset_type, max_samples)

​      

​    return self.datasets

```

## MFCC特征提取

MFCC的步骤为：

1. 分帧、加窗
2. 对于每一帧，计算功率谱的[周期图估计](http://en.wikipedia.org/wiki/Periodogram)
3. 将mel滤波器组应用于功率谱，求滤波器组的能量，将每个滤波器中的能量相加
4. 取所有滤波器组能量的对数
5. 取对数滤波器组能量的离散余弦变换（DCT）。
6. 保持DCT系数2-13，其余部分丢弃



### 音频处理参数

- `sample_rate=16000`: 标准语音采样率
- `n_fft=512`: FFT窗口大小，对应32ms（512/16000）
- `hop_length=256`: 帧移，对应16ms，提供50%重叠

### MFCC特定参数

- `n_mfcc=13`: 提取13个MFCC系数（通常包含0阶系数）
- `n_mels=40`: 梅尔滤波器数量，影响频率分辨率
- `fmin=0.0, fmax=None`: 频率范围，None表示使用奈奎斯特频率(8000Hz)

### 动态特征参数

- `delta=True`: 一阶差分，捕捉特征变化率
- `delta_delta=True`: 二阶差分，捕捉变化加速度



```
	 self.sample_rate = sample_rate

​    self.n_mfcc = n_mfcc

​    self.n_fft = n_fft

​    self.hop_length = hop_length

​    self.n_mels = n_mels

​    self.fmin = fmin

​    self.fmax = fmax

​    self.delta = delta

​    self.delta_delta = delta_delta
```



### 从音频数据中提取MFCC特征

这里除了普通的特征提取外，还加入了动态提取

```
def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:

        # 检查采样率是否匹配，如果不匹配则重采样

​        if sample_rate != self.sample_rate:
​            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
​            sample_rate = self.sample_rate
​        

        # 提取基础MFCC特征

​        """
​        对应步骤：MFCC核心提取过程
​        这一行代码实际上封装了完整的MFCC提取流程：
​        预加重：增强高频分量（librosa内部处理）
​        分帧：将音频分成重叠的短时帧
​            帧长：n_fft（512个采样点）
​            帧移：hop_length（256个采样点）
​        加窗：对每帧应用汉明窗减少频谱泄漏
​        FFT：快速傅里叶变换，时域→频域
​        梅尔滤波器组：应用n_mels个三角滤波器
​        频率范围：fmin到fmax
​        对数运算：计算每个滤波器的对数能量
​        DCT：离散余弦变换，得到MFCC系数
​        输出n_mfcc个系数
​        """
​        mfcc = librosa.feature.mfcc(
​            y=audio_data,
​            sr=sample_rate,
​            n_mfcc=self.n_mfcc,
​            n_fft=self.n_fft,
​            hop_length=self.hop_length,
​            n_mels=self.n_mels,
​            fmin=self.fmin,
​            fmax=self.fmax
​        )
​        

​        """
​        对应步骤：动态特征提取
​        Delta（一阶差分）：捕捉MFCC系数随时间的变化率
​        Delta-Delta（二阶差分）：捕捉变化率的变化率（加速度）
​        这些动态特征对语音识别和欺骗检测很重要
​        """
​        features = {'mfcc': mfcc}
​        

        # 计算一阶差分 (Delta)

​        if self.delta:
​            mfcc_delta = librosa.feature.delta(mfcc)
​            features['delta'] = mfcc_delta
​        

        # 计算二阶差分 (Delta-Delta)

​        if self.delta_delta:
​            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
​            features['delta_delta'] = mfcc_delta2
​        

​        """
​        对应步骤：特征组合
​        将静态MFCC、一阶差分、二阶差分拼接成完整特征矩阵
​        如果n_mfcc=13，则完整特征维度为：
​        只有MFCC：13 × 时间帧数
​        MFCC + Delta：26 × 时间帧数
​        MFCC + Delta + Delta-Delta：39 × 时间帧数
​        """

        # 如果同时有Delta和Delta-Delta，可以拼接成完整的特征矩阵

​        if self.delta and self.delta_delta:
​            features['full_features'] = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
​        elif self.delta:
​            features['full_features'] = np.vstack([mfcc, mfcc_delta])
​        else:
​            features['full_features'] = mfcc
​        
​        return features
```

### 批量特征提取提取

```
def extract_features_from_dataset(self, 

​                  dataset: List[Dict], 

​                  verbose: bool = True) -> List[Dict]:

   

​    processed_dataset = []

​    

​    for i, sample in enumerate(dataset):

​      if verbose and i % 100 == 0:

​        print(f"处理进度: {i}/{len(dataset)}")

​      

​      try:

​        \# 提取MFCC特征

​        features = self.extract_features(sample['audio'], sample['sample_rate'])

​        

​        \# 创建新的样本字典，包含原始信息和特征

​        processed_sample = sample.copy()

​        processed_sample['features'] = features

​        processed_sample['feature_shape'] = features['full_features'].shape

​        

​        processed_dataset.append(processed_sample)

​        

​      except Exception as e:

​        print(f"处理样本 {sample['file_name']} 时出错: {str(e)}")

​        continue

​    

​    if verbose:

​      print(f"特征提取完成! 成功处理 {len(processed_dataset)}/{len(dataset)} 个样本")

​      

​    return processed_dataset
```

### 进行特征分析

```
def get_feature_statistics(self, processed_dataset: List[Dict]) -> Dict:

​    if not processed_dataset:

​      return {}

​    

​    \# 收集所有特征

​    all_features = [sample['features']['full_features'] for sample in processed_dataset]

​    

​    \# 计算统计信息

​    feature_shapes = [features.shape for features in all_features]

​    feature_lengths = [shape[1] for shape in feature_shapes]  # 时间维度长度

​    

​    \# 计算每个特征的均值和标准差（按特征维度）

​    feature_means = []

​    feature_stds = []

​    

​    for features in all_features:

​      feature_means.append(np.mean(features, axis=1))

​      feature_stds.append(np.std(features, axis=1))

​    

​    stats = {

​      'total_samples': len(processed_dataset),

​      'feature_dimension': feature_shapes[0][0] if feature_shapes else 0,

​      'time_dimension_stats': {

​        'min': min(feature_lengths) if feature_lengths else 0,

​        'max': max(feature_lengths) if feature_lengths else 0,

​        'mean': np.mean(feature_lengths) if feature_lengths else 0,

​        'std': np.std(feature_lengths) if feature_lengths else 0

​      },

​      'feature_value_stats': {

​        'mean_of_means': np.mean(feature_means) if feature_means else 0,

​        'std_of_means': np.std(feature_means) if feature_means else 0,

​        'mean_of_stds': np.mean(feature_stds) if feature_stds else 0,

​        'std_of_stds': np.std(feature_stds) if feature_stds else 0

​      }

​    }

​    

​    return stats
```

