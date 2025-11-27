# MFCC_SVM
# 基于 MFCC 和 SVM 的语音伪造检测

所用数据集 ：ASVSpoof2019

使用梅尔频率倒谱系数 (MFCC) 作为特征，支持向量机 (SVM) 作为分类器，构建一个简单但有效的语音伪造检测系统。



## 加载数据

将ASVSpoof2019数据集加载进来，便于下一步进行特征提取

ASVSpoof2019数据集结构如下：

​           root_path/

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

