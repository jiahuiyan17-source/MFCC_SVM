import os
import pandas as pd
import soundfile as sf
import numpy as np
from typing import Dict, List, Optional

#这是一个ASVspoof2019LADataset类，专门用于加载数据集，后续进行数据处理可直接引用
class ASVspoof2019LADataset:
    #初始化数据集加载器，设置文件路径和基本参数。
    def __init__(self, root_path: str):
        """
        初始化ASVspoof2019 LA数据集加载器
        
        参数:
            root_path: 数据集根目录路径
                      预期目录结构:
                      root_path/
                      ├── ASVspoof2019_LA_train/ # 训练集音频文件
                      │   └── flac/
                      ├── ASVspoof2019_LA_dev/ # 开发集音频文件
                      │   └── flac/
                      ├── ASVspoof2019_LA_eval/ # 测试集音频文件
                      │   └── flac/
                      └── ASVspoof2019_LA_cm_protocols/
                          ├── ASVspoof2019.LA.cm.train.trn.txt # 训练集协议
                          ├── ASVspoof2019.LA.cm.dev.trl.txt # 开发集协议
                          └── ASVspoof2019.LA.cm.eval.trl.txt # 测试集协议
        """
        self.root_path = root_path
        self.sets = ['train', 'dev', 'eval']
        
        # 定义各数据集的路径
        #构建音频文件路径字典
        self.audio_paths = {
            'train': os.path.join(root_path, 'ASVspoof2019_LA_train', 'flac'),
            'dev': os.path.join(root_path, 'ASVspoof2019_LA_dev', 'flac'),
            'eval': os.path.join(root_path, 'ASVspoof2019_LA_eval', 'flac')
        }
        
        #构建协议文件路径字典，这些文件包含数据集的元数据信息。
        self.protocol_paths = {
            'train': os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt'),
            'dev': os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'),
            'eval': os.path.join(root_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt')
        }
        
        # 存储加载的数据
        #初始化两个空字典，用于存储加载的数据和元数据。
        self.datasets = {}
        self.metadata = {}
        
    #加载指定数据集的协议文件元数据，返回pandas DataFrame    
    def load_metadata(self, dataset_type: str) -> pd.DataFrame:
        """
        加载指定数据集的协议文件元数据
        
        参数:
            dataset_type: 数据集类型 ('train', 'dev', 'eval')
            
        返回:
            metadata: 包含元数据的DataFrame
        """
        #检查数据集类型是否有效，无效则抛出错误
        if dataset_type not in self.sets:
            raise ValueError(f"数据集类型必须是 {self.sets} 之一")

        #获取协议文件路径，初始化元数据列表    
        protocol_file = self.protocol_paths[dataset_type]
        metadata = []
        
        """
        打开协议文件，逐行读取并解析：
        line.strip().split(): 去除首尾空格并按空格分割成列表
        检查是否有足够的部分(至少4个)
        """
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    """
                    解析各行数据：
                    speaker_id: 说话人ID（第1部分）
                    audio_file: 音频文件名（第2部分）
                    spoof_type: 欺骗类型（第4部分）
                    label: 标签（如果有第5部分则取第5部分，否则取第4部分）
                    """
                    speaker_id = parts[0]
                    audio_file = parts[1]
                    spoof_type = parts[3]
                    label = parts[4] if len(parts) > 4 else parts[3]
                    
                    # 构建完整的音频文件路径
                    #构建完整的音频文件路径，格式为：音频目录/音频文件名.flac
                    audio_path = os.path.join(self.audio_paths[dataset_type], f"{audio_file}.flac")
                    
                    #将解析出的信息组织成字典并添加到元数据列表。
                    metadata.append({
                        'speaker_id': speaker_id,
                        'audio_file': audio_file,
                        'spoof_type': spoof_type,
                        'label': label,
                        'file_path': audio_path,
                        'dataset_type': dataset_type
                    })
        
        #将列表转换为DataFrame，存储到类变量中并返回。
        df = pd.DataFrame(metadata)
        self.metadata[dataset_type] = df
        return df
    
    #加载单个音频文件，返回音频数据和采样率。
    def load_audio(self, file_path: str) -> tuple:
        """
        加载单个音频文件
        
        参数:
            file_path: 音频文件路径
            
        返回:
            audio_data: 音频数据数组
            sample_rate: 采样率
        """

        #使用soundfile.read()读取音频文件，包含错误处理。
        try:
            audio_data, sample_rate = sf.read(file_path)
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None, None
    
    #加载指定数据集的核心方法。
    def load_dataset(self, dataset_type: str, max_samples: Optional[int] = None, 
                    verbose: bool = True) -> List[Dict]:
        """
        加载指定数据集
        
        参数:
            dataset_type: 数据集类型 ('train', 'dev', 'eval')
            max_samples: 最大样本数 (None表示加载所有数据)
            verbose: 是否显示详细信息
            
        返回:
            dataset: 包含音频数据和元数据的字典列表
        """
        #验证数据集类型 
        if dataset_type not in self.sets:
            raise ValueError(f"数据集类型必须是 {self.sets} 之一")
            
        # 加载元数据
        metadata = self.load_metadata(dataset_type)
        
        #初始化数据集列表和计数器
        dataset = []
        count = 0
        
        #遍历元数据的每一行，如果设置了最大样本数且已达到，则停止加载
        for idx, row in metadata.iterrows():
            if max_samples and count >= max_samples:
                break

            #加载音频文件，如果成功则构建样本信息字典并添加到数据集    
            audio_data, sample_rate = self.load_audio(row['file_path'])
            
            if audio_data is not None:
                sample_info = {
                    'audio': audio_data,
                    'sample_rate': sample_rate,
                    'label': row['label'],
                    'spoof_type': row['spoof_type'],
                    'speaker_id': row['speaker_id'],
                    'file_name': row['audio_file'],
                    'file_path': row['file_path'],
                    'dataset_type': dataset_type
                }
                dataset.append(sample_info)
                count += 1
        
        #存储数据集并打印加载信息，返回数据集
        self.datasets[dataset_type] = dataset
        
        if verbose:
            print(f"{dataset_type.upper()}数据集加载完成:")
            print(f"  成功加载 {len(dataset)} 个音频样本")
            label_counts = pd.Series([item['label'] for item in dataset]).value_counts()
            print(f"  标签分布: {dict(label_counts)}")
        
        return dataset
    
    #一次性加载所有三个数据集
    def load_all_datasets(self, max_samples_per_set: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict]]:
        """
        加载所有数据集
        
        参数:
            max_samples_per_set: 每个数据集的最大样本数，例如 {'train': 1000, 'dev': 500, 'eval': 500}
            
        返回:
            datasets: 包含所有数据集的字典
        """
        
        #遍历三种数据集类型，分别加载并返回所有数据集
        if max_samples_per_set is None:
            max_samples_per_set = {}
            
        for dataset_type in self.sets:
            max_samples = max_samples_per_set.get(dataset_type, None)
            self.load_dataset(dataset_type, max_samples)
            
        return self.datasets
    
    #返回标签到数字的映射字典，用于机器学习模型
    def get_label_encoding(self) -> Dict[str, int]:
        """
        获取标签编码映射
        """
        return {'bonafide': 1, 'spoof': 0}
    
    #获取指定数据集的详细统计信息
    def get_dataset_info(self, dataset_type: str) -> Dict:
        """
        获取数据集统计信息
        
        参数:
            dataset_type: 数据集类型
            
        返回:
            info: 包含数据集统计信息的字典
        """
        
        #检查数据集是否已加载
        if dataset_type not in self.datasets:
            raise ValueError(f"请先加载 {dataset_type} 数据集")

        #从数据集中提取各种统计信息    
        dataset = self.datasets[dataset_type]
        labels = [item['label'] for item in dataset]
        spoof_types = [item['spoof_type'] for item in dataset]
        sample_rates = [item['sample_rate'] for item in dataset]
        audio_lengths = [len(item['audio']) for item in dataset]
        
        """
        构建包含各种统计信息的字典：
        总样本数
        标签分布
        欺骗类型分布
        唯一说话人数量
        采样率
        音频长度统计（最小值、最大值、均值、标准差）
        """
        info = {
            'total_samples': len(dataset),
            'label_distribution': pd.Series(labels).value_counts().to_dict(),
            'spoof_type_distribution': pd.Series(spoof_types).value_counts().to_dict(),
            'unique_speakers': len(set([item['speaker_id'] for item in dataset])),
            'sample_rate': f"{sample_rates[0]} Hz" if sample_rates else "N/A",
            'audio_length_stats': {
                'min': min(audio_lengths),
                'max': max(audio_lengths),
                'mean': np.mean(audio_lengths),
                'std': np.std(audio_lengths)
            }
        }
        
        return info
    
    #遍历所有已加载的数据集，打印它们的统计信息
    def print_dataset_stats(self):
        """打印所有已加载数据集的统计信息"""
        for dataset_type in self.datasets.keys():
            print(f"\n{dataset_type.upper()}数据集统计:")
            info = self.get_dataset_info(dataset_type)
            for key, value in info.items():
                print(f"  {key}: {value}")

