import numpy as np
import librosa
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MFCCFeatureExtractor:
    """
    MFCC特征提取器类
    用于从音频数据中提取MFCC特征
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 n_mels: int = 40,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None,
                 delta: bool = True,
                 delta_delta: bool = True):
        """
        初始化MFCC特征提取器
        
        参数:
            sample_rate: 采样率 (Hz)
            n_mfcc: MFCC系数的数量
            n_fft: FFT窗口大小
            hop_length: 帧移大小
            n_mels: 梅尔滤波器的数量
            fmin: 最低频率 (Hz)
            fmax: 最高频率 (Hz)，None表示使用奈奎斯特频率
            delta: 是否计算一阶差分(Delta)
            delta_delta: 是否计算二阶差分(Delta-Delta)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta = delta
        self.delta_delta = delta_delta
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        从音频数据中提取MFCC特征
        
        参数:
            audio_data: 音频数据数组
            sample_rate: 音频采样率
            
        返回:
            features: 包含MFCC特征的字典
        """
        # 检查采样率是否匹配，如果不匹配则重采样
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            sample_rate = self.sample_rate
        
        # 提取基础MFCC特征
        """
        对应步骤：MFCC核心提取过程
        这一行代码实际上封装了完整的MFCC提取流程：
        预加重：增强高频分量（librosa内部处理）
        分帧：将音频分成重叠的短时帧
            帧长：n_fft（512个采样点）
            帧移：hop_length（256个采样点）
        加窗：对每帧应用汉明窗减少频谱泄漏
        FFT：快速傅里叶变换，时域→频域
        梅尔滤波器组：应用n_mels个三角滤波器
        频率范围：fmin到fmax
        对数运算：计算每个滤波器的对数能量
        DCT：离散余弦变换，得到MFCC系数
        输出n_mfcc个系数
        """
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        

        """
        对应步骤：动态特征提取
        Delta（一阶差分）：捕捉MFCC系数随时间的变化率
        Delta-Delta（二阶差分）：捕捉变化率的变化率（加速度）
        这些动态特征对语音识别和欺骗检测很重要
        """
        features = {'mfcc': mfcc}
        
        # 计算一阶差分 (Delta)
        if self.delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            features['delta'] = mfcc_delta
        
        # 计算二阶差分 (Delta-Delta)
        if self.delta_delta:
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features['delta_delta'] = mfcc_delta2
        

        """
        对应步骤：特征组合
        将静态MFCC、一阶差分、二阶差分拼接成完整特征矩阵
        如果n_mfcc=13，则完整特征维度为：
        只有MFCC：13 × 时间帧数
        MFCC + Delta：26 × 时间帧数
        MFCC + Delta + Delta-Delta：39 × 时间帧数
        """
        # 如果同时有Delta和Delta-Delta，可以拼接成完整的特征矩阵
        if self.delta and self.delta_delta:
            features['full_features'] = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        elif self.delta:
            features['full_features'] = np.vstack([mfcc, mfcc_delta])
        else:
            features['full_features'] = mfcc
        
        return features
    

    """
    对应步骤：批量特征提取
    遍历数据集中的每个音频样本
    对每个样本应用完整的MFCC提取流程
    保留原始元数据并添加特征信息
    """
    def extract_features_from_dataset(self, 
                                    dataset: List[Dict], 
                                    verbose: bool = True) -> List[Dict]:
        """
        从整个数据集中提取MFCC特征
        
        参数:
            dataset: 音频数据集
            verbose: 是否显示进度信息
            
        返回:
            processed_dataset: 包含特征的处理后数据集
        """
        processed_dataset = []
        
        for i, sample in enumerate(dataset):
            if verbose and i % 100 == 0:
                print(f"处理进度: {i}/{len(dataset)}")
            
            try:
                # 提取MFCC特征
                features = self.extract_features(sample['audio'], sample['sample_rate'])
                
                # 创建新的样本字典，包含原始信息和特征
                processed_sample = sample.copy()
                processed_sample['features'] = features
                processed_sample['feature_shape'] = features['full_features'].shape
                
                processed_dataset.append(processed_sample)
                
            except Exception as e:
                print(f"处理样本 {sample['file_name']} 时出错: {str(e)}")
                continue
        
        if verbose:
            print(f"特征提取完成! 成功处理 {len(processed_dataset)}/{len(dataset)} 个样本")
            
        return processed_dataset
    

    """
    对应步骤：特征分析
    时间维度统计：分析不同音频的特征序列长度
    特征值统计：分析MFCC系数的分布特性
    这些统计信息有助于了解数据特性，为后续模型设计提供参考
    """
    def get_feature_statistics(self, processed_dataset: List[Dict]) -> Dict:
        """
        获取特征统计信息
        
        参数:
            processed_dataset: 处理后的数据集
            
        返回:
            stats: 特征统计信息字典
        """
        if not processed_dataset:
            return {}
        
        # 收集所有特征
        all_features = [sample['features']['full_features'] for sample in processed_dataset]
        
        # 计算统计信息
        feature_shapes = [features.shape for features in all_features]
        feature_lengths = [shape[1] for shape in feature_shapes]  # 时间维度长度
        
        # 计算每个特征的均值和标准差（按特征维度）
        feature_means = []
        feature_stds = []
        
        for features in all_features:
            feature_means.append(np.mean(features, axis=1))
            feature_stds.append(np.std(features, axis=1))
        
        stats = {
            'total_samples': len(processed_dataset),
            'feature_dimension': feature_shapes[0][0] if feature_shapes else 0,
            'time_dimension_stats': {
                'min': min(feature_lengths) if feature_lengths else 0,
                'max': max(feature_lengths) if feature_lengths else 0,
                'mean': np.mean(feature_lengths) if feature_lengths else 0,
                'std': np.std(feature_lengths) if feature_lengths else 0
            },
            'feature_value_stats': {
                'mean_of_means': np.mean(feature_means) if feature_means else 0,
                'std_of_means': np.std(feature_means) if feature_means else 0,
                'mean_of_stds': np.mean(feature_stds) if feature_stds else 0,
                'std_of_stds': np.std(feature_stds) if feature_stds else 0
            }
        }
        
        return stats

    def save_features(self, processed_dataset: List[Dict], file_path: str):
        """
        保存特征到文件
        
        参数:
            processed_dataset: 处理后的数据集
            file_path: 保存路径
        """
        # 创建一个简化的版本用于保存（不保存原始音频数据以节省空间）
        save_data = []
        for sample in processed_dataset:
            save_sample = {
                'file_name': sample['file_name'],
                'label': sample['label'],
                'speaker_id': sample['speaker_id'],
                'spoof_type': sample['spoof_type'],
                'dataset_type': sample['dataset_type'],
                'features': sample['features'],
                'feature_shape': sample['feature_shape']
            }
            save_data.append(save_sample)
        
        np.save(file_path, save_data)
        print(f"特征已保存到: {file_path}")
    
    def load_features(self, file_path: str) -> List[Dict]:
        """
        从文件加载特征
        
        参数:
            file_path: 文件路径
            
        返回:
            processed_dataset: 处理后的数据集
        """
        processed_dataset = np.load(file_path, allow_pickle=True)
        print(f"特征已从 {file_path} 加载")
        return processed_dataset.tolist()