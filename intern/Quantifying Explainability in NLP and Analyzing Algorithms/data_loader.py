#============= data_loader.py =============
import pandas as pd
import os
import chardet
import logging
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_files_exist():
    """检查必需文件是否存在"""
    required_files = ['notebook\\ADMISSIONS.csv', 'notebook\\DIAGNOSES_ICD.csv', 'notebook\\PROCEDURES_ICD.csv']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        logging.error(f"Missing required files: {missing}")
        logging.info("Files in current directory:")
        logging.info('\n'.join(os.listdir('.')))
        raise FileNotFoundError(f"Missing {len(missing)} data files")

def detect_encoding(file_path: str) -> str:
    """自动检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)
        result = chardet.detect(rawdata)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except Exception as e:
        logging.warning(f"Encoding detection failed for {file_path}, using fallback: {str(e)}")
        return 'ISO-8859-1'

def safe_read_csv(file_path: str) -> pd.DataFrame:
    """安全读取CSV文件"""
    encoding = detect_encoding(file_path)
    logging.info(f"Reading {os.path.basename(file_path)} with encoding: {encoding}")
    
    try:
        return pd.read_csv(
            file_path,
            encoding=encoding,
            on_bad_lines='warn',
            low_memory=False
        )
    except UnicodeDecodeError:
        logging.warning("Fallback to ISO-8859-1 encoding")
        return pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='warn')

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名格式"""
    # 转换为小写并标准化列名
    df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)
    
    # 处理常见列名变体
    column_mapping = {
        'hadmid': 'hadm_id',
        'hospstayid': 'hadm_id',
        'icd9code': 'icd9_code',
        'icd_code': 'icd9_code'
    }
    return df.rename(columns=column_mapping)

def process_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """处理诊断数据"""
    # 清洗ICD代码
    if 'icd9_code' not in diagnoses.columns:
        raise KeyError("ICD9_CODE column missing in diagnoses data")
    
    diagnoses['icd_root'] = (
        diagnoses['icd9_code']
        .astype(str)
        .str.replace('.', '', regex=False)
        .str[:3]
        .str.pad(3, fillchar='0')
    )
    
    # 创建特征矩阵
    diag_features = (
        diagnoses.groupby(['hadm_id', 'icd_root'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('diag_')
    )
    return diag_features

def process_procedures(procedures: pd.DataFrame) -> pd.DataFrame:
    """处理手术数据"""
    if 'icd9_code' not in procedures.columns:
        raise KeyError("ICD9_CODE column missing in procedures data")
    
    procedures['icd_root'] = (
        procedures['icd9_code']
        .astype(str)
        .str.replace('.', '', regex=False)
        .str[:2]
        .str.pad(2, fillchar='0')
    )
    
    proc_features = (
        procedures.groupby(['hadm_id', 'icd_root'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('proc_')
    )
    return proc_features

def load_mimic_data() -> Tuple[pd.DataFrame, pd.Series]:
    """主数据加载函数"""
    check_files_exist()
    
    try:
        # 读取并标准化数据
        admissions = normalize_columns(safe_read_csv('notebook\\ADMISSIONS.csv'))
        diagnoses = normalize_columns(safe_read_csv('notebook\\DIAGNOSES_ICD.csv'))
        procedures = normalize_columns(safe_read_csv('notebook\\PROCEDURES_ICD.csv'))
        
        # 验证必要列存在
        for df, name in zip([admissions, diagnoses, procedures], 
                          ['ADMISSIONS', 'DIAGNOSES_ICD', 'PROCEDURES_ICD']):
            if 'hadm_id' not in df.columns:
                raise KeyError(f"HADM_ID column missing in {name}")
        
        # 特征工程
        logging.info("Processing diagnoses features...")
        diag_features = process_diagnoses(diagnoses)
        
        logging.info("Processing procedures features...")
        proc_features = process_procedures(procedures)
        
        # 合并数据
        logging.info("Merging datasets...")
        merged = (
            admissions
            .merge(diag_features, on='hadm_id', how='left')
            .merge(proc_features, on='hadm_id', how='left')
        )
        
        # 处理目标变量
        if 'hospital_expire_flag' not in merged.columns:
            available_cols = ', '.join(merged.columns)
            raise KeyError(f"Target column hospital_expire_flag missing. Available columns: {available_cols}")
        
        y = merged['hospital_expire_flag'].astype(int)
        
        # 选择特征
        exclude_cols = ['row_id', 'subject_id', 'hadm_id', 'admittime', 
                       'dischtime', 'deathtime', 'edregtime', 'edouttime']
        X = merged.drop(columns=[c for c in exclude_cols if c in merged.columns])
        X = X.select_dtypes(include='number').fillna(0)
        
        logging.info(f"Final dataset shape: {X.shape}")
        return X, y
        
    except Exception as e:
        logging.error("Data loading failed!")
        logging.error("Possible solutions:")
        logging.error("1. Check ICD9_CODE columns exist in diagnosis/procedure files")
        logging.error("2. Verify file encodings using chardet")
        logging.error("3. Check HADM_ID consistency across files")
        raise

