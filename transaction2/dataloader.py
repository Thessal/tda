import io
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from krx_config import translate_columns

logger = logging.getLogger(__name__)

class KrxDataLoader:
    """
    A fast, strict, and maintainable KRX data loader.
    - Connects to AWS S3 using readonly credentials.
    - Designed specifically for native KRX database extracts (not broker crawled data).
    - Dynamically queries `YYYYMM.json` to get the correct schema for each file.
    - Strictly enforces translations using `krx_config.COLUMN_TRANSLATIONS`.
    - If a KRX file contains untranslated (new/unexpected) columns, it instantly raises a Research Integrity Error.
    - Performs hyper-fast vectorized datetime parsing.
    """
    def __init__(
        self, 
        key_csv_path: str = '/home/jongkook90/antigravity/tda/krx-readonly_accessKeys.csv', 
        bucket: str = 'kospi200-research', 
        prefix: str = 'krx-futures-trade/', 
        region_name: str = 'ap-northeast-2'
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.cached_schemas = {}
        
        # Load AWS credentials from CSV
        creds_df = pd.read_csv(key_csv_path)
        access_key_id = creds_df['Access key ID'].iloc[0]
        secret_access_key = creds_df['Secret access key'].iloc[0]

        import boto3
        self.s3_client = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )

    def _get_file_list(self) -> List[str]:
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        files = []
        for page in pages:
            for obj in page.get('Contents', []):
                files.append(obj['Key'])
        return files

    def _get_file_content(self, key: str) -> io.BytesIO:
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return io.BytesIO(response['Body'].read())

    def load_data(
        self, 
        startdate: str, 
        enddate: str, 
        date_col: str = "date", 
        time_col: str = "time",
        date_format: str = "%Y%m%d", 
        time_format: str = "%H%M%S%f",
        query: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        
        logger.info(f"*** Loading Strict KRX data from AWS S3 ({self.bucket}/{self.prefix}) ***")
        paths = self._get_file_list()
        dfs = {}
        
        for path in sorted(paths):
            file_name = path.split("/")[-1]
            ext = '.'.join(file_name.split(".")[1:])
            stem = file_name.split(".")[0]
            
            if ext in ["csv.zstd", "csv.gz"]:
                date_str = stem.split("_")[-1]  
                
                if startdate <= date_str <= enddate:
                    month_str = date_str[:6] 
                    
                    cache_key = f"{self.prefix}{month_str}"
                    if cache_key not in self.cached_schemas:
                        if ext == "csv.gz":
                            self.cached_schemas[cache_key] = ("EMBEDDED", {})
                        else:
                            schema_key_json = f"{self.prefix}{month_str}.json"
                            schema_key_txt = f"{self.prefix}{month_str}.txt"
                            if schema_key_json in paths:
                                spec_content = self._get_file_content(schema_key_json)
                                spec = json.load(spec_content)
                                curr_kr_names = [a for a, _ in spec]
                                # Translating S3 JSON native types to pandas string counterparts
                                curr_dtypes = {a: "string" if b == "str" else b for a, b in spec} 
                                self.cached_schemas[cache_key] = (curr_kr_names, curr_dtypes)
                            elif schema_key_txt in paths:
                                spec_content = self._get_file_content(schema_key_txt)
                                content_str = spec_content.read().decode('cp949', errors='replace')
                                curr_kr_names = [
                                    l.strip() for l in content_str.split('\n') 
                                    if l.strip() and not l.startswith('-') and ':' not in l
                                ]
                                # Since TXT lacks types, default everything to string to preserve leading zeros
                                curr_dtypes = {a: "string" for a in curr_kr_names}
                                self.cached_schemas[cache_key] = (curr_kr_names, curr_dtypes)
                            else:
                                raise ValueError(f"Schema file (JSON or TXT) not found for month {month_str}.")
                    
                    schema_data = self.cached_schemas[cache_key]
                    compression_type = "zstd" if ext == "csv.zstd" else "gzip"
                    csv_content = self._get_file_content(path)
                    
                    if schema_data[0] == "EMBEDDED":
                        data = pd.read_csv(
                            csv_content, 
                            compression=compression_type,
                            index_col=False,
                            encoding="cp949",
                            encoding_errors="replace"
                        )
                        # Translate the embedded headers strictly
                        data.columns = [c.strip() for c in data.columns]
                        data.columns = translate_columns(data.columns)
                    else:
                        curr_kr_names, curr_dtypes = schema_data
                        curr_en_names = translate_columns(curr_kr_names)
                        
                        data = pd.read_csv(
                            csv_content, 
                            compression=compression_type, 
                            names=curr_en_names, 
                            index_col=False, 
                            encoding="cp949",
                            encoding_errors="replace",
                            dtype={en: curr_dtypes[kr] for en, kr in zip(curr_en_names, curr_kr_names)}
                        )

                    
                    if query:
                        data.query(query, inplace=True)
                        
                    # 2. Super-Fast Vectorized Datetime Parsing
                    # Instead of applying format string conversions to each row, vectorized `pd.to_datetime` natively 
                    # connects the C backend to map the ISO arrays correctly
                    if date_col in data.columns and time_col in data.columns:
                        datetime_strs = data[date_col].astype(str) + data[time_col].astype(str)
                        data.index = pd.to_datetime(
                            datetime_strs, 
                            format=date_format + time_format,
                            errors='coerce'  # Prevents crashing on corrupted single ticks, marks NaT
                        )
                    else:
                        logger.warning(f"Could not construct DatetimeIndex. Expected columns {date_col} and {time_col} not found.")
                    
                    dfs[date_str] = data
                    logger.info(f"Loaded DataFrame for: {stem}")

        return dfs
