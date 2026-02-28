# krx_config.py

# ==============================================================================
# 1. Column Name Translations (Korean to English)
# ==============================================================================
# We strictly map the KRX DB native Korean names to standard English names.
# If a column exists in the S3 DB but is missing here, the DataLoader will
# intentionally raise an error to enforce research integrity.

COLUMN_TRANSLATIONS = {
    # Expected KRX DB Columns (Trades & Quotes)
    "date": "date",    # Passthroughs for embedded headers
    "time": "time",
    "체결일자": "date",
    "체결시각": "time",
    "종목코드": "iscd",
    "선물 단축 종목코드": "shrn_iscd",
    "체결번호": "trd_no",
    "보드ID": "board_id",
    "세션ID": "session_id",
    "시장ID": "market_id",
    "시장구분코드": "market_type_code",
    "정규시간외구분코드": "reg_ext_type_code",
    "대량매매구분코드": "block_trade_code",
    "데이터구분": "data_type",
    "메세지일련번호": "msg_seq_no",
    "최종매도매수구분코드": "seln_byov_cls",
    
    # Legacy Quote specific
    "호가일자": "date",
    "호가접수시각": "time",
    "직전체결가격": "last_trd_prc",
    "직전체결수량": "last_trd_qty",
    "누적체결수량": "acml_trd_qty",
    "폴더명": "folder_name",
    "번호": "trd_no",
    "일자": "file_date",
    "종목인덱스": "iscd_idx",
    "ACVOL_1": "acml_trd_qty",
    "TRDPRC_1": "last_trd_prc",
    "TRDVOL_1": "last_trd_qty",
    
    # Trade execution stats
    "체결가격": "trd_prc",
    "체결수량": "trd_qty",
    "체결유형코드": "trd_type_code",
    "종가": "close_prc",
    "시가": "oprc",
    "고가": "hgpr",
    "저가": "lwpr",
    "직전가격": "last_trd_prc",
    "근월물체결가격": "front_mth_trd_prc",
    "원월물체결가격": "back_mth_trd_prc",
    "실시간상한가": "rt_upper_lmt",
    "실시간하한가": "rt_lower_lmt",
    "LP보유수량": "lp_qty",
    "누적거래대금": "acml_trd_value",
    "누적매도미결제약정수량": "acml_sell_otst_qty",
    "누적매수미결제약정수량": "acml_buy_otst_qty",  
    
    # Priority Quotes during Trade
    "매도1단계우선호가가격": "askp1",
    "매도1단계우선호가잔량": "askp1_qty",
    "매수1단계우선호가가격": "bidp1",
    "매수1단계우선호가잔량": "bidp1_qty",
    
    # Party details
    "매도호가가격": "askp",
    "매도국가코드": "sell_country_code",
    "매도투자자구분코드": "sell_trader_type",
    "매도위탁자기구분코드": "sell_broker_type",
    "매도외국인투자자구분코드": "sell_foreign_type",
    "매도회원사주문시각": "sell_member_order_time",
    "매도호가체결수량": "ask_trd_qty",
    "매도PT구분코드": "sell_pt_code",

    "매수호가가격": "bidp",
    "매수국가코드": "buy_country_code",
    "매수투자자구분코드": "buy_trader_type",
    "매수위탁자기구분코드": "buy_broker_type",
    "매수외국인투자자구분코드": "buy_foreign_type",
    "매수회원사주문시각": "buy_member_order_time",
    "매수호가체결수량": "bid_trd_qty",
    "매수PT구분코드": "buy_pt_code",
}

# Auto-generate Quotes (If any KRX Quote DB dumps use these precise labels)
for i in range(1, 11):
    COLUMN_TRANSLATIONS[f"매도호가{i}"] = f"askp{i}"
    COLUMN_TRANSLATIONS[f"매수호가{i}"] = f"bidp{i}"
    COLUMN_TRANSLATIONS[f"매도호가 잔량{i}"] = f"ask_qty_{i}"
    COLUMN_TRANSLATIONS[f"매수호가 잔량{i}"] = f"bid_qty_{i}"
    COLUMN_TRANSLATIONS[f"매도호가 건수{i}"] = f"ask_cnt_{i}"
    COLUMN_TRANSLATIONS[f"매수호가 건수{i}"] = f"bid_cnt_{i}"
    
    # Legacy specific formats and their typos
    COLUMN_TRANSLATIONS[f"매도{i}단계우선호가가격"] = f"askp{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계우선호가가격"] = f"bidp{i}"
    COLUMN_TRANSLATIONS[f"매도{i}단계우선호가잔량"] = f"ask_qty_{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계우선호가잔량"] = f"bid_qty_{i}"
    COLUMN_TRANSLATIONS[f"매도{i}단계우선호가건수"] = f"ask_cnt_{i}"
    COLUMN_TRANSLATIONS[f"BEST_BID{i}"] = f"bidp{i}"
    COLUMN_TRANSLATIONS[f"BEST_ASK{i}"] = f"askp{i}"
    COLUMN_TRANSLATIONS[f"BEST_BSIZ{i}"] = f"bid_qty_{i}"
    COLUMN_TRANSLATIONS[f"BEST_ASIZ{i}"] = f"ask_qty_{i}"
    
    # Resolving S3 text spacing inconsistencies found in 2017 .txt
    COLUMN_TRANSLATIONS[f"매도{i}단계우선호가 잔량"] = f"ask_qty_{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계우선호가 잔량"] = f"bid_qty_{i}"
    COLUMN_TRANSLATIONS[f"매도{i}단계 우선호가가격"] = f"askp{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계 우선호가가격"] = f"bidp{i}"
    COLUMN_TRANSLATIONS[f"매도{i}단계우선 호가잔량"] = f"ask_qty_{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계우선 호가잔량"] = f"bid_qty_{i}"
    COLUMN_TRANSLATIONS[f"매도{i}단계 우선호가잔량"] = f"ask_qty_{i}"
    COLUMN_TRANSLATIONS[f"매수{i}단계 우선호가잔량"] = f"bid_qty_{i}"

# ==============================================================================
# 2. KRX Code Parsing Rules (Both Pre-2026 and 2026+)
# ==============================================================================

MONTH_MAP = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12
}
MONTH_MAP_INV = {v: k for k, v in MONTH_MAP.items()}

YEAR_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'
]

YEAR_MAP_BASE_2020 = {YEAR_CHARS[i]: 2020 + i for i in range(len(YEAR_CHARS))}
YEAR_MAP_BASE_2020_INV = {2020 + i: YEAR_CHARS[i] for i in range(len(YEAR_CHARS))}

PROD_MAP = {
    '1': 'Futures',
    '2': 'Call_Option',
    '3': 'Put_Option',
    '4': 'Spread'
}

UNDERLYING_MAP = {
    '01': 'KOSPI200',
    '05': 'KOSDAQ150',
    '06': 'KRX300',
    '11': 'Mini_KOSPI200',
    '12': 'SSFs',
    '1M': 'Weekly_Options',
}

# ==============================================================================
# 3. Categorical Trader Types (Internal Codes -> English)
# ==============================================================================

TRADER_TYPES = {
    '1000': 'Securities',
    '2000': 'Insurance',
    '3000': 'Investment_Trust',
    '3100': 'Private_Equity',
    '4000': 'Bank',
    '5000': 'Other_Finance',
    '6000': 'Pension_Fund',
    '7000': 'Government',
    '7100': 'Local_Government',
    '8000': 'Individual',
    '9000': 'Foreigner',
    '9001': 'Foreign_Registered',
    '9002': 'Foreign_Unregistered',
    '9999': 'Other_Corporate'
}

def translate_columns(columns: list) -> list:
    """
    Translate KRX Korean column names to English strictly.
    Raises ValueError if a column isn't mapped, enforcing research integrity.
    """
    result = []
    unmapped = []
    for col in columns:
        col_en = COLUMN_TRANSLATIONS.get(col)
        if col_en is None:
            unmapped.append(col)
        result.append(col_en)
        
    if unmapped:
        raise ValueError(
            f"RESEARCH INTEGRITY ERROR: Found unmapped KRX database columns: {unmapped}. "
            "Please add them to krx_config.COLUMN_TRANSLATIONS to proceed."
        )
    return result
