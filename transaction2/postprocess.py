import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# FIXME : missing session(pre post main openingcross closingcross singleprice) and order type(fok ioc gtc / limit market cancel modify) information
# Baseline mappings for categorical features
# Users can expand these later. Currently mapped to standard ISO/KRX guesses.
COUNTRY_CODES = {
    '826': 'UK',
    '840': 'US',
    '410': 'KR',
    '344': 'HK',
    '702': 'SG'
}

TRADER_TYPES = {
    '1000': 'Securities',
    '2000': 'Insurance',
    '3000': 'Investment_Trust',
    '4000': 'Bank',
    '5000': 'Other_Finance',
    '6000': 'Pension_Fund',
    '7000': 'Government',
    '7100': 'Local_Government',
    '8000': 'Individual',
    '9000': 'Foreigner',
    '9001': 'Foreign_Reg_Institution',
    '9002': 'Foreign_Reg_Individual',
    '9999': 'Other_Corporation'
}

BROKER_TYPES = {
    '11': 'Member_Broker',
    '21': 'NonMember_Domestic_Broker',
    '31': 'NonMember_Foreign_Broker'
}

FOREIGN_TYPES = {
    '10': 'Domestic',
    '20': 'Foreigner'
}

QUOTE_EVENT_TYPES = {
    '1': 'New_Order',
    '2': 'Modification',
    '3': 'Cancellation'
}

QUOTE_BUY_SELL = {
    '1': 'Sell',
    '2': 'Buy'
}

QUOTE_CONDITIONS = {
    '1': 'Limit',
    '2': 'Market',
    '3': 'Conditional_Limit',
    '4': 'Best_Limit',
    '5': 'FOK',
    '6': 'IOC',
    '7': 'Price_Optimized'
}

def clean_and_convert_datetime(df: pd.DataFrame, date_col: str = 'date', time_col: str = 'time') -> pd.DataFrame:
    """
    Combines date and time columns, creating a high precision datetime index.
    Also handles buy_member_order_time and sell_member_order_time if present.
    """
    df = df.copy()
    if date_col in df.columns and time_col in df.columns:
        datetime_strs = df[date_col].astype(str) + df[time_col].astype(str).str.zfill(9)
        df['datetime'] = pd.to_datetime(datetime_strs, format='%Y%m%d%H%M%S%f', errors='coerce')
        
    for order_time_col in ['buy_member_order_time', 'sell_member_order_time']:
        if order_time_col in df.columns:
            # Handle float representations correctly
            dt_strs = df[date_col].astype(str) + df[order_time_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(9)
            df[order_time_col] = pd.to_datetime(dt_strs, format='%Y%m%d%H%M%S%f', errors='coerce')
            
    # Set the strict datetime index
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
        # Force the type implicitly
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
    return df

def map_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps country codes, trader types, etc. to human readable categories,
    and returns them strictly as pandas categorical types to save memory.
    """
    df = df.copy()
    
    mapping_configs = [
        ('sell_country_code', COUNTRY_CODES),
        ('buy_country_code', COUNTRY_CODES),
        ('sell_trader_type', TRADER_TYPES),
        ('buy_trader_type', TRADER_TYPES),
        ('sell_broker_type', BROKER_TYPES),
        ('buy_broker_type', BROKER_TYPES),
        ('sell_foreign_type', FOREIGN_TYPES),
        ('buy_foreign_type', FOREIGN_TYPES),
        ('정정취소구분코드', QUOTE_EVENT_TYPES), # Modify/Cancel Code
        ('매도매수구분코드', QUOTE_BUY_SELL),    # Sell/Buy Code
        ('호가유형코드', QUOTE_CONDITIONS)      # Quote Type Code
    ]
    
    for col, mapping in mapping_configs:
        if col in df.columns:
            # Map values and fill unmapped ones with their original code
            mapped = df[col].astype(str).map(mapping)
            df[col] = mapped.fillna(df[col].astype(str)).astype('category')
            
    return df

def align_trade_and_quote(trade_df: pd.DataFrame, quote_df: pd.DataFrame, tolerance_ms: int = 100) -> pd.DataFrame:
    """
    Aligns trade instances with the exact preceding quote strictly matching on timestamp.
    Because matching engines execute against the book, the preceding orderbook 
    reflects the exact liquidity taken.
    
    True KRX Quotes (opk2iq / fuk2iq) contain events. This AsOf Merge aligns trades to the last known 
    book state immediately prior to the trade execution microsecond.
    """
    # Force strictly matching M8[ns] dtypes prior to sorting and merging
    if str(trade_df.index.dtype) != 'datetime64[ns]':
         trade_df.index = pd.to_datetime(trade_df.index)
    if str(quote_df.index.dtype) != 'datetime64[ns]':
         quote_df.index = pd.to_datetime(quote_df.index)
         
    # Ensure both dataframes are sorted by their datetime index
    if not trade_df.index.is_monotonic_increasing:
        trade_df = trade_df.sort_index()
    if not quote_df.index.is_monotonic_increasing:
        quote_df = quote_df.sort_index()
        
    # Standardize column prefixing to avoid overlap
    quote_cols_to_keep = [col for col in quote_df.columns if col not in ['date', 'time', 'iscd']]
    quote_subset = quote_df[quote_cols_to_keep].copy()
    quote_subset.columns = [f"quote_{col}" for col in quote_cols_to_keep]
    
    merged = pd.merge_asof(
        trade_df, 
        quote_subset, 
        left_index=True, 
        right_index=True, 
        direction='backward',
        tolerance=pd.Timedelta(f"{tolerance_ms}ms")
    )
    
    # Calculate quoting dynamics (spread, mid-price, imbalance) exactly at the trade microsecond
    # Standard translation schemas for krx-options-quote usually map '매도1단계우선호가가격' to askp1
    ask_col = 'quote_askp1' if 'quote_askp1' in merged.columns else 'quote_BEST_ASK1'
    bid_col = 'quote_bidp1' if 'quote_bidp1' in merged.columns else 'quote_BEST_BID1'
    
    if ask_col in merged.columns and bid_col in merged.columns:
        merged['quote_mid_price'] = (merged[ask_col].astype(float) + merged[bid_col].astype(float)) / 2.0
        merged['quote_spread'] = merged[ask_col].astype(float) - merged[bid_col].astype(float)
        
    ask_qty_col = 'quote_ask_qty_1' if 'quote_ask_qty_1' in merged.columns else 'quote_BEST_ASIZ1'
    bid_qty_col = 'quote_bid_qty_1' if 'quote_bid_qty_1' in merged.columns else 'quote_BEST_BSIZ1'
    
    if ask_qty_col in merged.columns and bid_qty_col in merged.columns:
        b_qty = merged[bid_qty_col].astype(float)
        a_qty = merged[ask_qty_col].astype(float)
        merged['quote_imbalance'] = (b_qty - a_qty) / ((b_qty + a_qty).replace(0, np.nan))
                                    
    return merged

def encode_microstructure_state(df: pd.DataFrame, is_quote=False) -> pd.DataFrame:
    """
    Combines fields that represent the state of the matching engine into one categorical column.
    Useful for distinguishing between continuous trading, opening crossing, and block trades.
    If is_quote=True, it also encodes order modify/cancel events.
    """
    df = df.copy()
    
    # Required columns for the full composite state representation
    # '장상태구분코드' is KRX standard for Session ID, '보드ID' for Board ID
    state_cols = ['board_id', 'session_id', 'market_id', 'trd_type_code', 'market_type_code', '보드ID', '세션ID', '시장ID', '장상태구분코드']
    
    if is_quote:
        # For quotes, '정정취소구분코드', '매도매수구분코드' usually translated to e.g., 'mod_cxl_code', 'sell_buy_code'
        # Add them dynamically if they exist
        for q_col in ['mod_cxl_code', 'sell_buy_code', 'quote_type_code', '정정취소구분코드', '매도매수구분코드', '호가유형코드']:
            if q_col in df.columns and q_col not in state_cols:
                state_cols.append(q_col)
                
    # Check what columns exist in the DataFrame
    available_cols = [c for c in state_cols if c in df.columns]
    
    if len(available_cols) > 0:
        # Create a combined string e.g. "U2_40_DRV_1_2"
        # We fill nulls with 'None' string to represent missing state info in the category
        combined = df[available_cols[0]].astype(str).fillna('None')
        for col in available_cols[1:]:
            combined = combined + "_" + df[col].astype(str).fillna('None')
            
        df['market_state'] = combined.astype('category')
        
    return df

def extract_topology_graphs(df: pd.DataFrame, topology_type: str = 'country') -> pd.DataFrame:
    """
    Extracts directed network relationships between predefined nodes.
    Supported topology_types: 'country', 'trader', 'broker', 'foreign'.
    """
    mapping = {
        'country': ('sell_country_code', 'buy_country_code', '매도국가코드', '매수국가코드', 'quote_매도국가코드', 'quote_매수국가코드'),
        'trader': ('sell_trader_type', 'buy_trader_type', '매도위탁자기구분코드', '매수위탁자기구분코드', 'quote_매도위탁자기구분코드', 'quote_매수위탁자기구분코드'),
        'broker': ('sell_broker_type', 'buy_broker_type', '매도회원사기구분코드', '매수회원사기구분코드', 'quote_매도회원사기구분코드', 'quote_매수회원사기구분코드'),
        'foreign': ('sell_foreign_type', 'buy_foreign_type', '매도외국인투자자구분코드', '매수외국인투자자구분코드', 'quote_매도외국인투자자구분코드', 'quote_매수외국인투자자구분코드')
    }
    
    if topology_type not in mapping:
        raise ValueError(f"Topology {topology_type} not supported.")
        
    en_s, en_b, kr_s, kr_b, qkr_s, qkr_b = mapping[topology_type]
    
    # Check which suffix exists in the DataFrame
    if en_s in df.columns and en_b in df.columns:
        sell_node, buy_node = en_s, en_b
    elif qkr_s in df.columns and qkr_b in df.columns:
        sell_node, buy_node = qkr_s, qkr_b
    elif kr_s in df.columns and kr_b in df.columns:
        sell_node, buy_node = kr_s, kr_b
    else:
        logger.warning(f"Columns for topology {topology_type} not found.")
        return pd.DataFrame()
        
    # Isolate relevant columns
    # In trade files, volume is trd_qty, price is trd_prc. In raw KRX logs it might be '실정정취소호가수량' or '체결수량', and '직후가격'
    qty_col = 'trd_qty' if 'trd_qty' in df.columns else '순간체결수량'
    prc_col = 'trd_prc' if 'trd_prc' in df.columns else '직후가격'
    
    cols_to_keep = [sell_node, buy_node]
    if qty_col in df.columns: cols_to_keep.append(qty_col)
    if prc_col in df.columns: cols_to_keep.append(prc_col)
    
    # Add datetime if it exists as column, else reset index if it's the index
    if 'datetime' in df.index.names or isinstance(df.index, pd.DatetimeIndex) or str(df.index.dtype) == 'datetime64[ns]':
        temp = df[cols_to_keep].reset_index()
    else:
        temp = df[cols_to_keep].copy()

    # The group operation itself is standard, the kernel processing will happen on top of this.
    temp = temp.rename(columns={sell_node: 'sell_node', buy_node: 'buy_node', qty_col: 'trd_qty', prc_col: 'trd_prc'})
    temp['topology_type'] = topology_type
    
    return temp

def apply_triangular_kernel_aggregation(
    df: pd.DataFrame, 
    freq: str = '5s', 
    kernel_width_ms: int = 10000,
    price_cols: list = ['trd_prc', 'quote_askp1', 'quote_bidp1', 'quote_mid_price'],
    sum_cols: list = ['trd_qty', 'quote_ask_qty_1', 'quote_bid_qty_1']
) -> pd.DataFrame:
    """
    Groups data by frequency and 'market_state', summing quantities, and calculating 
    triangular-kernel weighted averages for prices centered exactly on the interval boundaries.
    """
    if 'market_state' not in df.columns:
        df['market_state'] = 'None'
        
    df = df.copy()
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
        
    # Standard sum aggregations
    sum_agg = {col: 'sum' for col in sum_cols if col in df.columns}
    
    # We must explicitly define the group boundaries to calculate the kernel distances
    start_time = df.index.min().floor(freq)
    end_time = df.index.max().ceil(freq)
    if pd.isna(start_time): return pd.DataFrame()
    
    boundaries = pd.date_range(start_time, end_time, freq=freq)
    
    # Result container
    results = []
    
    # We will use vectorized distance calculation per boundary for speed
    # df.index is the timestamp of the event
    kernel_half_width = pd.Timedelta(milliseconds=kernel_width_ms / 2)
    
    for bound in boundaries:
        # Filter rows within the kernel support window [bound - half_width, bound + half_width]
        masked = df[(df.index >= bound - kernel_half_width) & (df.index <= bound + kernel_half_width)].copy()
        if masked.empty:
            continue
            
        # Calculate triangular weights: max(0, 1 - |t - bound| / half_width)
        time_diff = np.abs(masked.index - bound)
        weights = 1.0 - (time_diff / kernel_half_width)
        weights = np.clip(weights, 0, None) # Should be redundant because of the mask, but safe
        masked['_weight'] = weights
        
        # Group by market_state in this localized window to sum and average
        for state, group_df in masked.groupby('market_state'):
            if group_df.empty: continue
            
            row = {'datetime': bound, 'market_state': state}
            
            # Weighted average for prices
            weight_sum = group_df['_weight'].sum()
            for pc in price_cols:
                if pc in group_df.columns:
                    if weight_sum > 0:
                        row[pc] = (group_df[pc].astype(float) * group_df['_weight']).sum() / weight_sum
                    else:
                        row[pc] = np.nan
                        
            results.append(row)
            
    # Compile kernel smoothed prices
    agg_prices = pd.DataFrame(results)
    if not agg_prices.empty:
        agg_prices.set_index('datetime', inplace=True)
        
    # Compute standard sums using normal resample (these map to the interval [bound, bound+freq))
    # Note: Sums are usually bucketed into the interval, whereas the kernel is centered on the boundary
    # This aligns the sum of the interval starting AT 'bound' with the kernel centered AT 'bound'
    sum_df = df.groupby(['market_state', pd.Grouper(freq=freq)])[list(sum_agg.keys())].sum().reset_index()
    sum_df.rename(columns={'datetime': 'interval_start'}, inplace=True) # Usually grouper retains 'datetime' or index name

    # Because resample/grouper might name the time column differently based on index, we ensure we have 'datetime'
    if 'datetime' not in sum_df.columns and sum_df.columns.nlevels == 1:
         # Find the datetime column
         dt_cols = sum_df.select_dtypes(include=['datetime64']).columns
         if len(dt_cols) > 0:
             sum_df.rename(columns={dt_cols[0]: 'datetime'}, inplace=True)
             
    if agg_prices.empty:
        return sum_df
        
    # Merge the boundary-centered smoothed prices with the interval-starting sums
    sum_df.set_index(['datetime', 'market_state'], inplace=True)
    agg_prices.set_index(['market_state'], append=True, inplace=True)
    
    final = pd.merge(sum_df, agg_prices, left_index=True, right_index=True, how='outer').reset_index()
    return final

from typing import Dict
def build_topology_features(df: pd.DataFrame, freq: str = '5s', kernel_width_ms: int = 10000) -> Dict[str, pd.DataFrame]:
    """
    Extracts all topology networks (Country, Trader, Broker, Foreign) and applies
    the kernel smoothing & aggregation on a per-edge basis using a MultiIndex.
    """
    topologies = ['country', 'trader', 'broker', 'foreign']
    results = {}
    
    for topo in topologies:
        graph_df = extract_topology_graphs(df, topology_type=topo)
        if graph_df.empty: continue
        
        if 'datetime' in graph_df.columns:
            graph_df.set_index('datetime', inplace=True)
            
        start_time = graph_df.index.min().floor(freq)
        end_time = graph_df.index.max().ceil(freq)
        boundaries = pd.date_range(start_time, end_time, freq=freq)
        kernel_half_width = pd.Timedelta(milliseconds=kernel_width_ms / 2)
        
        edge_results = []
        for bound in boundaries:
            masked = graph_df[(graph_df.index >= bound - kernel_half_width) & 
                              (graph_df.index <= bound + kernel_half_width)].copy()
            if masked.empty: continue
            
            time_diff = np.abs(masked.index - bound)
            weights = 1.0 - (time_diff / kernel_half_width)
            masked['_weight'] = np.clip(weights, 0, None)
            
            # Group by specific directed edge
            for (sender, receiver), edge_df in masked.groupby(['sell_node', 'buy_node']):
                 weight_sum = edge_df['_weight'].sum()
                 avg_prc = (edge_df['trd_prc'].astype(float) * edge_df['_weight']).sum() / weight_sum if weight_sum > 0 else np.nan
                 
                 edge_results.append({
                     'datetime': bound,
                     'sell_node': sender,
                     'buy_node': receiver,
                     'avg_price': avg_prc
                 })
                 
        edge_prices = pd.DataFrame(edge_results)
        
        # Sums for the edges in standard intervals
        edge_sums = graph_df.groupby(['sell_node', 'buy_node', pd.Grouper(freq=freq)])['trd_qty'].sum().reset_index()
        # Find time column
        dt_cols = edge_sums.select_dtypes(include=['datetime64']).columns
        if len(dt_cols) > 0:
             edge_sums.rename(columns={dt_cols[0]: 'datetime'}, inplace=True)
             
        if not edge_prices.empty:
            edge_prices.set_index(['datetime', 'sell_node', 'buy_node'], inplace=True)
            edge_sums.set_index(['datetime', 'sell_node', 'buy_node'], inplace=True)
            final_edge = pd.merge(edge_sums, edge_prices, left_index=True, right_index=True, how='outer')
            results[topo] = final_edge
        else:
            edge_sums.set_index(['datetime', 'sell_node', 'buy_node'], inplace=True)
            results[topo] = edge_sums
            
    return results

