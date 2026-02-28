import logging
import re
from dataclasses import dataclass
from typing import Optional

from krx_config import (
    MONTH_MAP, MONTH_MAP_INV, YEAR_CHARS, YEAR_MAP_BASE_2020, YEAR_MAP_BASE_2020_INV,
    PROD_MAP, UNDERLYING_MAP, TRADER_TYPES
)

logger = logging.getLogger(__name__)

@dataclass
class KrxInstrument:
    """
    Parses a standard KRX derivative code (e.g. KR4101W60000).
    Exposes structurally parsed fields like underlying asset, derivative type,
    expiration year, and month.
    """
    code: str
    prod_type: str            # e.g., 'Futures', 'Call_Option'
    underlying: str           # e.g., 'KOSPI200'
    expiry_year: int          # e.g., 2026
    expiry_month: int         # e.g., 6 (June)
    strike_price: Optional[float] = None
    
    @classmethod
    def from_code(cls, code: str) -> "KrxInstrument":
        if len(code) != 12:
            raise ValueError(f"Standard KRX code must be 12 characters long, got: {code}")
            
        # Example format: KR4 1 01 W 6 000 0
        # Pos 0-2: Country/Market code (KR4 for derivatives usually)
        # Pos 3: Product type (1=Fut, 2=Call, 3=Put, 4=Spread)
        # Pos 4-5: Underlying (01=KOSPI200, etc.)
        # Pos 6: Expiry Year char (e.g., W, C, 6, etc.)
        # Pos 7: Expiry Month char (1-9, A-C)
        # Pos 8-10: Strike Price or Specific Indentifier (000 for Futures)
        # Pos 11: Check digit
        
        prod_marker = code[3]
        underlying_marker = code[4:6]
        year_char = code[6]
        month_char = code[7]
        strike_raw = code[8:11]
        
        prod_type = PROD_MAP.get(prod_marker, f"Unknown({prod_marker})")
        underlying = UNDERLYING_MAP.get(underlying_marker, f"Unknown({underlying_marker})")
        
        # Parse month
        expiry_month = MONTH_MAP.get(month_char)
        if expiry_month is None:
            raise ValueError(f"Invalid month character '{month_char}' in code '{code}'")

        # Parse year (Standard KRX Rolling 30 char scale)
        if year_char in YEAR_MAP_BASE_2020:
            expiry_year = YEAR_MAP_BASE_2020[year_char]
        else:
            # Fallback if somehow it's a direct digit not in standard 30-char scale
            # Prior to standard alphabets, sometimes they just use digit for year if recent
            try:
                expiry_year = 2020 + int(year_char)
            except ValueError:
                raise ValueError(f"Unknown expiration year character '{year_char}' in code '{code}'")
                
        # Parse Strike (For Options)
        strike_price = None
        if prod_marker in ['2', '3']:  # Options
            try:
                # E.g. "355" might mean 355 strike
                # If they include decimals, usually the last digit is a fraction,
                # For simplified parsing: just convert to float, though exact convention might vary
                # KRX strike decimals: typically 000 means standard, but options have actual strikes
                # Many KOSPI200 options strikes look like '355' for 355.0 points
                strike_price = float(strike_raw)
            except ValueError:
                strike_price = None

        return cls(
            code=code,
            prod_type=prod_type,
            underlying=underlying,
            expiry_year=expiry_year,
            expiry_month=expiry_month,
            strike_price=strike_price
        )
        
    def shift_month(self, n: int) -> "KrxInstrument":
        """
        Returns a newly formulated instrument structurally shifted by `n` months.
        (E.g., n=1 for the 'next month' contract).
        Note: Does not validate if the resulting contract actually trades.
        """
        # Calculate new month and year mathematically
        total_months = (self.expiry_year * 12) + (self.expiry_month - 1) + n
        new_year = total_months // 12
        new_month = (total_months % 12) + 1
        
        # Get characters for new year and month
        try:
            new_month_char = MONTH_MAP_INV[new_month]
            new_year_char = YEAR_MAP_BASE_2020_INV[new_year]
        except KeyError as e:
            raise ValueError(f"Target date {new_year}-{new_month} cannot be mapped to KRX standard characters: {e}")
            
        # Reconstruct exactly the standard code replacing only the year and month chars
        # Note: Check digit (pos 11) is technically invalidated here. 
        # So we leave the 11th char as "X" or unverified flag if recalculating checksum is hard.
        # But commonly systems just accept matching prefixes.
        new_code = self.code[:6] + new_year_char + new_month_char + self.code[8:11] + "X"
        
        return KrxInstrument(
            code=new_code,
            prod_type=self.prod_type,
            underlying=self.underlying,
            expiry_year=new_year,
            expiry_month=new_month,
            strike_price=self.strike_price
        )

    def next_month(self) -> "KrxInstrument":
        return self.shift_month(1)
        
    def __repr__(self) -> str:
        s = f"KrxInstrument({self.code} -> {self.underlying} {self.prod_type} '{self.expiry_year}-{self.expiry_month:02d}'"
        if self.strike_price is not None:
            s += f" Strike={self.strike_price}"
        s += ")"
        return s
