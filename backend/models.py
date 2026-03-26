from pydantic import BaseModel
from typing import Optional

class SettingsUpdate(BaseModel):
    stub_mode: Optional[str] = None
    retrain: Optional[str] = None
    xgboost: Optional[str] = None
    threshold: Optional[str] = None
    admin_password: Optional[str] = None
    # MetaTrader 5 credentials
    mt5_account: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    # Scalping settings
    scalp_confirmations: Optional[str] = None
    scalp_max_trades: Optional[str] = None
    scalp_ticker: Optional[str] = None
    risk_reward_ratio: Optional[str] = None
