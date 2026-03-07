
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

requirements_path = Path(__file__).resolve().with_name("requirements.txt")
install_cmd = f'python -m pip install -r "{requirements_path}"'

try:
    import numpy as np
    import pandas as pd
    from faker import Faker
except ModuleNotFoundError as exc:
    missing_module = exc.name or "unknown"
    print(
        "缺少 Python 套件，無法執行 BitoGuard 模擬器。\n"
        f"缺少模組: {missing_module}\n"
        f"請先安裝依賴：`{install_cmd}`",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
except ImportError as exc:
    print(
        "Python 套件已存在，但載入失敗，通常是安裝損壞或動態函式庫缺失。\n"
        f"詳細錯誤: {exc}\n"
        f"建議先重新安裝依賴：`{install_cmd}`",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


ASSET_CONFIG = {
    "USDT": {"network": "TRON", "price_twd": 32.0, "precision": 6},
    "BTC": {"network": "BTC", "price_twd": 2_200_000.0, "precision": 8},
    "ETH": {"network": "ETH", "price_twd": 110_000.0, "precision": 8},
}

TAIWAN_CITIES = ["台北市", "新北市", "桃園市", "新竹市", "台中市", "台南市", "高雄市", "宜蘭縣"]
FOREIGN_LOCATIONS = [
    ("JP", "Tokyo"),
    ("HK", "Hong Kong"),
    ("SG", "Singapore"),
    ("US", "Los Angeles"),
    ("GB", "London"),
    ("AE", "Dubai"),
]
TW_BANKS = [
    ("004", "臺灣銀行"),
    ("005", "土地銀行"),
    ("006", "合作金庫"),
    ("008", "華南銀行"),
    ("009", "彰化銀行"),
    ("013", "國泰世華"),
    ("700", "中華郵政"),
    ("812", "台新銀行"),
]
OCCUPATIONS = [
    "software_engineer", "student", "retail_worker", "driver", "freelancer",
    "office_staff", "sales", "designer", "operations", "teacher", "self_employed"
]
SOURCE_OF_FUNDS = ["salary", "savings", "investment", "business_income", "family_support", "freelance_income"]


SEGMENT_CONFIG = {
    "retail_investor": {
        "login_daily_rate": 0.55,
        "daily_action_lambda": 0.35,
        "event_weights": {
            "fiat_deposit": 0.25, "trade_buy": 0.34, "trade_sell": 0.16,
            "crypto_deposit": 0.08, "crypto_withdrawal": 0.05, "fiat_withdrawal": 0.12,
        },
        "amount_range": (0.03, 0.15),
        "night_prob": 0.05,
        "vpn_prob": 0.02,
        "devices": (1, 2),
        "banks": (1, 2),
    },
    "salary_cycle": {
        "login_daily_rate": 0.45,
        "daily_action_lambda": 0.22,
        "event_weights": {
            "fiat_deposit": 0.38, "trade_buy": 0.28, "trade_sell": 0.10,
            "crypto_deposit": 0.05, "crypto_withdrawal": 0.03, "fiat_withdrawal": 0.16,
        },
        "amount_range": (0.04, 0.18),
        "night_prob": 0.04,
        "vpn_prob": 0.01,
        "devices": (1, 2),
        "banks": (1, 2),
    },
    "pro_trader": {
        "login_daily_rate": 1.30,
        "daily_action_lambda": 1.15,
        "event_weights": {
            "fiat_deposit": 0.14, "trade_buy": 0.36, "trade_sell": 0.30,
            "crypto_deposit": 0.08, "crypto_withdrawal": 0.06, "fiat_withdrawal": 0.06,
        },
        "amount_range": (0.06, 0.22),
        "night_prob": 0.10,
        "vpn_prob": 0.03,
        "devices": (2, 4),
        "banks": (1, 3),
    },
    "dormant": {
        "login_daily_rate": 0.12,
        "daily_action_lambda": 0.06,
        "event_weights": {
            "fiat_deposit": 0.20, "trade_buy": 0.25, "trade_sell": 0.10,
            "crypto_deposit": 0.10, "crypto_withdrawal": 0.05, "fiat_withdrawal": 0.30,
        },
        "amount_range": (0.02, 0.08),
        "night_prob": 0.03,
        "vpn_prob": 0.01,
        "devices": (1, 1),
        "banks": (1, 1),
    },
    "cross_border": {
        "login_daily_rate": 0.70,
        "daily_action_lambda": 0.55,
        "event_weights": {
            "fiat_deposit": 0.12, "trade_buy": 0.22, "trade_sell": 0.18,
            "crypto_deposit": 0.20, "crypto_withdrawal": 0.18, "fiat_withdrawal": 0.10,
        },
        "amount_range": (0.04, 0.18),
        "night_prob": 0.15,
        "vpn_prob": 0.30,
        "devices": (2, 3),
        "banks": (1, 2),
    },
    "night_owl": {
        "login_daily_rate": 0.58,
        "daily_action_lambda": 0.33,
        "event_weights": {
            "fiat_deposit": 0.20, "trade_buy": 0.32, "trade_sell": 0.15,
            "crypto_deposit": 0.10, "crypto_withdrawal": 0.08, "fiat_withdrawal": 0.15,
        },
        "amount_range": (0.03, 0.12),
        "night_prob": 0.45,
        "vpn_prob": 0.02,
        "devices": (1, 2),
        "banks": (1, 2),
    },
    "vpn_nomad": {
        "login_daily_rate": 0.62,
        "daily_action_lambda": 0.28,
        "event_weights": {
            "fiat_deposit": 0.18, "trade_buy": 0.24, "trade_sell": 0.15,
            "crypto_deposit": 0.18, "crypto_withdrawal": 0.15, "fiat_withdrawal": 0.10,
        },
        "amount_range": (0.03, 0.13),
        "night_prob": 0.10,
        "vpn_prob": 0.55,
        "devices": (2, 3),
        "banks": (1, 2),
    },
}


SCHEMA = {
    "users": {
        "description": "帳戶主體與 KYC 相關屬性。truth 欄位是 simulator ground truth；實戰時可在 ingest adapter 隱藏。",
        "columns": [
            ("user_id", "string"), ("created_at", "datetime"), ("segment", "string"), ("risk_seed_band", "string"),
            ("kyc_level", "string"), ("occupation", "string"), ("monthly_income_twd", "float"),
            ("expected_monthly_volume_twd", "float"), ("declared_source_of_funds", "string"),
            ("residence_country", "string"), ("residence_city", "string"), ("nationality", "string"),
            ("activity_window", "string"), ("hidden_suspicious_label", "int"), ("observed_blacklist_label", "int"),
            ("scenario_types", "string"), ("evidence_tags", "string"),
        ],
    },
    "devices": {
        "description": "裝置主檔，可被多個 user 共用。",
        "columns": [
            ("device_id", "string"), ("device_type", "string"), ("os_family", "string"),
            ("app_channel", "string"), ("device_fingerprint", "string"), ("first_seen_at", "datetime"),
            ("shared_group_id", "string"),
        ],
    },
    "user_device_links": {
        "description": "user 與 device 的關聯。",
        "columns": [
            ("link_id", "string"), ("user_id", "string"), ("device_id", "string"),
            ("is_primary", "int"), ("first_seen_at", "datetime"), ("last_seen_at", "datetime"),
        ],
    },
    "bank_accounts": {
        "description": "銀行帳號主檔，可被多個 user 共用，支援共享銀行帳號情境。",
        "columns": [
            ("bank_account_id", "string"), ("bank_code", "string"), ("bank_name", "string"),
            ("country", "string"), ("currency", "string"), ("opened_at", "datetime"), ("shared_group_id", "string"),
        ],
    },
    "user_bank_links": {
        "description": "user 與 bank account 的關聯。",
        "columns": [
            ("link_id", "string"), ("user_id", "string"), ("bank_account_id", "string"),
            ("is_primary", "int"), ("linked_at", "datetime"),
        ],
    },
    "crypto_wallets": {
        "description": "內部錢包與外部錢包/cluster 都在此表中，利於建立關聯圖。",
        "columns": [
            ("wallet_id", "string"), ("wallet_kind", "string"), ("user_id", "string"),
            ("cluster_id", "string"), ("asset", "string"), ("network", "string"),
            ("risk_seed", "string"), ("created_at", "datetime"),
        ],
    },
    "login_events": {
        "description": "登入事件，支援 IP 異常跳動、新裝置、VPN 等特徵。",
        "columns": [
            ("login_id", "string"), ("user_id", "string"), ("occurred_at", "datetime"),
            ("device_id", "string"), ("ip_address", "string"), ("ip_country", "string"), ("ip_city", "string"),
            ("is_vpn", "int"), ("is_new_device", "int"), ("is_geo_jump", "int"), ("success", "int"),
            ("scenario_id", "string"), ("tags", "string"),
        ],
    },
    "fiat_transactions": {
        "description": "法幣出入金事件。",
        "columns": [
            ("fiat_txn_id", "string"), ("user_id", "string"), ("occurred_at", "datetime"),
            ("direction", "string"), ("amount_twd", "float"), ("currency", "string"),
            ("bank_account_id", "string"), ("method", "string"), ("status", "string"),
            ("scenario_id", "string"), ("tags", "string"),
        ],
    },
    "trade_orders": {
        "description": "撮合交易事件，將 fiat/crypto 串起來以計算 dwell time 與 volume mismatch。",
        "columns": [
            ("trade_id", "string"), ("user_id", "string"), ("occurred_at", "datetime"),
            ("side", "string"), ("base_asset", "string"), ("quote_asset", "string"),
            ("price_twd", "float"), ("quantity", "float"), ("notional_twd", "float"),
            ("fee_twd", "float"), ("order_type", "string"), ("status", "string"),
            ("scenario_id", "string"), ("tags", "string"),
        ],
    },
    "crypto_transactions": {
        "description": "鏈上加值/提領事件，透過 counterparty_wallet_id 建立 1-hop/2-hop/n-hop 關聯。",
        "columns": [
            ("crypto_txn_id", "string"), ("user_id", "string"), ("occurred_at", "datetime"),
            ("direction", "string"), ("asset", "string"), ("network", "string"),
            ("wallet_id", "string"), ("counterparty_wallet_id", "string"), ("amount_asset", "float"),
            ("amount_twd_equiv", "float"), ("tx_hash", "string"), ("status", "string"),
            ("onchain_risk_seed", "string"), ("scenario_id", "string"), ("tags", "string"),
        ],
    },
    "scenarios": {
        "description": "注入的行為劇本：AMLSim 風格 pattern injection。",
        "columns": [
            ("scenario_id", "string"), ("scenario_type", "string"), ("start_at", "datetime"),
            ("end_at", "datetime"), ("description", "string"),
        ],
    },
    "scenario_members": {
        "description": "每個 scenario 的參與 user / wallet / device。",
        "columns": [
            ("scenario_id", "string"), ("entity_type", "string"), ("entity_id", "string"), ("role", "string"),
        ],
    },
    "entity_edges": {
        "description": "關聯圖用 edge 表；可直接餵給 graph explorer 或 graph features job。",
        "columns": [
            ("edge_id", "string"), ("occurred_at", "datetime"), ("src_type", "string"), ("src_id", "string"),
            ("relation_type", "string"), ("dst_type", "string"), ("dst_id", "string"),
            ("amount_twd_equiv", "float"), ("asset", "string"), ("scenario_id", "string"),
        ],
    },
    "manifest": {
        "description": "資料集摘要。",
        "columns": [
            ("table_name", "string"), ("row_count", "int"),
        ],
    },
}


@dataclass
class SimulationConfig:
    n_users: int = 1200
    start_date: str = "2026-01-01"
    days: int = 30
    injection_window_days: int = 3
    suspicious_user_ratio: float = 0.03
    grey_user_ratio: float = 0.08
    observed_blacklist_ratio: float = 0.35
    locale: str = "zh_TW"
    seed: int = 42
    output_dir: str = "bitoguard_sim_output"


class TransactionSimulator:
    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.fake = Faker(config.locale)
        Faker.seed(config.seed)
        self.rng = random.Random(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        self.start_dt = datetime.fromisoformat(config.start_date)
        self.base_end_dt = self.start_dt + timedelta(days=max(1, config.days - config.injection_window_days))
        self.end_dt = self.start_dt + timedelta(days=config.days)
        self.id_counters = defaultdict(int)

        self.users: List[Dict] = []
        self.devices: List[Dict] = []
        self.user_device_links: List[Dict] = []
        self.bank_accounts: List[Dict] = []
        self.user_bank_links: List[Dict] = []
        self.crypto_wallets: List[Dict] = []
        self.login_events: List[Dict] = []
        self.fiat_transactions: List[Dict] = []
        self.trade_orders: List[Dict] = []
        self.crypto_transactions: List[Dict] = []
        self.scenarios: List[Dict] = []
        self.scenario_members: List[Dict] = []
        self.entity_edges: List[Dict] = []

        self.user_lookup: Dict[str, Dict] = {}
        self.user_state: Dict[str, Dict] = {}
        self.user_devices_map: Dict[str, List[str]] = defaultdict(list)
        self.user_banks_map: Dict[str, List[str]] = defaultdict(list)
        self.user_wallets_map: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.wallet_lookup: Dict[str, Dict] = {}
        self.device_lookup: Dict[str, Dict] = {}
        self.bank_lookup: Dict[str, Dict] = {}

        self.fiat_balance: Dict[str, float] = defaultdict(float)
        self.crypto_balance: Dict[Tuple[str, str], float] = defaultdict(float)

    # -----------------------
    # ID / sampling helpers
    # -----------------------
    def _next_id(self, prefix: str) -> str:
        self.id_counters[prefix] += 1
        return f"{prefix}_{self.id_counters[prefix]:06d}"

    def _round_asset(self, asset: str, value: float) -> float:
        precision = ASSET_CONFIG[asset]["precision"]
        return round(max(0.0, value), precision)

    def _pick_user_wallet(self, user_id: str, asset: str) -> str:
        return self.user_wallets_map[user_id][asset]

    def _asset_price(self, asset: str) -> float:
        return float(ASSET_CONFIG[asset]["price_twd"])

    def _random_ts(self, start: datetime, end: datetime, prefer_night: bool = False) -> datetime:
        if end <= start:
            return start
        total_seconds = int((end - start).total_seconds())
        if total_seconds <= 0:
            return start
        day_offset = self.rng.randint(0, max(0, (end.date() - start.date()).days))
        base_day = start + timedelta(days=day_offset)
        if prefer_night:
            hour = self.rng.randint(0, 4)
        else:
            # weighted daytime/evening hours
            bucket = self.rng.choices(
                ["morning", "day", "evening", "night"],
                weights=[0.18, 0.48, 0.27, 0.07],
                k=1,
            )[0]
            if bucket == "morning":
                hour = self.rng.randint(7, 10)
            elif bucket == "day":
                hour = self.rng.randint(11, 17)
            elif bucket == "evening":
                hour = self.rng.randint(18, 23)
            else:
                hour = self.rng.randint(0, 4)
        minute = self.rng.randint(0, 59)
        second = self.rng.randint(0, 59)
        ts = base_day.replace(hour=hour, minute=minute, second=second)
        if ts < start:
            ts = start + timedelta(minutes=self.rng.randint(0, 60))
        if ts >= end:
            ts = end - timedelta(minutes=1)
        return ts

    def _sample_home_location(self) -> Tuple[str, str]:
        if self.rng.random() < 0.88:
            return ("TW", self.rng.choice(TAIWAN_CITIES))
        return self.rng.choice(FOREIGN_LOCATIONS)

    def _sample_activity_window(self, segment: str) -> str:
        if segment == "night_owl":
            return "18:00-04:00"
        if segment in {"pro_trader", "vpn_nomad"}:
            return "08:00-23:00"
        if segment == "dormant":
            return "09:00-20:00"
        return "09:00-22:00"

    def _base_ip_prefix(self) -> Tuple[int, int]:
        return (self.rng.randint(1, 223), self.rng.randint(0, 255))

    def _generate_ip(self, user_id: str, foreign: bool = False, vpn: bool = False) -> Tuple[str, str, str]:
        state = self.user_state[user_id]
        if foreign:
            country, city = self.rng.choice(FOREIGN_LOCATIONS)
            a, b = self.rng.randint(1, 223), self.rng.randint(0, 255)
            return f"{a}.{b}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}", country, city
        if vpn:
            if self.rng.random() < 0.7:
                country, city = self.rng.choice(FOREIGN_LOCATIONS)
            else:
                country, city = ("TW", self.rng.choice(TAIWAN_CITIES))
            a, b = self.rng.randint(1, 223), self.rng.randint(0, 255)
            return f"{a}.{b}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}", country, city
        a, b = state["home_ip_prefix"]
        country = state["home_country"]
        city = state["home_city"]
        return f"{a}.{b}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}", country, city

    def _sample_segment(self, band: str) -> str:
        if band == "grey":
            return self.rng.choices(
                ["pro_trader", "vpn_nomad", "cross_border", "night_owl"],
                weights=[0.30, 0.28, 0.25, 0.17],
                k=1,
            )[0]
        return self.rng.choices(
            ["retail_investor", "salary_cycle", "dormant", "cross_border", "night_owl", "pro_trader"],
            weights=[0.40, 0.26, 0.14, 0.08, 0.07, 0.05],
            k=1,
        )[0]

    def _sample_income(self, segment: str) -> float:
        if segment == "pro_trader":
            return float(self.rng.randint(120_000, 400_000))
        if segment == "salary_cycle":
            return float(self.rng.randint(35_000, 90_000))
        if segment == "dormant":
            return float(self.rng.randint(20_000, 60_000))
        if segment == "cross_border":
            return float(self.rng.randint(50_000, 150_000))
        if segment == "vpn_nomad":
            return float(self.rng.randint(40_000, 120_000))
        if segment == "night_owl":
            return float(self.rng.randint(30_000, 90_000))
        return float(self.rng.randint(25_000, 110_000))

    def _sample_expected_volume(self, segment: str, income: float) -> float:
        multipliers = {
            "retail_investor": (0.8, 2.5),
            "salary_cycle": (0.4, 1.6),
            "pro_trader": (4.0, 15.0),
            "dormant": (0.1, 0.6),
            "cross_border": (1.2, 3.5),
            "night_owl": (0.8, 2.2),
            "vpn_nomad": (1.0, 2.8),
        }
        low, high = multipliers[segment]
        return round(income * self.rng.uniform(low, high), 2)

    def _event_choice(self, segment: str) -> str:
        items = list(SEGMENT_CONFIG[segment]["event_weights"].items())
        events = [k for k, _ in items]
        weights = [v for _, v in items]
        return self.rng.choices(events, weights=weights, k=1)[0]

    def _sample_asset(self, segment: str) -> str:
        if segment in {"cross_border", "vpn_nomad"}:
            return self.rng.choices(["USDT", "BTC", "ETH"], weights=[0.70, 0.15, 0.15], k=1)[0]
        if segment == "pro_trader":
            return self.rng.choices(["USDT", "BTC", "ETH"], weights=[0.50, 0.25, 0.25], k=1)[0]
        return self.rng.choices(["USDT", "BTC", "ETH"], weights=[0.68, 0.18, 0.14], k=1)[0]

    def _sample_twd_amount(self, user_id: str, segment: str, event_type: str, multiplier: Optional[Tuple[float, float]] = None) -> float:
        user = self.user_lookup[user_id]
        expected = user["expected_monthly_volume_twd"]
        if multiplier is None:
            multiplier = SEGMENT_CONFIG[segment]["amount_range"]
        lo, hi = multiplier
        base = expected * self.rng.uniform(lo, hi)
        if event_type in {"fiat_withdrawal", "fiat_deposit"}:
            base = max(2_000.0, base)
        elif event_type in {"trade_buy", "trade_sell"}:
            base = max(1_500.0, base)
        else:
            base = max(2_000.0, base)
        return round(min(base, expected * 0.35 + 200_000.0), 2)

    # -----------------------
    # Row creation helpers
    # -----------------------
    def _add_device(self, shared_group_id: str = "") -> str:
        device_id = self._next_id("dev")
        row = {
            "device_id": device_id,
            "device_type": self.rng.choices(["mobile", "desktop", "tablet"], weights=[0.62, 0.33, 0.05], k=1)[0],
            "os_family": self.rng.choices(["iOS", "Android", "Windows", "macOS"], weights=[0.32, 0.34, 0.23, 0.11], k=1)[0],
            "app_channel": self.rng.choices(["app", "web"], weights=[0.78, 0.22], k=1)[0],
            "device_fingerprint": self.fake.sha1(raw_output=False),
            "first_seen_at": self._random_ts(self.start_dt - timedelta(days=40), self.start_dt + timedelta(days=1)).isoformat(),
            "shared_group_id": shared_group_id,
        }
        self.devices.append(row)
        self.device_lookup[device_id] = row
        return device_id

    def _link_user_device(self, user_id: str, device_id: str, is_primary: int = 0, linked_at: Optional[datetime] = None) -> None:
        ts = linked_at or self._random_ts(self.start_dt - timedelta(days=20), self.start_dt + timedelta(days=1))
        self.user_device_links.append({
            "link_id": self._next_id("udl"),
            "user_id": user_id,
            "device_id": device_id,
            "is_primary": int(is_primary),
            "first_seen_at": ts.isoformat(),
            "last_seen_at": self.end_dt.isoformat(),
        })
        if device_id not in self.user_devices_map[user_id]:
            self.user_devices_map[user_id].append(device_id)

    def _add_bank_account(self, shared_group_id: str = "", country: str = "TW") -> str:
        bank_code, bank_name = self.rng.choice(TW_BANKS)
        bank_id = self._next_id("bank")
        row = {
            "bank_account_id": bank_id,
            "bank_code": bank_code,
            "bank_name": bank_name,
            "country": country,
            "currency": "TWD",
            "opened_at": self._random_ts(self.start_dt - timedelta(days=180), self.start_dt + timedelta(days=1)).isoformat(),
            "shared_group_id": shared_group_id,
        }
        self.bank_accounts.append(row)
        self.bank_lookup[bank_id] = row
        return bank_id

    def _link_user_bank(self, user_id: str, bank_account_id: str, is_primary: int = 0, linked_at: Optional[datetime] = None) -> None:
        ts = linked_at or self._random_ts(self.start_dt - timedelta(days=20), self.start_dt + timedelta(days=1))
        self.user_bank_links.append({
            "link_id": self._next_id("ubl"),
            "user_id": user_id,
            "bank_account_id": bank_account_id,
            "is_primary": int(is_primary),
            "linked_at": ts.isoformat(),
        })
        if bank_account_id not in self.user_banks_map[user_id]:
            self.user_banks_map[user_id].append(bank_account_id)

    def _add_wallet(self, wallet_kind: str, asset: str, network: str, user_id: Optional[str] = None,
                    cluster_id: str = "", risk_seed: str = "clean") -> str:
        wallet_id = self._next_id("wal")
        row = {
            "wallet_id": wallet_id,
            "wallet_kind": wallet_kind,
            "user_id": user_id or "",
            "cluster_id": cluster_id,
            "asset": asset,
            "network": network,
            "risk_seed": risk_seed,
            "created_at": self._random_ts(self.start_dt - timedelta(days=60), self.start_dt + timedelta(days=1)).isoformat(),
        }
        self.crypto_wallets.append(row)
        self.wallet_lookup[wallet_id] = row
        if user_id:
            self.user_wallets_map[user_id][asset] = wallet_id
        return wallet_id

    def _get_or_create_external_wallet(self, asset: str, risk_seed: str = "clean", cluster_id: Optional[str] = None) -> str:
        network = ASSET_CONFIG[asset]["network"]
        if cluster_id:
            for wallet in self.crypto_wallets:
                if wallet["wallet_kind"] == "external_cluster" and wallet["cluster_id"] == cluster_id and wallet["asset"] == asset:
                    return wallet["wallet_id"]
        cluster = cluster_id or f"cluster_{self.fake.lexify(text='????????')}"
        return self._add_wallet("external_cluster", asset, network, user_id=None, cluster_id=cluster, risk_seed=risk_seed)

    def _add_login_event(
        self,
        user_id: str,
        occurred_at: datetime,
        device_id: Optional[str] = None,
        ip_country: Optional[str] = None,
        ip_city: Optional[str] = None,
        is_vpn: Optional[int] = None,
        ip_address: Optional[str] = None,
        success: int = 1,
        scenario_id: str = "",
        tags: str = "",
    ) -> None:
        if not device_id:
            device_id = self.rng.choice(self.user_devices_map[user_id])
        if ip_address is None or ip_country is None or ip_city is None:
            vpn = bool(is_vpn) if is_vpn is not None else False
            ip_address, ip_country, ip_city = self._generate_ip(user_id, vpn=vpn)
        self.login_events.append({
            "login_id": self._next_id("login"),
            "user_id": user_id,
            "occurred_at": occurred_at.isoformat(),
            "device_id": device_id,
            "ip_address": ip_address,
            "ip_country": ip_country,
            "ip_city": ip_city,
            "is_vpn": int(is_vpn or 0),
            "is_new_device": 0,
            "is_geo_jump": 0,
            "success": int(success),
            "scenario_id": scenario_id,
            "tags": tags,
        })

    def _add_fiat_tx(self, user_id: str, occurred_at: datetime, direction: str, amount_twd: float,
                     bank_account_id: Optional[str] = None, method: str = "bank_transfer",
                     scenario_id: str = "", tags: str = "") -> None:
        bank_account_id = bank_account_id or self.rng.choice(self.user_banks_map[user_id])
        amount_twd = round(max(100.0, amount_twd), 2)
        if direction == "withdrawal" and self.fiat_balance[user_id] < amount_twd:
            amount_twd = round(max(100.0, self.fiat_balance[user_id] * 0.85), 2)
        if amount_twd <= 0:
            return
        if direction == "deposit":
            self.fiat_balance[user_id] += amount_twd
        else:
            self.fiat_balance[user_id] = max(0.0, self.fiat_balance[user_id] - amount_twd)
        self.fiat_transactions.append({
            "fiat_txn_id": self._next_id("fiat"),
            "user_id": user_id,
            "occurred_at": occurred_at.isoformat(),
            "direction": direction,
            "amount_twd": amount_twd,
            "currency": "TWD",
            "bank_account_id": bank_account_id,
            "method": method,
            "status": "completed",
            "scenario_id": scenario_id,
            "tags": tags,
        })

    def _add_trade(self, user_id: str, occurred_at: datetime, side: str, asset: str,
                   notional_twd: Optional[float] = None, quantity: Optional[float] = None,
                   order_type: str = "market", scenario_id: str = "", tags: str = "") -> None:
        price = self._asset_price(asset)
        fee_rate = 0.001
        if quantity is None and notional_twd is None:
            raise ValueError("Either notional_twd or quantity must be provided.")
        if quantity is None:
            quantity = float(notional_twd) / price
        quantity = self._round_asset(asset, quantity)
        notional_twd = round(float(quantity) * price, 2)
        fee_twd = round(notional_twd * fee_rate, 2)
        if side == "buy":
            total_cost = notional_twd + fee_twd
            if self.fiat_balance[user_id] < total_cost:
                total_cost = self.fiat_balance[user_id] * 0.95
                notional_twd = round(max(0.0, total_cost / (1 + fee_rate)), 2)
                fee_twd = round(notional_twd * fee_rate, 2)
                quantity = self._round_asset(asset, notional_twd / price)
            if quantity <= 0 or notional_twd <= 0:
                return
            self.fiat_balance[user_id] = max(0.0, self.fiat_balance[user_id] - (notional_twd + fee_twd))
            self.crypto_balance[(user_id, asset)] += quantity
        else:
            current_qty = self.crypto_balance[(user_id, asset)]
            if current_qty <= 0:
                return
            if quantity > current_qty:
                quantity = self._round_asset(asset, current_qty * 0.98)
                notional_twd = round(quantity * price, 2)
                fee_twd = round(notional_twd * fee_rate, 2)
            if quantity <= 0 or notional_twd <= 0:
                return
            self.crypto_balance[(user_id, asset)] = max(0.0, current_qty - quantity)
            self.fiat_balance[user_id] += max(0.0, notional_twd - fee_twd)
        self.trade_orders.append({
            "trade_id": self._next_id("trade"),
            "user_id": user_id,
            "occurred_at": occurred_at.isoformat(),
            "side": side,
            "base_asset": asset,
            "quote_asset": "TWD",
            "price_twd": price,
            "quantity": quantity,
            "notional_twd": notional_twd,
            "fee_twd": fee_twd,
            "order_type": order_type,
            "status": "filled",
            "scenario_id": scenario_id,
            "tags": tags,
        })

    def _add_crypto_tx(self, user_id: str, occurred_at: datetime, direction: str, asset: str, amount_asset: float,
                       counterparty_wallet_id: str, wallet_id: Optional[str] = None,
                       scenario_id: str = "", tags: str = "") -> None:
        wallet_id = wallet_id or self._pick_user_wallet(user_id, asset)
        amount_asset = self._round_asset(asset, amount_asset)
        if amount_asset <= 0:
            return
        if direction == "withdrawal":
            available = self.crypto_balance[(user_id, asset)]
            if available <= 0:
                return
            if amount_asset > available:
                amount_asset = self._round_asset(asset, available * 0.98)
            if amount_asset <= 0:
                return
            self.crypto_balance[(user_id, asset)] = max(0.0, available - amount_asset)
        else:
            self.crypto_balance[(user_id, asset)] += amount_asset
        wallet_meta = self.wallet_lookup[counterparty_wallet_id]
        self.crypto_transactions.append({
            "crypto_txn_id": self._next_id("ctx"),
            "user_id": user_id,
            "occurred_at": occurred_at.isoformat(),
            "direction": direction,
            "asset": asset,
            "network": ASSET_CONFIG[asset]["network"],
            "wallet_id": wallet_id,
            "counterparty_wallet_id": counterparty_wallet_id,
            "amount_asset": amount_asset,
            "amount_twd_equiv": round(amount_asset * self._asset_price(asset), 2),
            "tx_hash": self.fake.sha1(raw_output=False),
            "status": "completed",
            "onchain_risk_seed": wallet_meta["risk_seed"],
            "scenario_id": scenario_id,
            "tags": tags,
        })

    def _record_scenario(self, scenario_type: str, start_at: datetime, end_at: datetime, description: str) -> str:
        scenario_id = self._next_id("scn")
        self.scenarios.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
            "description": description,
        })
        return scenario_id

    def _add_scenario_member(self, scenario_id: str, entity_type: str, entity_id: str, role: str) -> None:
        self.scenario_members.append({
            "scenario_id": scenario_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "role": role,
        })

    def _mark_user(self, user_id: str, scenario_type: str, role: str,
                   evidence_tags: List[str], observed_prob: Optional[float] = None) -> None:
        state = self.user_state[user_id]
        state["hidden_label"] = 1
        state["scenario_types"].add(scenario_type)
        state["roles"].add(role)
        state["evidence_tags"].update(evidence_tags)
        if observed_prob is not None:
            state["observed_prob"] = max(state.get("observed_prob", 0.0), observed_prob)

    # -----------------------
    # Generation stages
    # -----------------------
    def create_users(self) -> None:
        grey_count = int(self.cfg.n_users * self.cfg.grey_user_ratio)
        grey_indices = set(self.rng.sample(range(self.cfg.n_users), grey_count))
        for idx in range(self.cfg.n_users):
            band = "grey" if idx in grey_indices else "clean"
            segment = self._sample_segment(band)
            income = self._sample_income(segment)
            expected_volume = self._sample_expected_volume(segment, income)
            kyc_level = self.rng.choices(["basic", "standard", "enhanced"], weights=[0.18, 0.57, 0.25], k=1)[0]
            residence_country, residence_city = self._sample_home_location()
            created_at = self._random_ts(self.start_dt - timedelta(days=730), self.start_dt)
            user_id = self._next_id("usr")
            row = {
                "user_id": user_id,
                "created_at": created_at.isoformat(),
                "segment": segment,
                "risk_seed_band": band,
                "kyc_level": kyc_level,
                "occupation": self.rng.choice(OCCUPATIONS),
                "monthly_income_twd": round(income, 2),
                "expected_monthly_volume_twd": round(expected_volume, 2),
                "declared_source_of_funds": self.rng.choice(SOURCE_OF_FUNDS),
                "residence_country": residence_country,
                "residence_city": residence_city,
                "nationality": residence_country,
                "activity_window": self._sample_activity_window(segment),
                "hidden_suspicious_label": 0,
                "observed_blacklist_label": 0,
                "scenario_types": "",
                "evidence_tags": "",
            }
            self.users.append(row)
            self.user_lookup[user_id] = row
            self.user_state[user_id] = {
                "home_country": residence_country,
                "home_city": residence_city,
                "home_ip_prefix": self._base_ip_prefix(),
                "hidden_label": 0,
                "scenario_types": set(),
                "roles": set(),
                "evidence_tags": set(),
                "observed_prob": 0.0,
            }

    def create_static_entities(self) -> None:
        for user in self.users:
            user_id = user["user_id"]
            segment = user["segment"]
            dev_lo, dev_hi = SEGMENT_CONFIG[segment]["devices"]
            bank_lo, bank_hi = SEGMENT_CONFIG[segment]["banks"]
            n_devices = self.rng.randint(dev_lo, dev_hi)
            n_banks = self.rng.randint(bank_lo, bank_hi)

            for i in range(n_devices):
                device_id = self._add_device(shared_group_id="")
                self._link_user_device(user_id, device_id, is_primary=int(i == 0))

            for i in range(n_banks):
                bank_id = self._add_bank_account(country=user["residence_country"])
                self._link_user_bank(user_id, bank_id, is_primary=int(i == 0))

            for asset, meta in ASSET_CONFIG.items():
                self._add_wallet("user_wallet", asset=asset, network=meta["network"], user_id=user_id, risk_seed="clean")

    def generate_login_events(self) -> None:
        for user in self.users:
            user_id = user["user_id"]
            segment = user["segment"]
            cfg = SEGMENT_CONFIG[segment]
            approx_events = int(max(1, self.np_rng.poisson(cfg["login_daily_rate"] * self.cfg.days)))
            for _ in range(approx_events):
                prefer_night = self.rng.random() < cfg["night_prob"]
                ts = self._random_ts(self.start_dt, self.end_dt, prefer_night=prefer_night)
                device_id = self.rng.choice(self.user_devices_map[user_id])
                use_vpn = self.rng.random() < cfg["vpn_prob"]
                foreign = False
                if segment == "cross_border" and self.rng.random() < 0.25:
                    foreign = True
                    use_vpn = False
                ip, country, city = self._generate_ip(user_id, foreign=foreign, vpn=use_vpn)
                self._add_login_event(
                    user_id=user_id,
                    occurred_at=ts,
                    device_id=device_id,
                    ip_country=country,
                    ip_city=city,
                    ip_address=ip,
                    is_vpn=int(use_vpn),
                )

    def generate_base_transactions(self) -> None:
        end_dt = self.base_end_dt
        for user in self.users:
            user_id = user["user_id"]
            segment = user["segment"]
            seg_cfg = SEGMENT_CONFIG[segment]
            n_actions = int(max(1, self.np_rng.poisson(seg_cfg["daily_action_lambda"] * max(1, (self.base_end_dt - self.start_dt).days))))
            planned = []
            for _ in range(n_actions):
                ts = self._random_ts(self.start_dt, end_dt, prefer_night=self.rng.random() < seg_cfg["night_prob"])
                planned.append((ts, self._event_choice(segment)))
            for ts, event_type in sorted(planned, key=lambda x: x[0]):
                if event_type == "fiat_deposit":
                    amount = self._sample_twd_amount(user_id, segment, event_type)
                    self._add_fiat_tx(user_id, ts, "deposit", amount, tags="base_flow")
                elif event_type == "fiat_withdrawal":
                    amount = self._sample_twd_amount(user_id, segment, event_type)
                    self._add_fiat_tx(user_id, ts, "withdrawal", amount, tags="base_flow")
                elif event_type == "trade_buy":
                    amount = self._sample_twd_amount(user_id, segment, event_type)
                    if self.fiat_balance[user_id] < amount * 1.02:
                        pre_ts = ts - timedelta(minutes=self.rng.randint(15, 180))
                        if pre_ts < self.start_dt:
                            pre_ts = self.start_dt + timedelta(minutes=10)
                        self._add_fiat_tx(user_id, pre_ts, "deposit", amount * self.rng.uniform(1.05, 1.50), tags="topup_for_trade")
                    asset = self._sample_asset(segment)
                    self._add_trade(user_id, ts, "buy", asset, notional_twd=amount, tags="base_trade")
                elif event_type == "trade_sell":
                    assets = [a for a in ASSET_CONFIG if self.crypto_balance[(user_id, a)] > 0]
                    if not assets:
                        continue
                    asset = self.rng.choice(assets)
                    current_qty = self.crypto_balance[(user_id, asset)]
                    qty = self._round_asset(asset, current_qty * self.rng.uniform(0.15, 0.65))
                    self._add_trade(user_id, ts, "sell", asset, quantity=qty, tags="base_trade")
                elif event_type == "crypto_deposit":
                    asset = self._sample_asset(segment)
                    amount_twd = self._sample_twd_amount(user_id, segment, event_type)
                    qty = self._round_asset(asset, amount_twd / self._asset_price(asset))
                    cp_wallet = self._get_or_create_external_wallet(asset, risk_seed="clean")
                    self._add_crypto_tx(user_id, ts, "deposit", asset, qty, cp_wallet, tags="base_onchain")
                elif event_type == "crypto_withdrawal":
                    asset_candidates = [a for a in ASSET_CONFIG if self.crypto_balance[(user_id, a)] > 0]
                    if not asset_candidates:
                        continue
                    asset = self.rng.choice(asset_candidates)
                    qty = self._round_asset(asset, self.crypto_balance[(user_id, asset)] * self.rng.uniform(0.1, 0.7))
                    cp_wallet = self._get_or_create_external_wallet(asset, risk_seed="clean")
                    self._add_crypto_tx(user_id, ts, "withdrawal", asset, qty, cp_wallet, tags="base_onchain")

    # -----------------------
    # Suspicious scenario injection
    # -----------------------
    def _suspicious_window(self) -> Tuple[datetime, datetime]:
        start = self.base_end_dt
        end = self.end_dt
        return start, end

    def _take_available_users(self, available: List[str], count: int) -> List[str]:
        count = min(count, len(available))
        chosen = self.rng.sample(available, count)
        chosen_set = set(chosen)
        available[:] = [u for u in available if u not in chosen_set]
        return chosen

    def inject_suspicious_scenarios(self) -> None:
        target = max(8, int(self.cfg.n_users * self.cfg.suspicious_user_ratio))
        clean_or_grey_users = [u["user_id"] for u in self.users]
        available = clean_or_grey_users[:]
        injected = 0

        # More weight on low-income / retail profiles by prioritizing them in selection pool
        available.sort(key=lambda uid: (
            self.user_lookup[uid]["monthly_income_twd"],
            self.user_lookup[uid]["segment"] not in {"retail_investor", "salary_cycle", "dormant"},
            uid,
        ))

        while injected < target and available:
            remaining = target - injected
            scenario_type = self.rng.choices(
                ["mule_quick_out", "fan_in_hub", "shared_device_ring", "blacklist_2hop_chain"],
                weights=[0.35, 0.27, 0.20, 0.18],
                k=1,
            )[0]
            if scenario_type == "mule_quick_out" or remaining <= 2:
                users = self._take_available_users(available, 1)
                if users:
                    self.inject_mule_quick_out(users[0])
                    injected += 1
            elif scenario_type == "fan_in_hub":
                group_size = min(remaining, self.rng.randint(4, 6))
                users = self._take_available_users(available, group_size)
                if len(users) >= 4:
                    hub = users[0]
                    feeders = users[1:]
                    self.inject_fan_in_hub(hub, feeders)
                    injected += len(users)
                else:
                    for uid in users:
                        self.inject_mule_quick_out(uid)
                        injected += 1
            elif scenario_type == "shared_device_ring":
                group_size = min(remaining, self.rng.randint(3, 5))
                users = self._take_available_users(available, group_size)
                if len(users) >= 3:
                    self.inject_shared_device_ring(users)
                    injected += len(users)
                else:
                    for uid in users:
                        self.inject_mule_quick_out(uid)
                        injected += 1
            else:
                group_size = min(remaining, self.rng.randint(3, 4))
                users = self._take_available_users(available, group_size)
                if len(users) >= 3:
                    controller = users[0]
                    neighbors = users[1:]
                    self.inject_blacklist_2hop_chain(controller, neighbors)
                    injected += len(users)
                else:
                    for uid in users:
                        self.inject_mule_quick_out(uid)
                        injected += 1

    def inject_mule_quick_out(self, user_id: str) -> None:
        win_start, win_end = self._suspicious_window()
        t1 = self._random_ts(win_start, win_end - timedelta(hours=6))
        t2 = t1 + timedelta(minutes=self.rng.randint(8, 120))
        t3 = t2 + timedelta(minutes=self.rng.randint(6, 60))
        t4 = t3 + timedelta(hours=self.rng.randint(4, 18))

        scenario_id = self._record_scenario(
            "mule_quick_out", t1, t4,
            "低 KYC / 低收入帳戶在短時間內法幣入金、買幣、鏈上提領，模擬快進快出與短滯留時間。"
        )
        self._add_scenario_member(scenario_id, "user", user_id, "mule")

        # Force KYC-volume mismatch
        self.user_lookup[user_id]["monthly_income_twd"] = round(min(self.user_lookup[user_id]["monthly_income_twd"], self.rng.uniform(18_000, 45_000)), 2)
        self.user_lookup[user_id]["expected_monthly_volume_twd"] = round(
            max(self.user_lookup[user_id]["expected_monthly_volume_twd"], self.rng.uniform(180_000, 650_000)), 2
        )

        amount = round(self.rng.uniform(80_000, 380_000), 2)
        self._add_fiat_tx(user_id, t1, "deposit", amount, scenario_id=scenario_id, tags="quick_in|kyc_mismatch")
        self._add_trade(user_id, t2, "buy", "USDT", notional_twd=amount * self.rng.uniform(0.94, 0.985),
                        scenario_id=scenario_id, tags="convert_to_stablecoin|short_dwell")
        suspicious_cluster = f"scn_{scenario_id}_cashout"
        cp_wallet = self._get_or_create_external_wallet("USDT", risk_seed="suspicious_neighbor", cluster_id=suspicious_cluster)
        self._add_scenario_member(scenario_id, "wallet", cp_wallet, "cashout_cluster")

        qty = self._round_asset("USDT", self.crypto_balance[(user_id, "USDT")] * self.rng.uniform(0.92, 0.995))
        self._add_crypto_tx(user_id, t3, "withdrawal", "USDT", qty, cp_wallet, scenario_id=scenario_id,
                            tags="quick_out|short_dwell|cashout")

        # Optional late-night login and another unusual withdrawal
        shared_device = self.rng.choice(self.user_devices_map[user_id])
        night_ts = t4.replace(hour=self.rng.randint(1, 3), minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
        ip, country, city = self._generate_ip(user_id, foreign=True)
        self._add_login_event(user_id, night_ts, device_id=shared_device, ip_address=ip, ip_country=country, ip_city=city,
                              is_vpn=0, scenario_id=scenario_id, tags="late_night|foreign_ip")
        self._mark_user(
            user_id, "mule_quick_out", "mule",
            ["quick_in_out", "short_dwell_time", "kyc_volume_mismatch", "late_night_activity"], observed_prob=0.28
        )

    def inject_fan_in_hub(self, hub_user_id: str, feeder_user_ids: List[str]) -> None:
        win_start, win_end = self._suspicious_window()
        t0 = self._random_ts(win_start, win_end - timedelta(hours=12))
        scenario_id = self._record_scenario(
            "fan_in_hub", t0, t0 + timedelta(hours=20),
            "多個 feeder 入金買幣後提至同一 external cluster，再由 hub 重新入金、賣出並法幣出金。"
        )
        cp_cluster_id = f"scn_{scenario_id}_shared_cluster"
        shared_wallet = self._get_or_create_external_wallet("USDT", risk_seed="suspicious_neighbor", cluster_id=cp_cluster_id)
        self._add_scenario_member(scenario_id, "wallet", shared_wallet, "shared_cluster")
        self._add_scenario_member(scenario_id, "user", hub_user_id, "hub")

        total_qty = 0.0
        for idx, feeder in enumerate(feeder_user_ids):
            self._add_scenario_member(scenario_id, "user", feeder, f"feeder_{idx+1}")
            base_ts = t0 + timedelta(minutes=self.rng.randint(0, 240))
            amount = round(self.rng.uniform(25_000, 95_000), 2)
            self.user_lookup[feeder]["monthly_income_twd"] = round(min(self.user_lookup[feeder]["monthly_income_twd"], self.rng.uniform(20_000, 50_000)), 2)
            self._add_fiat_tx(feeder, base_ts, "deposit", amount, scenario_id=scenario_id, tags="fan_in_seed")
            buy_ts = base_ts + timedelta(minutes=self.rng.randint(15, 120))
            self._add_trade(feeder, buy_ts, "buy", "USDT", notional_twd=amount * self.rng.uniform(0.95, 0.985),
                            scenario_id=scenario_id, tags="fan_in_convert")
            wd_ts = buy_ts + timedelta(minutes=self.rng.randint(10, 90))
            qty = self._round_asset("USDT", self.crypto_balance[(feeder, "USDT")] * self.rng.uniform(0.90, 0.99))
            total_qty += qty
            self._add_crypto_tx(feeder, wd_ts, "withdrawal", "USDT", qty, shared_wallet, scenario_id=scenario_id,
                                tags="fan_out_to_shared_cluster|short_dwell")
            self._mark_user(feeder, "fan_in_hub", "feeder", ["fan_out_to_shared_cluster", "short_dwell_time", "kyc_volume_mismatch"], observed_prob=0.15)

        # Hub side: same cluster sends back to hub later
        hub_dep_ts = t0 + timedelta(hours=self.rng.randint(8, 16))
        chunks = max(2, min(4, len(feeder_user_ids)))
        for _ in range(chunks):
            dep_qty = self._round_asset("USDT", total_qty / chunks * self.rng.uniform(0.85, 1.05))
            self._add_crypto_tx(hub_user_id, hub_dep_ts + timedelta(minutes=self.rng.randint(0, 180)), "deposit", "USDT",
                                dep_qty, shared_wallet, scenario_id=scenario_id, tags="fan_in_from_shared_cluster")

        sell_ts = hub_dep_ts + timedelta(minutes=self.rng.randint(25, 180))
        sell_qty = self._round_asset("USDT", self.crypto_balance[(hub_user_id, "USDT")] * self.rng.uniform(0.75, 0.95))
        self._add_trade(hub_user_id, sell_ts, "sell", "USDT", quantity=sell_qty, scenario_id=scenario_id, tags="hub_sell")
        wd_ts = sell_ts + timedelta(minutes=self.rng.randint(20, 240))
        fiat_out = min(self.fiat_balance[hub_user_id] * 0.90, round(total_qty * self._asset_price("USDT") * self.rng.uniform(0.55, 0.85), 2))
        self._add_fiat_tx(hub_user_id, wd_ts, "withdrawal", fiat_out, scenario_id=scenario_id, tags="hub_cashout")

        # Hub may login at night before cash-out
        shared_device = self.rng.choice(self.user_devices_map[hub_user_id])
        night_ts = wd_ts.replace(hour=self.rng.randint(1, 3), minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
        ip, country, city = self._generate_ip(hub_user_id, vpn=True)
        self._add_login_event(hub_user_id, night_ts, device_id=shared_device, ip_address=ip, ip_country=country, ip_city=city,
                              is_vpn=1, scenario_id=scenario_id, tags="late_night|vpn_before_cashout")
        self._mark_user(hub_user_id, "fan_in_hub", "hub", ["fan_in_cluster", "aggregation_hub", "late_night_cashout"], observed_prob=0.82)

    def inject_shared_device_ring(self, member_user_ids: List[str]) -> None:
        win_start, win_end = self._suspicious_window()
        start_ts = self._random_ts(win_start, win_end - timedelta(hours=10))
        end_ts = start_ts + timedelta(hours=18)
        scenario_id = self._record_scenario(
            "shared_device_ring", start_ts, end_ts,
            "多個帳戶共用裝置與銀行帳號，並出現相似的快進快出與夜間操作。"
        )
        shared_device_id = self._add_device(shared_group_id=scenario_id)
        shared_bank_id = self._add_bank_account(shared_group_id=scenario_id)
        self._add_scenario_member(scenario_id, "device", shared_device_id, "shared_device")
        self._add_scenario_member(scenario_id, "bank_account", shared_bank_id, "shared_bank")
        shared_ip, ip_country, ip_city = self._generate_ip(member_user_ids[0], vpn=False)

        for idx, user_id in enumerate(member_user_ids):
            self._link_user_device(user_id, shared_device_id, is_primary=0, linked_at=start_ts - timedelta(days=1))
            self._link_user_bank(user_id, shared_bank_id, is_primary=0, linked_at=start_ts - timedelta(days=1))
            self._add_scenario_member(scenario_id, "user", user_id, f"ring_member_{idx+1}")

            # Coordinated logins from same device/IP
            login_ts = start_ts + timedelta(minutes=idx * self.rng.randint(10, 40))
            self._add_login_event(
                user_id, login_ts, device_id=shared_device_id, ip_address=shared_ip, ip_country=ip_country, ip_city=ip_city,
                is_vpn=0, scenario_id=scenario_id, tags="shared_device|shared_ip"
            )

            amount = round(self.rng.uniform(35_000, 120_000), 2)
            deposit_ts = login_ts + timedelta(minutes=self.rng.randint(5, 60))
            self._add_fiat_tx(user_id, deposit_ts, "deposit", amount, bank_account_id=shared_bank_id,
                              scenario_id=scenario_id, tags="shared_bank|ring_deposit")
            buy_ts = deposit_ts + timedelta(minutes=self.rng.randint(10, 80))
            self._add_trade(user_id, buy_ts, "buy", "USDT", notional_twd=amount * self.rng.uniform(0.95, 0.985),
                            scenario_id=scenario_id, tags="shared_device_ring_buy")
            cashout_wallet = self._get_or_create_external_wallet("USDT", risk_seed="suspicious_neighbor",
                                                                 cluster_id=f"ring_{scenario_id}_cluster")
            qty = self._round_asset("USDT", self.crypto_balance[(user_id, "USDT")] * self.rng.uniform(0.92, 0.995))
            wd_ts = buy_ts + timedelta(minutes=self.rng.randint(10, 70))
            self._add_crypto_tx(user_id, wd_ts, "withdrawal", "USDT", qty, cashout_wallet, scenario_id=scenario_id,
                                tags="shared_device_ring_out|quick_out")
            # One or two members do deep-night cashout
            if idx == 0 or (idx == len(member_user_ids) - 1 and self.rng.random() < 0.5):
                late_ts = wd_ts.replace(hour=self.rng.randint(1, 3), minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
                foreign_ip, foreign_country, foreign_city = self._generate_ip(user_id, foreign=True)
                self._add_login_event(
                    user_id, late_ts, device_id=shared_device_id, ip_address=foreign_ip, ip_country=foreign_country,
                    ip_city=foreign_city, is_vpn=0, scenario_id=scenario_id, tags="late_night|foreign_ip"
                )
            self._mark_user(
                user_id, "shared_device_ring", "ring_member",
                ["shared_device", "shared_bank_account", "shared_ip", "quick_in_out"], observed_prob=0.32
            )

    def inject_blacklist_2hop_chain(self, controller_user_id: str, neighbor_user_ids: List[str]) -> None:
        win_start, win_end = self._suspicious_window()
        start_ts = self._random_ts(win_start, win_end - timedelta(hours=12))
        end_ts = start_ts + timedelta(hours=24)
        scenario_id = self._record_scenario(
            "blacklist_2hop_chain", start_ts, end_ts,
            "黑名單 seed wallet -> controller -> shared mid wallet -> neighbors，形成 1-hop / 2-hop 鏈上關聯。"
        )

        seed_wallet = self._get_or_create_external_wallet("USDT", risk_seed="blacklist_seed", cluster_id=f"bl_seed_{scenario_id}")
        mid_wallet = self._get_or_create_external_wallet("USDT", risk_seed="suspicious_neighbor", cluster_id=f"mid_{scenario_id}")
        self._add_scenario_member(scenario_id, "wallet", seed_wallet, "blacklist_seed")
        self._add_scenario_member(scenario_id, "wallet", mid_wallet, "mid_wallet")
        self._add_scenario_member(scenario_id, "user", controller_user_id, "controller")

        dep_qty = self._round_asset("USDT", self.rng.uniform(4_000, 16_000))
        self._add_crypto_tx(controller_user_id, start_ts, "deposit", "USDT", dep_qty, seed_wallet, scenario_id=scenario_id,
                            tags="blacklist_1hop_seed")
        sell_qty = self._round_asset("USDT", dep_qty * self.rng.uniform(0.22, 0.48))
        self._add_trade(controller_user_id, start_ts + timedelta(minutes=self.rng.randint(15, 150)), "sell", "USDT",
                        quantity=sell_qty, scenario_id=scenario_id, tags="seed_sell")
        rem_qty = self._round_asset("USDT", self.crypto_balance[(controller_user_id, "USDT")] * self.rng.uniform(0.45, 0.80))
        out_ts = start_ts + timedelta(minutes=self.rng.randint(90, 240))
        self._add_crypto_tx(controller_user_id, out_ts, "withdrawal", "USDT", rem_qty, mid_wallet, scenario_id=scenario_id,
                            tags="blacklist_to_mid_wallet")
        self._mark_user(controller_user_id, "blacklist_2hop_chain", "controller",
                        ["direct_1hop_blacklist_seed", "mid_wallet_distribution"], observed_prob=0.92)

        per_neighbor = max(100.0, rem_qty / max(1, len(neighbor_user_ids)))
        for idx, user_id in enumerate(neighbor_user_ids):
            self._add_scenario_member(scenario_id, "user", user_id, f"neighbor_{idx+1}")
            dep_ts = out_ts + timedelta(hours=self.rng.randint(1, 12), minutes=self.rng.randint(0, 59))
            qty = self._round_asset("USDT", per_neighbor * self.rng.uniform(0.75, 1.10))
            self._add_crypto_tx(user_id, dep_ts, "deposit", "USDT", qty, mid_wallet, scenario_id=scenario_id,
                                tags="2hop_from_blacklist_seed")
            sell_ts = dep_ts + timedelta(minutes=self.rng.randint(10, 180))
            self._add_trade(user_id, sell_ts, "sell", "USDT", quantity=self._round_asset("USDT", qty * self.rng.uniform(0.88, 0.98)),
                            scenario_id=scenario_id, tags="neighbor_sell")
            # Late-night fiat withdrawal after sell
            night_ts = sell_ts.replace(hour=self.rng.randint(1, 4), minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
            self._add_fiat_tx(user_id, night_ts, "withdrawal", self.fiat_balance[user_id] * self.rng.uniform(0.60, 0.92),
                              scenario_id=scenario_id, tags="late_night_cashout")
            new_dev = self._add_device(shared_group_id=scenario_id)
            self._link_user_device(user_id, new_dev, is_primary=0, linked_at=dep_ts - timedelta(hours=2))
            ip, country, city = self._generate_ip(user_id, foreign=True)
            self._add_login_event(user_id, night_ts - timedelta(minutes=10), device_id=new_dev, ip_address=ip, ip_country=country,
                                  ip_city=city, is_vpn=0, scenario_id=scenario_id, tags="new_device|foreign_ip")
            self._mark_user(
                user_id, "blacklist_2hop_chain", "neighbor",
                ["two_hop_blacklist_relation", "late_night_withdrawal", "new_device_before_cashout"], observed_prob=0.10
            )

    def finalize_labels(self) -> None:
        hidden_users = []
        for user in self.users:
            state = self.user_state[user["user_id"]]
            user["hidden_suspicious_label"] = int(state["hidden_label"])
            if state["hidden_label"]:
                p = max(state["observed_prob"], self.cfg.observed_blacklist_ratio * 0.5)
                user["observed_blacklist_label"] = int(self.rng.random() < p)
                hidden_users.append(user)
            else:
                user["observed_blacklist_label"] = 0
            user["scenario_types"] = "|".join(sorted(state["scenario_types"]))
            user["evidence_tags"] = "|".join(sorted(state["evidence_tags"]))

        # Keep the "partially labeled" property, but avoid pathological runs with too few observed labels.
        target_observed = max(1, int(round(len(hidden_users) * self.cfg.observed_blacklist_ratio))) if hidden_users else 0
        current_observed = sum(int(u["observed_blacklist_label"]) for u in hidden_users)
        if current_observed < target_observed:
            def rank_key(user_row: Dict) -> Tuple[float, int]:
                state = self.user_state[user_row["user_id"]]
                role_bonus = 0
                if "controller" in state["roles"]:
                    role_bonus += 3
                if "hub" in state["roles"]:
                    role_bonus += 2
                if "mule" in state["roles"]:
                    role_bonus += 1
                return (state.get("observed_prob", 0.0), role_bonus)

            candidates = [u for u in hidden_users if not u["observed_blacklist_label"]]
            candidates.sort(key=rank_key, reverse=True)
            needed = target_observed - current_observed
            for user in candidates[:needed]:
                user["observed_blacklist_label"] = 1

    def finalize_login_flags(self) -> None:
        events = sorted(self.login_events, key=lambda x: (x["user_id"], x["occurred_at"], x["login_id"]))
        seen_devices = defaultdict(set)
        prev_geo = {}
        for row in events:
            uid = row["user_id"]
            row["is_new_device"] = int(row["device_id"] not in seen_devices[uid])
            seen_devices[uid].add(row["device_id"])
            geo = (row["ip_country"], row["ip_city"])
            if uid in prev_geo:
                row["is_geo_jump"] = int(geo != prev_geo[uid])
            else:
                row["is_geo_jump"] = 0
            prev_geo[uid] = geo
        self.login_events = events

    def build_entity_edges(self) -> None:
        for row in self.user_device_links:
            self.entity_edges.append({
                "edge_id": self._next_id("edge"),
                "occurred_at": row["first_seen_at"],
                "src_type": "user",
                "src_id": row["user_id"],
                "relation_type": "USES_DEVICE",
                "dst_type": "device",
                "dst_id": row["device_id"],
                "amount_twd_equiv": None,
                "asset": "",
                "scenario_id": "",
            })
        for row in self.user_bank_links:
            self.entity_edges.append({
                "edge_id": self._next_id("edge"),
                "occurred_at": row["linked_at"],
                "src_type": "user",
                "src_id": row["user_id"],
                "relation_type": "USES_BANK_ACCOUNT",
                "dst_type": "bank_account",
                "dst_id": row["bank_account_id"],
                "amount_twd_equiv": None,
                "asset": "",
                "scenario_id": "",
            })
        for row in self.login_events:
            ip_entity = f"ip::{row['ip_address']}"
            self.entity_edges.append({
                "edge_id": self._next_id("edge"),
                "occurred_at": row["occurred_at"],
                "src_type": "user",
                "src_id": row["user_id"],
                "relation_type": "LOGGED_IN_FROM_IP",
                "dst_type": "ip",
                "dst_id": ip_entity,
                "amount_twd_equiv": None,
                "asset": "",
                "scenario_id": row["scenario_id"],
            })
        for row in self.crypto_transactions:
            if row["direction"] == "deposit":
                src_type, src_id = "wallet", row["counterparty_wallet_id"]
                dst_type, dst_id = "user", row["user_id"]
                relation = "CRYPTO_DEPOSIT_TO"
            else:
                src_type, src_id = "user", row["user_id"]
                dst_type, dst_id = "wallet", row["counterparty_wallet_id"]
                relation = "CRYPTO_WITHDRAW_TO"
            self.entity_edges.append({
                "edge_id": self._next_id("edge"),
                "occurred_at": row["occurred_at"],
                "src_type": src_type,
                "src_id": src_id,
                "relation_type": relation,
                "dst_type": dst_type,
                "dst_id": dst_id,
                "amount_twd_equiv": row["amount_twd_equiv"],
                "asset": row["asset"],
                "scenario_id": row["scenario_id"],
            })
        for row in self.fiat_transactions:
            if row["direction"] == "deposit":
                src_type, src_id = "bank_account", row["bank_account_id"]
                dst_type, dst_id = "user", row["user_id"]
                relation = "FIAT_DEPOSIT_TO"
            else:
                src_type, src_id = "user", row["user_id"]
                dst_type, dst_id = "bank_account", row["bank_account_id"]
                relation = "FIAT_WITHDRAW_TO"
            self.entity_edges.append({
                "edge_id": self._next_id("edge"),
                "occurred_at": row["occurred_at"],
                "src_type": src_type,
                "src_id": src_id,
                "relation_type": relation,
                "dst_type": dst_type,
                "dst_id": dst_id,
                "amount_twd_equiv": row["amount_twd"],
                "asset": "TWD",
                "scenario_id": row["scenario_id"],
            })

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        tables = {
            "users": pd.DataFrame(self.users).sort_values(["user_id"]).reset_index(drop=True),
            "devices": pd.DataFrame(self.devices).sort_values(["device_id"]).reset_index(drop=True),
            "user_device_links": pd.DataFrame(self.user_device_links).sort_values(["user_id", "device_id"]).reset_index(drop=True),
            "bank_accounts": pd.DataFrame(self.bank_accounts).sort_values(["bank_account_id"]).reset_index(drop=True),
            "user_bank_links": pd.DataFrame(self.user_bank_links).sort_values(["user_id", "bank_account_id"]).reset_index(drop=True),
            "crypto_wallets": pd.DataFrame(self.crypto_wallets).sort_values(["wallet_id"]).reset_index(drop=True),
            "login_events": pd.DataFrame(self.login_events).sort_values(["occurred_at", "login_id"]).reset_index(drop=True),
            "fiat_transactions": pd.DataFrame(self.fiat_transactions).sort_values(["occurred_at", "fiat_txn_id"]).reset_index(drop=True),
            "trade_orders": pd.DataFrame(self.trade_orders).sort_values(["occurred_at", "trade_id"]).reset_index(drop=True),
            "crypto_transactions": pd.DataFrame(self.crypto_transactions).sort_values(["occurred_at", "crypto_txn_id"]).reset_index(drop=True),
            "scenarios": pd.DataFrame(self.scenarios).sort_values(["scenario_id"]).reset_index(drop=True),
            "scenario_members": pd.DataFrame(self.scenario_members).sort_values(["scenario_id", "entity_type", "entity_id"]).reset_index(drop=True),
            "entity_edges": pd.DataFrame(self.entity_edges).sort_values(["occurred_at", "edge_id"]).reset_index(drop=True),
        }
        tables["manifest"] = pd.DataFrame(
            [{"table_name": name, "row_count": int(len(df))} for name, df in tables.items()]
        ).sort_values(["table_name"]).reset_index(drop=True)
        return tables

    def run(self) -> Dict[str, pd.DataFrame]:
        self.create_users()
        self.create_static_entities()
        self.generate_login_events()
        self.generate_base_transactions()
        self.inject_suspicious_scenarios()
        self.finalize_labels()
        self.finalize_login_flags()
        self.build_entity_edges()
        return self.to_dataframes()

    def write_outputs(self, output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
        with open(output_dir / "schema.json", "w", encoding="utf-8") as f:
            json.dump(SCHEMA, f, ensure_ascii=False, indent=2)
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="BitoGuard pseudo-data transaction simulator")
    parser.add_argument("--n-users", type=int, default=1200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="bitoguard_sim_output")
    args = parser.parse_args()

    cfg = SimulationConfig(
        n_users=args.n_users,
        days=args.days,
        start_date=args.start_date,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    sim = TransactionSimulator(cfg)
    tables = sim.run()
    sim.write_outputs(Path(args.output_dir), tables)

    summary = tables["manifest"].to_dict(orient="records")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
