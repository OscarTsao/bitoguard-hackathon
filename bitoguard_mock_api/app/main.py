from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .data import DataStore, InvalidFilterValue


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: list[dict[str, Any]] | None = None


class UserRecord(BaseModel):
    user_id: str
    created_at: str
    segment: str | None
    kyc_level: str | None
    occupation: str | None
    monthly_income_twd: float | None
    expected_monthly_volume_twd: float | None
    declared_source_of_funds: str | None
    residence_country: str | None
    residence_city: str | None
    nationality: str | None
    activity_window: str | None


class LoginEventRecord(BaseModel):
    login_id: str
    user_id: str
    occurred_at: str
    device_id: str
    ip_address: str | None
    ip_country: str | None
    ip_city: str | None
    is_vpn: bool | None
    is_new_device: bool | None
    is_geo_jump: bool | None
    success: bool | None


class FiatTransactionRecord(BaseModel):
    fiat_txn_id: str
    user_id: str
    occurred_at: str
    direction: str | None
    amount_twd: float | None
    currency: str | None
    bank_account_id: str | None
    method: str | None
    status: str | None


class TradeOrderRecord(BaseModel):
    trade_id: str
    user_id: str
    occurred_at: str
    side: str | None
    base_asset: str | None
    quote_asset: str | None
    price_twd: float | None
    quantity: float | None
    notional_twd: float | None
    fee_twd: float | None
    order_type: str | None
    status: str | None


class CryptoTransactionRecord(BaseModel):
    crypto_txn_id: str
    user_id: str
    occurred_at: str
    direction: str | None
    asset: str | None
    network: str | None
    wallet_id: str | None
    counterparty_wallet_id: str | None
    amount_asset: float | None
    amount_twd_equiv: float | None
    tx_hash: str | None
    status: str | None


class KnownBlacklistUserRecord(BaseModel):
    blacklist_entry_id: str
    user_id: str
    observed_at: str
    source: str
    reason_code: str
    is_active: bool


class DeviceRecord(BaseModel):
    device_id: str
    device_type: str | None
    os_family: str | None
    app_channel: str | None
    device_fingerprint: str | None
    first_seen_at: str


class UserDeviceLinkRecord(BaseModel):
    link_id: str
    user_id: str
    device_id: str
    is_primary: bool | None
    first_seen_at: str
    last_seen_at: str


class BankAccountRecord(BaseModel):
    bank_account_id: str
    bank_code: str | None
    bank_name: str | None
    country: str | None
    currency: str | None
    opened_at: str


class UserBankLinkRecord(BaseModel):
    link_id: str
    user_id: str
    bank_account_id: str
    is_primary: bool | None
    linked_at: str


class CryptoWalletRecord(BaseModel):
    wallet_id: str
    wallet_kind: str | None
    user_id: str | None
    asset: str | None
    network: str | None
    created_at: str


class UsersPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedUsersResponse"})
    items: list[UserRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class LoginEventsPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedLoginEventsResponse"})
    items: list[LoginEventRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class FiatTransactionsPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedFiatTransactionsResponse"})
    items: list[FiatTransactionRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class TradeOrdersPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedTradeOrdersResponse"})
    items: list[TradeOrderRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class CryptoTransactionsPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedCryptoTransactionsResponse"})
    items: list[CryptoTransactionRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class KnownBlacklistUsersPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedKnownBlacklistUsersResponse"})
    items: list[KnownBlacklistUserRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class DevicesPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedDevicesResponse"})
    items: list[DeviceRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class UserDeviceLinksPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedUserDeviceLinksResponse"})
    items: list[UserDeviceLinkRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class BankAccountsPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedBankAccountsResponse"})
    items: list[BankAccountRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class UserBankLinksPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedUserBankLinksResponse"})
    items: list[UserBankLinkRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


class CryptoWalletsPage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "PaginatedCryptoWalletsResponse"})
    items: list[CryptoWalletRecord]
    page: int
    page_size: int
    total: int
    has_next: bool


def get_default_data_dir() -> Path:
    value = os.getenv("BITOGUARD_DATA_DIR")
    if value:
        return Path(value)
    return Path(__file__).resolve().parents[2] / "bitoguard_sim_output"


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="BitoGuard v1 Mock API",
        version="1.0.0",
        description="Source-facing, read-only mock API built from BitoGuard pseudo data.",
    )
    app.state.store = DataStore(data_dir or get_default_data_dir())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code="invalid_request",
                message="Invalid request parameters.",
                details=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(InvalidFilterValue)
    async def invalid_filter_exception_handler(_: Request, exc: InvalidFilterValue) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code="invalid_filter",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code="http_error",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    def query_endpoint(
        endpoint_name: str,
        *,
        filters: dict[str, Any],
        start_time: datetime | None,
        end_time: datetime | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        return app.state.store.query(
            endpoint_name=endpoint_name,
            filters=filters,
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/users", response_model=UsersPage, responses={400: {"model": ErrorResponse}})
    def list_users(
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "users",
            filters={"user_id": user_id},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/login-events", response_model=LoginEventsPage, responses={400: {"model": ErrorResponse}})
    def list_login_events(
        user_id: str | None = None,
        device_id: str | None = None,
        success: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "login_events",
            filters={"user_id": user_id, "device_id": device_id, "success": success},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/fiat-transactions", response_model=FiatTransactionsPage, responses={400: {"model": ErrorResponse}})
    def list_fiat_transactions(
        user_id: str | None = None,
        bank_account_id: str | None = None,
        direction: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "fiat_transactions",
            filters={
                "user_id": user_id,
                "bank_account_id": bank_account_id,
                "direction": direction,
                "status": status,
            },
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/trade-orders", response_model=TradeOrdersPage, responses={400: {"model": ErrorResponse}})
    def list_trade_orders(
        user_id: str | None = None,
        side: str | None = None,
        status: str | None = None,
        base_asset: str | None = None,
        quote_asset: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "trade_orders",
            filters={
                "user_id": user_id,
                "side": side,
                "status": status,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
            },
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/crypto-transactions", response_model=CryptoTransactionsPage, responses={400: {"model": ErrorResponse}})
    def list_crypto_transactions(
        user_id: str | None = None,
        wallet_id: str | None = None,
        counterparty_wallet_id: str | None = None,
        direction: str | None = None,
        status: str | None = None,
        asset: str | None = None,
        network: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "crypto_transactions",
            filters={
                "user_id": user_id,
                "wallet_id": wallet_id,
                "counterparty_wallet_id": counterparty_wallet_id,
                "direction": direction,
                "status": status,
                "asset": asset,
                "network": network,
            },
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/known-blacklist-users", response_model=KnownBlacklistUsersPage, responses={400: {"model": ErrorResponse}})
    def list_known_blacklist_users(
        user_id: str | None = None,
        is_active: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "known_blacklist_users",
            filters={"user_id": user_id, "is_active": is_active},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/devices", response_model=DevicesPage, responses={400: {"model": ErrorResponse}})
    def list_devices(
        device_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "devices",
            filters={"device_id": device_id},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/user-device-links", response_model=UserDeviceLinksPage, responses={400: {"model": ErrorResponse}})
    def list_user_device_links(
        user_id: str | None = None,
        device_id: str | None = None,
        is_primary: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "user_device_links",
            filters={"user_id": user_id, "device_id": device_id, "is_primary": is_primary},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/bank-accounts", response_model=BankAccountsPage, responses={400: {"model": ErrorResponse}})
    def list_bank_accounts(
        bank_account_id: str | None = None,
        country: str | None = None,
        currency: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "bank_accounts",
            filters={"bank_account_id": bank_account_id, "country": country, "currency": currency},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/user-bank-links", response_model=UserBankLinksPage, responses={400: {"model": ErrorResponse}})
    def list_user_bank_links(
        user_id: str | None = None,
        bank_account_id: str | None = None,
        is_primary: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "user_bank_links",
            filters={"user_id": user_id, "bank_account_id": bank_account_id, "is_primary": is_primary},
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    @app.get("/v1/crypto-wallets", response_model=CryptoWalletsPage, responses={400: {"model": ErrorResponse}})
    def list_crypto_wallets(
        wallet_id: str | None = None,
        user_id: str | None = None,
        wallet_kind: str | None = None,
        asset: str | None = None,
        network: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return query_endpoint(
            "crypto_wallets",
            filters={
                "wallet_id": wallet_id,
                "user_id": user_id,
                "wallet_kind": wallet_kind,
                "asset": asset,
                "network": network,
            },
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

    return app


app = create_app()
