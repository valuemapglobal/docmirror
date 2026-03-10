"""
Bank Statement Domain Plugin (Built-in)
=======================================

Provides bank-statement-specific processing:
- Scene detection keywords
- Identity field definitions (account holder, account number, etc.)
- Domain data construction (BankStatementData)
- Standard column definitions for transaction tables
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from docmirror.plugins import ColumnHint, DomainPlugin


class BankStatementPlugin(DomainPlugin):
    """Built-in plugin for bank statement document processing."""

    @property
    def domain_name(self) -> str:
        return "bank_statement"

    @property
    def display_name(self) -> str:
        return "Bank Statement"

    @property
    def scene_keywords(self) -> Sequence[str]:
        return (
            "bank statement",
            "account statement",
            "transaction history",
            "statement of account",
            # Chinese
            "银行流水",
            "交易明细",
            "对账单",
            "账户流水",
        )

    @property
    def identity_fields(self) -> Sequence[Tuple[str, Sequence[str]]]:
        return (
            ("account_holder", ("Account holder", "Account name", "Card holder", "Customer name")),
            ("account_number", ("Account number", "Card number", "Customer account number")),
            ("bank_name", ("Bank name", "Bank branch", "bank_name")),
            ("query_period", ("Query period", "From/to date", "Period")),
            ("currency", ("Currency",)),
            ("print_date", ("Print date",)),
        )

    @property
    def standard_columns(self) -> Sequence[ColumnHint]:
        return (
            ColumnHint("transaction_date", ("Date", "Trans Date", "Transaction date"), required=True),
            ColumnHint("description", ("Description", "Summary", "Particulars", "Memo"), required=True),
            ColumnHint("debit", ("Debit", "Withdrawal", "Debit amount")),
            ColumnHint("credit", ("Credit", "Deposit", "Credit amount")),
            ColumnHint("balance", ("Balance", "Running Balance", "Closing Balance")),
            ColumnHint("amount", ("Amount", "Transaction amount")),
            ColumnHint("currency", ("Currency", "Ccy")),
            ColumnHint("reference", ("Reference", "Ref No", "Txn Ref")),
            ColumnHint("counterparty", ("Counterparty", "Payee", "Beneficiary")),
            ColumnHint("channel", ("Channel", "Transaction Channel")),
        )

    def build_domain_data(
        self,
        metadata: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> Optional[Any]:
        """Build BankStatementData from extracted metadata and entities."""
        from docmirror.models.entities.domain_models import (
            BankStatementData,
            DomainData,
        )

        bs = BankStatementData(
            account_holder=str(metadata.get("Account holder", entities.get("account_holder", ""))),
            account_number=str(metadata.get("Account number", entities.get("account_number", ""))),
            bank_name=str(entities.get("bank_name", "")),
            query_period=str(metadata.get("Query period", entities.get("query_period", ""))),
            currency=str(metadata.get("Currency", entities.get("currency", "CNY"))) or "CNY",
        )
        return DomainData(document_type="bank_statement", bank_statement=bs)

    def get_middleware_config(self) -> Dict[str, Any]:
        return {
            "column_mapper_enabled": True,
            "institution_detector_enabled": True,
            "amount_splitter_enabled": True,
        }


# Auto-discovery convention: module-level `plugin` instance
plugin = BankStatementPlugin()
