"""
Bank Statement Domain Plugin (Built-in)
=======================================

Provides bank-statement-specific processing:
- Scene detection keywords
- Identity field definitions (account holder, account number, etc.)
- Domain data construction (BankStatementData)
"""
from __future__ import annotations


from typing import Any, Dict, Optional, Sequence, Tuple

from docmirror.plugins import DomainPlugin


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
            ("account_holder", (
                "Account holder", "Account name",
                "Card holder", "Customer name"
            )),
            ("account_number", (
                "Account number", "Card number", "Customer account number"
            )),
            ("bank_name", ("Bank name", "Bank branch", "bank_name")),
            ("query_period", ("Query period", "From/to date", "Period")),
            ("currency", ("Currency",)),
            ("print_date", ("Print date",)),
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
            account_holder=str(metadata.get(
                "Account holder", entities.get("account_holder", "")
            )),
            account_number=str(metadata.get(
                "Account number", entities.get("account_number", "")
            )),
            bank_name=str(entities.get("bank_name", "")),
            query_period=str(metadata.get(
                "Query period", entities.get("query_period", "")
            )),
            currency=(
                str(metadata.get(
                    "Currency", entities.get("currency", "CNY")
                )) or "CNY"
            ),
        )
        return DomainData(document_type="bank_statement", bank_statement=bs)

    def get_middleware_config(self) -> Dict[str, Any]:
        return {
            "institution_detector_enabled": True,
            "amount_splitter_enabled": True,
        }


# Auto-discovery convention: module-level `plugin` instance
plugin = BankStatementPlugin()
