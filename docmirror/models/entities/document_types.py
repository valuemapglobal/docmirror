# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiModal Document Certificates and Struct Schema Definitions
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class DocumentType(str, Enum):
    """Document Type Enumeration"""

    FINANCIAL_REPORT = "financial_report"
    INVOICE = "invoice"
    CONTRACT = "contract"
    BANK_STATEMENT = "bank_statement"
    TAX_REPORT = "tax_report"
    BUSINESS_LICENSE = "business_license"
    ID_CARD = "id_card"
    OTHER = "other"


# Target field schemas for each DocumentType definition
DOCUMENT_FIELD_SCHEMAS: dict[DocumentType, dict[str, str]] = {
    DocumentType.FINANCIAL_REPORT: {
        "company_name": "Company name",
        "report_period": "Report period",
        "report_type": "Report type (Annual/Quarter/Month)",
        "total_revenue": "Total revenue",
        "operating_cost": "Operating constraints/cost",
        "gross_profit": "Gross profit",
        "net_profit": "Net profit",
        "total_assets": "Total assets",
        "total_liabilities": "Total liabilities",
        "owner_equity": "Owner equity",
        "cash_and_equivalents": "Cash and equivalents",
        "accounts_receivable": "Accounts receivable",
        "inventory": "Inventory",
        "fixed_assets": "Fixed assets",
        "accounts_payable": "Accounts payable",
        "short_term_loan": "Short term loans",
        "long_term_loan": "Long term loans",
        "operating_cash_flow": "Operating cash flows",
        "investing_cash_flow": "Investing cash flows",
        "financing_cash_flow": "Financing cash flows",
    },
    DocumentType.INVOICE: {
        "invoice_code": "Invoice code",
        "invoice_number": "Invoice number",
        "invoice_date": "Invoice date",
        "buyer_name": "Buyer name",
        "buyer_tax_id": "Buyer tax identification number",
        "seller_name": "Seller name",
        "seller_tax_id": "Seller tax identification number",
        "items": "Service or goods details",
        "amount_without_tax": "Amount without tax",
        "tax_amount": "Tax amount",
        "total_amount": "Total with tax",
        "invoice_type": "Invoice Type",
    },
    DocumentType.BANK_STATEMENT: {
        "account_name": "Account name",
        "account_number": "Account number",
        "bank_name": "Bank name",
        "statement_period": "Statement period",
        "opening_balance": "Opening balance",
        "closing_balance": "Closing balance",
        "total_deposits": "Total deposits",
        "total_withdrawals": "Total withdrawals",
        "transaction_count": "Total transaction count",
    },
    DocumentType.BUSINESS_LICENSE: {
        "company_name": "Company name",
        "unified_social_credit_code": "Unified social credit code",
        "legal_representative": "Legal representative",
        "registered_capital": "Registered capital",
        "establishment_date": "Establishment date",
        "business_term": "Business term",
        "registered_address": "Registered address",
        "business_scope": "Business scope",
        "company_type": "Company Type",
    },
    DocumentType.CONTRACT: {
        "contract_title": "Contract Title",
        "contract_number": "Contract number",
        "party_a": "Party A",
        "party_b": "Party B",
        "signing_date": "Signing date",
        "effective_date": "Effective date",
        "expiry_date": "Expiry/Expiration date",
        "contract_amount": "Contract amount",
        "payment_terms": "Payment terms",
        "contract_subject": "Contract subject",
    },
    DocumentType.TAX_REPORT: {
        "taxpayer_name": "Taxpayer name",
        "taxpayer_id": "Taxpayer identification number",
        "tax_period": "Tax period",
        "tax_type": "Tax type",
        "taxable_income": "Taxable income",
        "tax_rate": "Tax rate",
        "tax_amount": "Tax amount",
        "tax_paid": "Tax amount paid",
        "tax_due": "Tax amount due",
    },
    DocumentType.ID_CARD: {
        "name": "Name",
        "gender": "Gender",
        "ethnicity": "Ethnicity",
        "birth_date": "Birth date",
        "address": "Address",
        "id_number": "ID card number",
        "issuing_authority": "Issuing authority",
        "valid_period": "Valid period",
    },
    DocumentType.OTHER: {
        "title": "Title",
        "content_summary": "Content abstraction or summary",
        "key_information": "Key information",
    },
}
