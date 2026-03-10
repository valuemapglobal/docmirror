"""
Domain Models — 领域特定数据模型

按文档类型 (DocumentType) 定义结构化字段,
由 Parser 在识别文档类型后选择性填充。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# 银行流水
# ═══════════════════════════════════════════════════════════════════════════

class TransactionRow(BaseModel):
    """单笔交易"""
    date: str = ""
    description: str = ""
    amount: float = 0.0
    balance: Optional[float] = None
    counterparty: str = ""
    transaction_type: str = ""         # "消费" | "转账" | "代付" ...
    currency: str = "CNY"


class BankStatementData(BaseModel):
    """银行流水领域数据"""
    account_holder: str = ""           # 账户持有人
    account_number: str = ""           # 账号
    bank_name: str = ""                # 开户行
    query_period: str = ""             # "20240620 - 20250620"
    currency: str = "CNY"
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    transaction_count: int = 0
    total_deposits: Optional[float] = None
    total_withdrawals: Optional[float] = None
    transactions: List[TransactionRow] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 发票
# ═══════════════════════════════════════════════════════════════════════════

class InvoiceItem(BaseModel):
    """发票明细行"""
    name: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    tax_rate: Optional[str] = None
    tax_amount: Optional[float] = None


class InvoiceData(BaseModel):
    """发票领域数据"""
    invoice_code: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    invoice_type: str = ""             # "增值税专用发票" | "增值税普通发票"
    buyer_name: str = ""
    buyer_tax_id: str = ""
    seller_name: str = ""
    seller_tax_id: str = ""
    amount_without_tax: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    items: List[InvoiceItem] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 税务报告
# ═══════════════════════════════════════════════════════════════════════════

class TaxReportData(BaseModel):
    """税务报告领域数据"""
    taxpayer_name: str = ""
    taxpayer_id: str = ""
    tax_period: str = ""
    tax_type: str = ""
    taxable_income: Optional[float] = None
    tax_rate: Optional[str] = None
    tax_amount: Optional[float] = None
    tax_paid: Optional[float] = None
    tax_due: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# 营业执照
# ═══════════════════════════════════════════════════════════════════════════

class BusinessLicenseData(BaseModel):
    """营业执照领域数据"""
    company_name: str = ""
    unified_social_credit_code: str = ""
    legal_representative: str = ""
    registered_capital: str = ""
    establishment_date: str = ""
    business_term: str = ""
    registered_address: str = ""
    business_scope: str = ""
    company_type: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 合同
# ═══════════════════════════════════════════════════════════════════════════

class ContractData(BaseModel):
    """合同领域数据"""
    contract_title: str = ""
    contract_number: str = ""
    party_a: str = ""
    party_b: str = ""
    signing_date: str = ""
    effective_date: str = ""
    expiry_date: str = ""
    contract_amount: Optional[float] = None
    payment_terms: str = ""
    contract_subject: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 财务报告
# ═══════════════════════════════════════════════════════════════════════════

class FinancialReportData(BaseModel):
    """财务报告领域数据"""
    company_name: str = ""
    report_period: str = ""
    report_type: str = ""              # "年报" | "季报" | "月报"
    total_revenue: Optional[float] = None
    operating_cost: Optional[float] = None
    gross_profit: Optional[float] = None
    net_profit: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    owner_equity: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# 身份证
# ═══════════════════════════════════════════════════════════════════════════

class IDCardData(BaseModel):
    """身份证领域数据"""
    name: str = ""
    gender: str = ""
    ethnicity: str = ""
    birth_date: str = ""
    address: str = ""
    id_number: str = ""
    issuing_authority: str = ""
    valid_period: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DomainData 容器
# ═══════════════════════════════════════════════════════════════════════════

class DomainData(BaseModel):
    """
    领域特定数据容器。

    根据 ``document_type`` 填充对应子字段。
    同一时刻通常只有一个子字段非 None。
    """
    document_type: str = "other"

    bank_statement: Optional[BankStatementData] = None
    invoice: Optional[InvoiceData] = None
    tax_report: Optional[TaxReportData] = None
    business_license: Optional[BusinessLicenseData] = None
    contract: Optional[ContractData] = None
    financial_report: Optional[FinancialReportData] = None
    id_card: Optional[IDCardData] = None

    @property
    def active_model(self) -> Optional[BaseModel]:
        """返回当前已填充的领域模型 (若有)"""
        for field_name in [
            "bank_statement", "invoice", "tax_report",
            "business_license", "contract", "financial_report", "id_card",
        ]:
            val = getattr(self, field_name, None)
            if val is not None:
                return val
        return None
