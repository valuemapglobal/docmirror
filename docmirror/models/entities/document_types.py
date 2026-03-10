"""
MultiModal 证照与文档 Schema 定义
"""

from enum import Enum
from typing import Dict

class DocumentType(str, Enum):
    """文档类型枚举"""
    FINANCIAL_REPORT = "financial_report"  # 财务报告
    INVOICE = "invoice"  # 发票
    CONTRACT = "contract"  # 合同
    BANK_STATEMENT = "bank_statement"  # 银行流水
    TAX_REPORT = "tax_report"  # 税务报告
    BUSINESS_LICENSE = "business_license"  # 营业执照
    ID_CARD = "id_card"  # 身份证
    OTHER = "other"  # 其他

# 各文档类型的字段 Schema
DOCUMENT_FIELD_SCHEMAS: Dict[DocumentType, Dict[str, str]] = {
    DocumentType.FINANCIAL_REPORT: {
        "company_name": "企业名称",
        "report_period": "报告期间",
        "report_type": "报告类型（年报/季报/月报）",
        "total_revenue": "营业总收入",
        "operating_cost": "营业成本",
        "gross_profit": "毛利润",
        "net_profit": "净利润",
        "total_assets": "总资产",
        "total_liabilities": "总负债",
        "owner_equity": "所有者权益",
        "cash_and_equivalents": "货币资金",
        "accounts_receivable": "应收账款",
        "inventory": "存货",
        "fixed_assets": "固定资产",
        "accounts_payable": "应付账款",
        "short_term_loan": "短期借款",
        "long_term_loan": "长期借款",
        "operating_cash_flow": "经营活动现金流量净额",
        "investing_cash_flow": "投资活动现金流量净额",
        "financing_cash_flow": "筹资活动现金流量净额",
    },
    DocumentType.INVOICE: {
        "invoice_code": "发票代码",
        "invoice_number": "发票号码",
        "invoice_date": "开票日期",
        "buyer_name": "购买方名称",
        "buyer_tax_id": "购买方纳税人识别号",
        "seller_name": "销售方名称",
        "seller_tax_id": "销售方纳税人识别号",
        "items": "商品或服务明细",
        "amount_without_tax": "不含税金额",
        "tax_amount": "税额",
        "total_amount": "价税合计",
        "invoice_type": "发票类型",
    },
    DocumentType.BANK_STATEMENT: {
        "account_name": "账户名称",
        "account_number": "账号",
        "bank_name": "开户行",
        "statement_period": "账单周期",
        "opening_balance": "期初余额",
        "closing_balance": "期末余额",
        "total_deposits": "存入总额",
        "total_withdrawals": "支出总额",
        "transaction_count": "交易笔数",
    },
    DocumentType.BUSINESS_LICENSE: {
        "company_name": "企业名称",
        "unified_social_credit_code": "统一社会信用代码",
        "legal_representative": "法定代表人",
        "registered_capital": "注册资本",
        "establishment_date": "成立日期",
        "business_term": "营业期限",
        "registered_address": "住所",
        "business_scope": "经营范围",
        "company_type": "公司类型",
    },
    DocumentType.CONTRACT: {
        "contract_title": "合同标题",
        "contract_number": "合同编号",
        "party_a": "甲方",
        "party_b": "乙方",
        "signing_date": "签订日期",
        "effective_date": "生效日期",
        "expiry_date": "到期日期",
        "contract_amount": "合同金额",
        "payment_terms": "付款条款",
        "contract_subject": "合同标的",
    },
    DocumentType.TAX_REPORT: {
        "taxpayer_name": "纳税人名称",
        "taxpayer_id": "纳税人识别号",
        "tax_period": "税款所属期",
        "tax_type": "税种",
        "taxable_income": "应纳税所得额",
        "tax_rate": "税率",
        "tax_amount": "应纳税额",
        "tax_paid": "已缴税额",
        "tax_due": "应补（退）税额",
    },
    DocumentType.ID_CARD: {
        "name": "姓名",
        "gender": "性别",
        "ethnicity": "民族",
        "birth_date": "出生日期",
        "address": "住址",
        "id_number": "身份证号码",
        "issuing_authority": "签发机关",
        "valid_period": "有效期限",
    },
    DocumentType.OTHER: {
        "title": "标题",
        "content_summary": "内容摘要",
        "key_information": "关键信息",
    },
}
