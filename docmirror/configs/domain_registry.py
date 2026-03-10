"""
Domain 注册表 — 按文档类型注册身份字段
==========================================

替代 to_api_dict() 中硬编码的银行流水字段。
扩展方式: 在 DOMAIN_IDENTITY 中添加新 domain 即可。
"""

from typing import Any, Dict, List, Tuple


# domain_type → [(展示名, 候选key1, 候选key2, ...)]
DOMAIN_IDENTITY: Dict[str, List[Tuple[str, ...]]] = {
    "bank_statement": [
        ("institution", "bank_name", "开户行", "开户机构"),
        ("account_holder", "户名", "本方户名", "账户名", "开户人", "持卡人", "客户姓名", "客户名称"),
        ("account_number", "账号", "卡号", "账户", "客户账号"),
        ("query_period", "查询期间", "期间", "起止日期"),
        ("currency", "币种"),
        ("print_date", "打印日期"),
    ],
    "invoice": [
        ("supplier", "供应商", "销售方", "开票方"),
        ("buyer", "购买方", "购方", "收票方"),
        ("invoice_no", "发票号码", "发票号", "Invoice No"),
        ("amount", "金额", "合计金额", "价税合计"),
        ("tax", "税额", "税率"),
        ("date", "开票日期", "日期"),
    ],
    "contract": [
        ("party_a", "甲方", "Party A"),
        ("party_b", "乙方", "Party B"),
        ("contract_no", "合同编号", "Contract No"),
        ("sign_date", "签署日期", "签订日期"),
        ("amount", "合同金额", "金额"),
    ],
    "receipt": [
        ("merchant", "商户名", "商户", "Merchant"),
        ("amount", "交易金额", "金额", "Amount"),
        ("date", "交易日期", "日期", "Date"),
    ],
    # 通配 fallback
    "*": [
        ("title", "标题", "Title"),
        ("date", "日期", "Date"),
        ("author", "作者", "Author"),
    ],
}


def resolve_identity(domain: str, entities: Dict[str, Any]) -> Dict[str, str]:
    """
    根据 domain 类型从 entities 中提取标准化身份字段。

    Args:
        domain:   文档类型 (bank_statement, invoice, contract, ...)
        entities: 提取到的 key-value 实体

    Returns:
        标准化身份字典 {display_name: value}
    """
    fields = DOMAIN_IDENTITY.get(domain, DOMAIN_IDENTITY.get("*", []))
    identity: Dict[str, str] = {"document_type": domain}

    for field_def in fields:
        display_name = field_def[0]
        candidates = field_def[1:]
        for key in candidates:
            val = entities.get(key, "")
            if val:
                identity[display_name] = str(val)
                break
        else:
            identity[display_name] = ""

    return identity
