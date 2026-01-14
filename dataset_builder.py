import re
import csv
import argparse
from collections import Counter
import numpy as np

PARAGRAPHS = [
    # 1
    "A retail bank evaluates customer profiles by analysing income statements, transaction history, and credit bureau scores. Relationship managers assess eligibility before recommending suitable savings or loan products.",

    # 2
    "During personal loan processing, the bank performs KYC verification, identity validation, and income assessment. Approved applications generate sanction letters with interest rates and EMI schedules.",

    # 3
    "Credit card systems track spending patterns, billing cycles, and repayment behaviour. Minimum due amounts, late fees, and interest charges are calculated automatically during statement generation.",

    # 4
    "A mortgage application requires property valuation, legal verification, and risk assessment. Loan disbursement occurs only after compliance checks and collateral documentation are completed.",

    # 5
    "Retail banking platforms support account opening, balance enquiry, fund transfer, and cheque services. Core banking systems ensure real-time transaction consistency across branches.",

    # 6
    "Digital banking channels allow customers to transfer funds using UPI, NEFT, and IMPS. Transaction limits and authentication rules reduce fraud and unauthorised access.",

    # 7
    "A bank’s risk management team monitors non-performing assets by analysing overdue accounts and repayment delays. Early warning signals help reduce default probability.",

    # 8
    "Treasury operations manage liquidity by balancing deposits, advances, and interbank borrowing. Interest rate movements influence investment and lending strategies.",

    # 9
    "Customer service teams resolve complaints related to failed transactions, incorrect charges, and delayed refunds. Service level agreements define resolution timelines.",

    # 10
    "Loan underwriting evaluates borrower risk using credit scores, employment stability, and debt-to-income ratios. Automated decision engines improve approval efficiency.",

    # 11
    "Fraud detection systems flag unusual transactions based on velocity checks, location mismatch, and spending anomalies. Alerts trigger manual review by compliance officers.",

    # 12
    "Retail banks offer fixed deposits with predefined tenure and interest payouts. Customers choose cumulative or non-cumulative options based on financial goals.",

    # 13
    "Wealth management services recommend mutual funds, bonds, and insurance products. Portfolio diversification helps customers manage market volatility.",

    # 14
    "Branch operations handle cash deposits, withdrawals, and teller balancing. Daily reconciliation ensures accurate accounting and audit readiness.",

    # 15
    "Credit bureaus maintain borrower histories, recording loan accounts, repayment status, and defaults. Banks use this data to assess lending risk.",

    # 16
    "Payment gateways process online transactions by validating card details and routing requests securely. Settlement cycles determine fund availability to merchants.",

    # 17
    "Retail finance analytics evaluate customer lifetime value using transaction frequency and product usage. Insights support cross-selling and retention strategies.",

    # 18
    "ATM networks enable cash withdrawal, balance enquiry, and mini statements. Monitoring systems track uptime, cash levels, and transaction failures.",

    # 19
    "Interest rate changes impact loan affordability and deposit returns. Banks adjust pricing strategies based on central bank policy updates.",

    # 20
    "Retail loan collections involve reminder notifications, restructuring options, and recovery processes. Ethical practices ensure regulatory compliance.",

    # 21
    "Mobile banking applications provide secure login, biometric authentication, and transaction alerts. Regular updates address performance and security issues.",

    # 22
    "Banks maintain capital adequacy ratios to absorb potential losses. Regulatory reporting ensures transparency and financial stability.",

    # 23
    "Merchant acquiring services enable businesses to accept card and QR payments. Settlement reports help merchants track daily collections.",

    # 24
    "Retail banking compliance teams enforce anti-money laundering rules through transaction monitoring and reporting suspicious activities.",

    # 25
    "Savings accounts earn interest based on average monthly balance. Fee structures vary depending on account type and customer segment.",

    # 26
    "Loan repayment schedules define principal and interest components. Amortisation tables help customers understand outstanding balances.",

    # 27
    "Digital wallets store prepaid balances for quick payments. KYC levels determine wallet limits and transfer capabilities.",

    # 28
    "Bank audits review internal controls, transaction logs, and policy adherence. Findings guide process improvement initiatives.",

    # 29
    "Retail finance dashboards track deposits, advances, and profitability metrics. Management uses these indicators for strategic planning.",

    # 30
    "Customer onboarding involves document verification, risk profiling, and account activation. Automation reduces turnaround time.",

    # 31
    "Chargeback handling resolves disputed card transactions through evidence review and network arbitration. Timely response reduces losses.",

    # 32
    "Interest compounding frequency affects deposit maturity value. Customers compare options before selecting investment products.",

    # 33
    "Retail banks offer overdraft facilities linked to savings accounts. Usage attracts interest only on utilised amounts.",

    # 34
    "Loan prepayment options allow borrowers to reduce tenure or EMI burden. Policies define applicable charges and limits.",

    # 35
    "Customer relationship management systems store interaction history, preferences, and service requests to improve engagement quality."
]



WINDOW_SIZE = 4


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def build_vocab(tokens):
    freq = Counter(tokens)
    word2id = {w: i for i, w in enumerate(freq.keys())}
    return word2id, freq


def save_vocab(word2id, freq):
    with open("vocab.txt", "w") as f:
        f.write("word,id,frequency\n")
        for w, i in word2id.items():
            f.write(f"{w},{i},{freq[w]}\n")


def generate_datasets(tokens, word2id):
    cbow_rows = []
    skip_rows = []
    for i in range(WINDOW_SIZE, len(tokens) - WINDOW_SIZE):
        target = word2id[tokens[i]]
        context = []
        for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
            if j != i:
                context.append(word2id[tokens[j]])
        cbow_rows.append(context + [target])
        for c in context:
            skip_rows.append((target, c))
    return cbow_rows, skip_rows


def save_dataset(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    tokens = []
    for p in PARAGRAPHS:
        tokens.extend(preprocess(p))

    word2id, freq = build_vocab(tokens)
    save_vocab(word2id, freq)

    cbow, skip = generate_datasets(tokens, word2id)
    save_dataset("cbow_dataset.csv", cbow)
    save_dataset("skipgram_dataset.csv", skip)

    print("Dataset and vocabulary generated successfully")
