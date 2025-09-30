import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager

from state import State

# ───────────────────────────────────────────────────────────────────────────────
# ENV
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv()
# OPENAI_API_KEY는 langchain_openai가 내부에서 env로 읽어가므로 여기서 별도 사용 안 함.

# ───────────────────────────────────────────────────────────────────────────────
# 폰트 설정 (ReportLab + Matplotlib)
# ───────────────────────────────────────────────────────────────────────────────
FONT_PATH = os.path.join(os.path.dirname(__file__), "NotoSansKR-VariableFont_wght.ttf")
FONT_NAME = "NotoSansKR"
DOC_FONT_NAME = "Helvetica"  # ReportLab 기본 폰트 fallback
font_prop = None             # Matplotlib fallback

# ReportLab 폰트 등록
if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        DOC_FONT_NAME = FONT_NAME
    except Exception as e:
        print(f"⚠️ ReportLab 폰트 등록 실패: {e} → 기본 폰트 사용")
else:
    print(f"⚠️ ReportLab: {FONT_PATH} 경로에 폰트 없음 → 기본 폰트 사용")

# Matplotlib 폰트 등록
if os.path.exists(FONT_PATH):
    try:
        font_prop = font_manager.FontProperties(fname=FONT_PATH)
        # 폰트 이름이 시스템에 등록되지 않은 경우 대비: 직접 family 지정
        plt.rcParams["font.sans-serif"] = [font_prop.get_name()] if font_prop.get_name() else []
        plt.rcParams["font.family"] = "sans-serif"
    except Exception as e:
        print(f"⚠️ Matplotlib 폰트 등록 실패: {e} → 기본 폰트 사용")
else:
    print(f"⚠️ Matplotlib: {FONT_PATH} 경로에 폰트 없음 → 기본 폰트 사용")

# ───────────────────────────────────────────────────────────────────────────────
# 보고서 프롬프트
# ───────────────────────────────────────────────────────────────────────────────
report_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 벤처캐피탈(VC) 애널리스트다. 모든 판단은 근거 기반으로 명확히 제시한다. "
        "아래 구조로 '스타트업 투자 분석 보고서'를 한국어로 작성해라."
    ),
    (
        "user",
        """[입력 데이터]
회사명: {company_name}
요약 정보: {summary}
세부 정보: {details}
투자 평가 기준: {criteria_list}

[작성 형식]
## 스타트업 투자 분석 보고서

### 1. 회사 개요
- 회사명/설립연도/주요 서비스/산업 분야(가능한 경우)
- 핵심 가치제안(USP, 경쟁사 대비 차별점)

### 2. 투자 평가 기준별 분석
{criteria_bullets}

### 3. 종합 평가 및 리스크 요인
- 장점 요약 : 투자 권장 이유를 평가지표를 통해 구체적으로 설명
- 리스크 요약
"""
    )
])

DEFAULT_CRITERIA = [
    "창업자", "시장성", "제품기술력",
    "경쟁우위", "실적", "투자조건", "리스크"
]

# ───────────────────────────────────────────────────────────────────────────────
# 레이더차트 생성
# ───────────────────────────────────────────────────────────────────────────────
def generate_radar_chart(scores: Dict[str, int], filename: str = "radar_chart.png"):
    if not scores:
        print("⚠️ 레이더차트 생성 불가: 점수 데이터 없음")
        return None

    categories = list(scores.keys())
    try:
        values = [int(scores[k]) for k in categories]
    except Exception:
        print("⚠️ 레이더차트: 점수 변환 실패 → 스킵")
        return None

    if not categories or not values:
        print("⚠️ 레이더차트 생성 불가: 카테고리/값 없음")
        return None

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles_c, values_c, linewidth=2)
    ax.fill(angles_c, values_c, alpha=0.25)

    ax.set_xticks(angles)
    if font_prop:
        ax.set_xticklabels(categories, fontsize=9, fontproperties=font_prop)
        ax.set_title(" ", fontsize=14, pad=20, fontproperties=font_prop)
    else:
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(" ", fontsize=14, pad=20)

    vmax = max(values) if values else 0
    try:
        ax.set_yticks(range(0, vmax + 1))
    except Exception:
        pass
    ax.set_ylim(0, max(vmax, 5))  # 최소 범위 확보

    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️ 레이더차트 저장 실패: {e}")
        return None
    return filename

# ───────────────────────────────────────────────────────────────────────────────
# PDF 저장
# ───────────────────────────────────────────────────────────────────────────────
def save_pdf(company_name: str, report_json: Dict[str, Any], chart_file: str, output_path: str = "report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    styles = getSampleStyleSheet()
    if "KoreanNormal" not in styles:
        styles.add(ParagraphStyle(name="KoreanNormal", fontName=DOC_FONT_NAME, fontSize=10, leading=14))
    if "KoreanHeading" not in styles:
        styles.add(ParagraphStyle(name="KoreanHeading", fontName=DOC_FONT_NAME, fontSize=14, leading=18, spaceAfter=12))

    # 1) 기업소개
    story.append(Paragraph("1. 기업소개", styles["KoreanHeading"]))
    intro_text = (report_json.get("기업소개", {}) or {}).get("text", "정보 없음")
    intro_text = (intro_text or "").replace("\n", "<br/>")
    story.append(Paragraph(intro_text, styles["KoreanNormal"]))
    story.append(Spacer(1, 12))

    # 2) 레이더차트
    story.append(Paragraph("2. 종합평가 레이더그래프", styles["KoreanHeading"]))
    if chart_file and os.path.exists(chart_file):
        story.append(Image(chart_file, width=250, height=250))
    else:
        story.append(Paragraph("차트 이미지 없음", styles["KoreanNormal"]))
    story.append(Spacer(1, 12))

    # 3) 평가점수 리뷰
    story.append(Paragraph("3. 평가점수 리뷰", styles["KoreanHeading"]))
    table_data = [["항목", "총점", "세부 점수(0 : 낮음 / 1 : 보통 / 2 : 높음)"]]
    for row in (report_json.get("평가점수리뷰", {}) or {}).get("table", []):
        table_data.append([row.get("항목", ""), row.get("총점", ""), row.get("세부", "")])

    table = Table(table_data, colWidths=[80, 40, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, -1), DOC_FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    try:
        doc.build(story)
        print(f"✅ PDF 보고서 저장 완료: {output_path}")
    except Exception as e:
        print(f"⚠️ PDF 생성 실패: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────────────────────────
def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _safe_filename(name: str) -> str:
    out = re.sub(r"[^\w\-.가-힣 ]", "_", name or "")
    return out.strip() or "report"

# ───────────────────────────────────────────────────────────────────────────────
# LangGraph 노드: 보고서 작성
# ───────────────────────────────────────────────────────────────────────────────
def report_writer_node(state: State) -> State:
    """
    evaluation(점수/판정)을 바탕으로 보고서 텍스트 생성 → 레이더차트 → PDF 저장.
    state 업데이트:
      - report_written: True
      - report_path: 생성된 PDF 경로
    """
    evaluation = state.get("evaluation") or {}
    if not evaluation:
        # 평가가 없으면 종료
        return state
    if state.get("investment_decision") is False:
        # 불합격이면 보고서 스킵
        return state

    company = state.get("current_company") or "startup"

    # LLM 호출 준비
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = report_prompt | llm

    summary = evaluation.get("요약", "정보 없음")
    details = json.dumps(evaluation, ensure_ascii=False, indent=2)
    criteria_list = ", ".join(DEFAULT_CRITERIA)
    criteria_bullets = "\n".join([f"- {c}: {evaluation.get(c, 'N/A')}" for c in DEFAULT_CRITERIA])

    # 보고서 본문 텍스트 생성(회사 개요/지표별 분석/종합평가 문단)
    try:
        result = chain.invoke({
            "company_name": company,
            "summary": summary,
            "details": details,
            "criteria_list": criteria_list,
            "criteria_bullets": criteria_bullets,
        })
        intro_text = (getattr(result, "content", "") or "").strip() or "정보 없음"
    except Exception as e:
        print(f"⚠️ LLM 보고서 본문 생성 실패: {e}")
        intro_text = "정보 없음"

    # 점수/테이블 구성
    scores: Dict[str, int] = {}
    table_rows = []
    for c in DEFAULT_CRITERIA:
        cell = evaluation.get(c)
        total = 0
        detail = ""
        if isinstance(cell, dict):
            if "총점" in cell:
                total = _to_int(cell["총점"])
            elif "score" in cell:
                total = _to_int(cell["score"])
            parts = [f"{k}:{v}" for k, v in cell.items() if k not in ("총점", "score")]
            detail = ", ".join(parts)
        elif isinstance(cell, (int, float, str)):
            if isinstance(cell, (int, float)):
                total = int(cell)
            else:
                total = _to_int(cell)
            detail = str(cell)
        scores[c] = total
        table_rows.append({"항목": c, "총점": total, "세부": detail})

    report_json = {
        "기업소개": {"text": intro_text},
        "레이더차트": {"scores": scores},
        "평가점수리뷰": {"table": table_rows},
        "종합평가": {
            "장점": evaluation.get("장점", ""),
            "리스크": evaluation.get("리스크", ""),
            "최종권고": evaluation.get("최종판정", ""),
        },
    }

    # 차트/ PDF 생성
    base = _safe_filename(company)
    chart_path = generate_radar_chart(report_json.get("레이더차트", {}).get("scores", {}), filename=f"{base}_radar.png")
    pdf_path = f"{base}_investment_report.pdf"
    save_pdf(company, report_json, chart_path, pdf_path)

    # state 업데이트 후 반환
    state.update({
        "report_written": True,
        "report_path": pdf_path
    })
    return state

# --- alias for graph.py compatibility ---
def report_writer_agent(state: State) -> State:
    return report_writer_node(state)

__all__ = ["report_writer_node", "report_writer_agent"]
