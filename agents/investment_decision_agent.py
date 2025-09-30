import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from state import State

# .env에서 OPENAI_API_KEY 불러오기
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# 📌 평가 프롬프트
evaluation_prompt = ChatPromptTemplate.from_template("""
너는 스타트업 투자 평가 전문가다.
주어진 기업 정보를 바탕으로 아래 항목별 평가를 진행하라.

⚠️ 반드시 JSON 형식으로만 출력할 것.  
설명 문장이나 보고서 형태는 절대 출력하지 말고, 아래 스키마와 동일하게 출력한다.

점수 기준:
- 창업자.전문성: 0~2점
  - 2점: 구성원의 일정 % 이상이 관련 전공 또는 경력 보유
  - 1점: 일부만 관련 전공/경력
  - 0점: 전혀 없음
- 창업자.실행력: 0~1점

- 시장성.시장크기: 0~2점
  - 2점: 현재와 미래 모두 성장 가능성이 높음
  - 1점: 현재는 성장 중이나 미래 불확실
  - 0점: 시장 제한적
- 시장성.성장가능성: 0~1점
- 시장성.고객수요: 0~1점

- 제품기술력.독창성: 0~2점
  - 2점: 독보적인 기술/특허/완전히 새로운 서비스
  - 1점: 경쟁사 대비 차별적 기술 있음
  - 0점: 평범
- 제품기술력.구현가능성: 0~1점

- 경쟁우위.차별성: 0~2점
  - 2점: 완전히 새로운 서비스/기술
  - 1점: 뚜렷한 차별점 존재
  - 0점: 차별성 없음
- 경쟁우위.진입장벽: 0~1점

- 실적.고객반응: 0~2점
  - 2점: 고객 반응 매우 긍정 + 피드백 적극 반영
  - 1점: 일부 긍정적 반응
  - 0점: 반응 미약/부정
- 실적.매출계약: 0~1점

- 투자조건.투자단계: 0~1점
- 투자조건.투자금액: 0~2점
  - 2점: 충분히 실현 가능하며 감당 가능한 투자 규모
  - 1점: 소규모, 리스크 대비 보수적
  - 0점: 불명확

- 리스크 항목(각각 0~2점): 점수가 높을수록 위험
  - 2점: 심각한 리스크 존재(지나치게 도전적인 목표, 소송여부 등)
  - 1점: 일부 리스크 존재(타의에 의한 리스크 포함 여부)
  - 기술리스크, 운영리스크, 법률리스크

출력 스키마 예시:
{{
  "기업소개": "간단 요약",
  "창업자": {{
    "전문성": int,
    "실행력": int,
    "총점": int
  }},
  "시장성": {{
    "시장크기": int,
    "성장가능성": int,
    "고객수요": int,
    "총점": int
  }},
  "제품기술력": {{
    "독창성": int,
    "구현가능성": int,
    "총점": int
  }},
  "경쟁우위": {{
    "차별성": int,
    "진입장벽": int,
    "총점": int
  }},
  "실적": {{
    "고객반응": int,
    "매출계약": int,
    "총점": int
  }},
  "투자조건": {{
    "투자단계": int,
    "투자금액": int,
    "총점": int
  }},
  "리스크": {{
    "기술리스크": int,
    "운영리스크": int,
    "법률리스크": int,
    "총점": int
  }},
  "최종점수": int,
  "최종판정": "합격/불합격"
}}

기업 정보:
{company_info}
""")
class EvaluationAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = evaluation_prompt | self.llm

    def evaluate(self, company_info: str) -> dict:
        """기업 정보를 평가해서 JSON(dict) 반환"""
        result = self.chain.invoke({"company_info": company_info})
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패, 원본 응답 반환")
            return {"raw_output": result.content}


        # ✅ 점수 합산 + 판정
        data = self._calculate_scores(data)

        # ✅ 근거 설명 요청
        explanation_chain = self.explain_prompt | self.llm
        explain_result = explanation_chain.invoke({"evaluation": json.dumps(data, ensure_ascii=False)})
        try:
            explanations = json.loads(explain_result.content)
            # 점수 JSON에 설명 병합
            for section, details in explanations.items():
                if section in data:
                    data[section].update(details)
        except json.JSONDecodeError:
            print("⚠️ 설명 JSON 파싱 실패, 근거 없음")

        return data

    def _calculate_scores(self, data: dict) -> dict:
        """총점 계산 및 최종판정"""
        def subtotal(section: dict) -> int:
            return sum(v for v in section.values() if isinstance(v, int))

        for key in ["창업자", "시장성", "제품기술력", "경쟁우위", "실적", "투자조건", "리스크"]:
            section = data.get(key, {})
            section["총점"] = subtotal(section)
            data[key] = section

        total_score = sum(
            data[k]["총점"] for k in ["창업자", "시장성", "제품기술력", "경쟁우위", "실적", "투자조건"]
        ) - data["리스크"]["총점"]
        data["최종점수"] = total_score

        if total_score >= 20 and data["시장성"]["총점"] >= 2:
            data["최종판정"] = "합격"
        else:
            data["최종판정"] = "불합격"

        return data
    
    
    
class EvaluationAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = evaluation_prompt | self.llm

    def evaluate(self, company_info: str) -> Dict[str, Any]:
        result = self.chain.invoke({"company_info": company_info})
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"raw_output": result.content}

# ✅ LangGraph 노드
def evaluation_agent_node(state: State) -> State:
    company = state.get("current_company") or ""
    tags = ", ".join(state.get("current_tags", []))

    # 앞선 노드에서 수집한 정보들을 합쳐 한 덩어리 텍스트로 평가 입력 생성
    stitched_info = f"""회사명: {company}
태그: {tags}
시장성: {state.get('market_analysis') or ''}
경쟁사분석: {state.get('competitor_analysis') or ''}"""

    agent = EvaluationAgent()
    evaluation = agent.evaluate(stitched_info)

    # 최종판정 → investment_decision(bool)
    decision_bool = (evaluation.get("최종판정") == "합격")
    state["evaluation"] = evaluation
    state["investment_decision"] = decision_bool
    return state    
    
    
# 실행 예시 (GRIDY 데이터 기반)
if __name__ == "__main__":
    gridy_info = """
    GRIDY는 한국 최초로 제공되는 자전거 기반의 친환경 배송 서비스 스타트업이다.
    2025년 8월 서울 4개 구에서 3주간 시범 서비스를 성공적으로 운영했고, 고객 재사용율은 61%였다.
    공식 서비스는 2025년 10월 오픈 예정이다.
    ESG, 기후테크, 탄소감축, 클린테크 등 친환경 트렌드와 맞물려 있으며
    현재 엔젤투자를 받았고, 희망 투자 단계는 시드, 희망 투자금액은 1억~3억원이다.
    창업자는 자전거 메신저 경험과 디자인·전략, 앱 개발 전문성을 갖춘 인물들로 구성된 1~10명의 팀이다.
    """

    agent = EvaluationAgent()
    evaluation = agent.evaluate(gridy_info)

    print("✅ GRIDY 평가 결과")
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    
    