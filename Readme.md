# Mobility Startup Investment Evaluation Agent
본 프로젝트는 **모빌리티 스타트업**에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다. 
NextUnicorn에서 모빌리티 분야 스타트업을 탐색하고, 시장성·경쟁사 등을 분석하여 투자 적합성을 판단합니다.

## Overview

- Objective : 모빌리티 스타트업의 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석
- Method : AI Agent + Agentic RAG
- Tools : 시장성 분석 에이전트, 경쟁사 분석 에이전트, 투자 판단 에이전트, 보고서 생성 에이전트

## Features

- ✅ **모빌리티 스타트업 자동 탐색**  
  NextUnicorn 크롤링을 통해 모빌리티 스타트업 리스트 자동 확보
  
- ✅ **시장성 분석**  
  해당 스타트업의 산업 동향과 스타트업의 동향을 비교해 시장성 분석
  
- ✅ **경쟁사 분석**  
  유사 모빌리티 스타트업 경쟁사 3곳을 선정하여 경쟁사 분석
  
- ✅ **투자 판단**  
  시장성과 경쟁사 분석 결과를 종합하여 투자 여부 결정

- ✅ **보고서 생성**  
  투자 전문가 관점에서 기업별 보고서 작성

## Tech Stack 

| Category     | Details                                   |
|--------------|-------------------------------------------|
| Framework    | **LangGraph**, LangChain                  |
| LLM          | GPT-4o-mini (OpenAI API)                  |
| Embedding model | GPT-4o-mini (OpenAI API)                  |
| VDB | **Chroma**                   |
| Crawling     | **Playwright** (NextUnicorn 크롤러)        |

## Agents
 
- **startup_search_agent** : NextUnicorn 크롤링 → LLM 요약/태깅 → Chroma 저장
- **industry_search_agent** : Tavily Search → 검색 결과 전처리 → Chroma 저장
- **market_eval_agent** : 산업 분석 내용 기반으로 스타트업의 시장성 분석
- **competitor_analysis_agent** : 경쟁사 Top-3 검색 → 비교 분석 보고서 생성  
- **investment_decision_agent** : 투자 여부(True/False) 판단 / (창업자 역량, 시장성, 제품력, 경쟁력, 실적, 투자 조건, 리스크)
- **report_writer_agent** : 투자 요약 보고서 생성

## Architecture
<img width="343" height="735" alt="image" src="https://github.com/user-attachments/assets/4581dffd-1db4-4799-9b8c-1e38760a1699" />

## Directory Structure
<img width="428" height="339" alt="image" src="https://github.com/user-attachments/assets/3a8e991a-5ba7-47b8-ae98-11440949efe1" />

## Contributors 
- 고은렬 : 시장성 평가 에이전트 개발
- 고서아 : 경쟁사 분석 에이전트 개발
- 신수민 : 워크플로우 정의(Graph, State 설계), VDB 세팅
- 신순호 : 크롤링 + 기업 분석 에이전트 개발
- 우찬민 : 투자 여부 판단 에이전트, 투자 요약 보고서 에이전트 개발
