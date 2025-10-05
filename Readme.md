# Mobility Startup Investment Evaluation Agent
**모빌리티 스타트업**에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 프로젝트입니다. 
NextUnicorn에서 모빌리티 분야 스타트업을 탐색하고, 시장성·경쟁사 등을 분석하여 투자 적합성을 판단하여 보고서를 작성합니다.

## Overview

- **User Scenario** :  
  투자자가 특정 도메인(예: 모빌리티)에 대한 스타트업을 조사하고, 투자 판단 보고서를 작성하는 상황을 가정했습니다.  
  본 프로젝트는 투자자가 직접 스타트업 목록을 조사하고, 분석하고, 보고서를 작성하는 과정을 **에이전트 워크플로우로 자동화**합니다.  

- **Key Differentiators** :  
  - **실시간 스타트업 발굴**: NextUnicorn 플랫폼에서 실시간으로 스타트업을 크롤링하여 최신 기업 정보 자동 수집
  - **자동 분석 대상 선정**: 수집된 스타트업 중 투자 분석 대상을 AI가 자동으로 선별 및 태깅
  - **동적 데이터 반영**: 고정된 기업 리스트가 아닌, NextUnicorn에 새로 등록되는 기업들을 즉시 반영하여 분석 범위 확장
  - **End-to-End 자동화**: 기업 발굴부터 투자 판단, 보고서 생성까지 전 과정을 AI 에이전트가 자동 수행

- **Objective** : 모빌리티 스타트업의 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석  
- **Method** : AI Agent + Agentic RAG + Real-time Web Crawling

## Features

- **모빌리티 스타트업 자동 탐색**  
  NextUnicorn 크롤링을 통해 모빌리티 스타트업 리스트 자동 확보
  
- **시장성 분석**  
  산업 동향과 해당 스타트업의 동향을 비교해 시장성 분석
  
- **경쟁사 분석**  
  유사 모빌리티 스타트업 경쟁사 3곳을 선정하여 경쟁사 분석
  
- **투자 판단**  
  시장성과 경쟁사 분석 결과를 종합하여 투자 여부 결정

- **보고서 생성**  
  투자 전문가 관점에서 기업별 보고서 작성

## Tech Stack 

| Category     | Details                                   |
|--------------|-------------------------------------------|
| Framework    | **LangGraph**, LangChain                  |
| LLM          | GPT-4o-mini (OpenAI API)                  |
| Embedding model | BAAI/bge-m3 (HuggingFace)                 |
| VDB | **Chroma**                   |
| Crawling     | **Playwright** (NextUnicorn 크롤러)        |

## RAG Metadata

- name: 기업명
- kind: 기업 분석/산업 분석 구분

## Agents Workflow

#### 1. **startup_search_agent** - 스타트업 발굴 및 데이터 수집
- **Step 1**: NextUnicorn 플랫폼에서 Playwright를 이용한 실시간 크롤링
- **Step 2**: 수집된 기업 정보를 LLM에 전달하여 자동 태깅 및 정제
- **Step 3**: 처리된 데이터를 Chroma VectorDB에 임베딩 및 저장
- **Step 4**: 분석 대상 기업 10개 자동 선정

#### 2. **industry_search_agent** - 산업 동향 데이터 수집
- **Step 1**: 모빌리티 도메인별 검색 쿼리 자동 생성 (전기차, 자율주행, 전동킥보드 등)
- **Step 2**: Tavily Search API를 통한 실시간 산업 동향 수집
- **Step 3**: 수집된 산업 데이터를 청크 단위로 분할 후 VectorDB 저장

#### 3. **market_eval_agent** - 시장성 평가
- **Step 1**: VectorDB에서 해당 기업 관련 산업 동향 데이터 검색
- **Step 2**: 기업 정보와 산업 동향을 비교 분석하여 시장 규모, 성장 가능성, 고객 수요 평가
- **Step 3**: 점수 기반 시장성 평가 결과 생성

#### 4. **competitor_analysis_agent** - 경쟁사 분석
- **Step 1**: VectorDB에서 벡터 유사도 기반 경쟁사 Top-3 자동 선정
- **Step 2**: 경쟁사별 비즈니스 모델, 수익 구조, 경쟁 우위 분석
- **Step 3**: 시장 내 포지셔닝 및 경쟁 전략 제안 보고서 생성

#### 5. **investment_decision_agent** - 투자 의사결정
- **Step 1**: 7가지 평가 기준으로 종합 점수 산정 (창업자, 시장성, 제품기술력, 경쟁우위, 실적, 투자조건, 리스크)
- **Step 2**: 점수 기반 투자 여부 결정 (True/False)
- **Step 3**: 투자 거부 시 다음 기업으로 이동, 승인 시 보고서 작성 단계로 진행

#### 6. **report_writer_agent** - 보고서 생성
- **Step 1**: 투자 승인 기업 대상 상세 투자 보고서 작성
- **Step 2**: 모든 기업 거부 시 종합 거부 사유 분석 보고서 작성
- **Step 3**: 레이더 차트 생성 및 PDF 보고서 출력

## State Management

- List[str]: 선정 기업 10개 기업명 목록 (분석 완료 시 pop)
- str: 현재 분석 중인 기업명
- List[str]: 현재 기업의 태깅 목록
- str: 시장성 분석 내용
- str: 경쟁사 분석 내용
- bool: 보고서 작성 여부 (최소 1개 이상 보고서 생성 시 True)
- bool: 투자 여부 판단 결과 ( True / False )

## Architecture
<img width="479" height="768" alt="image" src="https://github.com/user-attachments/assets/af539e15-9516-4e3a-9260-b9a4451c6259" />

## Directory Structure
<img width="428" height="339" alt="image" src="https://github.com/user-attachments/assets/3a8e991a-5ba7-47b8-ae98-11440949efe1" />

## Contributors 
- **고은렬** : 시장성 평가 에이전트 개발 (산업 동향 기반 시장 규모/성장성/수요 분석 로직)
- **고서아** : 경쟁사 분석 에이전트 개발 (벡터 유사도 기반 경쟁사 선정 및 비교 분석)
- **신수민** : LangGraph 워크플로우 설계 (State 관리, 노드 연결, 라우팅 로직), Chroma VectorDB 설정, 경쟁사 분석 에이전트 개발 보조
- **신순호** : NextUnicorn 크롤링 개발 (Playwright 기반), 스타트업 데이터 수집 및 전처리 에이전트
- **우찬민** : 투자 의사결정 에이전트 (7가지 평가 기준 점수화), PDF 보고서 생성 에이전트 (레이더차트 포함)
- **이재휘** : 산업 동향 수집 에이전트 개발 (Tavily Search API 활용, 도메인별 쿼리 생성)

## Future Work & Scalability
- 모빌리티 외 헬스케어, 핀테크, 에듀테크 등으로 도메인 확장 가능  
- IR PDF, 뉴스 기사, 정부 공공데이터 등 입력 확장
