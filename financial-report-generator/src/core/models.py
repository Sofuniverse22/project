"""
핵심 데이터 모델
Pydantic 모델을 사용하여 타입 안정성 확보
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# === Enums ===

class DocumentType(str, Enum):
    """문서 타입"""
    BUSINESS_REPORT = "business_report"  # 사업보고서
    AUDIT_REPORT = "audit_report"  # 감사보고서
    QUARTERLY_REPORT = "quarterly_report"  # 분기보고서
    NEWS_ARTICLE = "news_article"  # 뉴스 기사
    INDUSTRY_REPORT = "industry_report"  # 산업 리포트
    IR_MATERIAL = "ir_material"  # IR 자료
    UNKNOWN = "unknown"  # 알 수 없음


class AnalysisDepth(str, Enum):
    """분석 깊이"""
    QUICK = "quick"  # 간략
    STANDARD = "standard"  # 표준
    DEEP = "deep"  # 심층


class InvestmentRecommendation(str, Enum):
    """투자 의견"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class Sentiment(str, Enum):
    """감성 (긍정/중립/부정)"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# === 문서 관련 모델 ===

class ParsedDocument(BaseModel):
    """파싱된 문서 데이터"""
    doc_id: str = Field(..., description="문서 고유 ID")
    doc_type: DocumentType = Field(..., description="문서 타입")
    company_name: str = Field(..., description="기업명")
    report_date: Optional[str] = Field(None, description="보고서 날짜 (YYYY-MM-DD)")

    # 재무 데이터
    financial_statements: Dict[str, Any] = Field(
        default_factory=dict,
        description="재무제표 (balance_sheet, income_statement, cash_flow)"
    )

    # 텍스트 데이터
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="섹션별 텍스트 (회사개요, 사업내용, 리스크 등)"
    )

    # 메타데이터
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="메타데이터 (파일명, 페이지 수, 언어 등)"
    )

    # 추출된 표
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="문서에서 추출된 모든 표"
    )

    # 원본 파일 정보
    source_file: str = Field(..., description="원본 파일 경로")
    parsing_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="파싱 시각"
    )


# === 재무 분석 모델 ===

class FinancialRatios(BaseModel):
    """재무 비율"""

    # 수익성 지표
    roe: Optional[float] = Field(None, description="자기자본이익률 (%)")
    roa: Optional[float] = Field(None, description="총자산이익률 (%)")
    operating_margin: Optional[float] = Field(None, description="영업이익률 (%)")
    net_margin: Optional[float] = Field(None, description="순이익률 (%)")
    ebitda_margin: Optional[float] = Field(None, description="EBITDA 마진 (%)")
    roic: Optional[float] = Field(None, description="투하자본이익률 (%)")

    # 안정성 지표
    debt_ratio: Optional[float] = Field(None, description="부채비율 (%)")
    current_ratio: Optional[float] = Field(None, description="유동비율 (%)")
    quick_ratio: Optional[float] = Field(None, description="당좌비율 (%)")
    interest_coverage: Optional[float] = Field(None, description="이자보상배율 (배)")
    debt_to_assets: Optional[float] = Field(None, description="차입금의존도 (%)")
    equity_ratio: Optional[float] = Field(None, description="자기자본비율 (%)")

    # 성장성 지표
    revenue_growth_yoy: Optional[float] = Field(None, description="매출 성장률 YoY (%)")
    operating_income_growth_yoy: Optional[float] = Field(None, description="영업이익 성장률 YoY (%)")
    net_income_growth_yoy: Optional[float] = Field(None, description="순이익 성장률 YoY (%)")
    assets_growth_yoy: Optional[float] = Field(None, description="총자산 성장률 YoY (%)")

    # 활동성 지표
    asset_turnover: Optional[float] = Field(None, description="총자산회전율 (회)")
    inventory_turnover: Optional[float] = Field(None, description="재고자산회전율 (회)")
    receivables_turnover: Optional[float] = Field(None, description="매출채권회전율 (회)")

    # 밸류에이션 지표
    per: Optional[float] = Field(None, description="주가수익비율 (배)")
    pbr: Optional[float] = Field(None, description="주가순자산비율 (배)")
    psr: Optional[float] = Field(None, description="주가매출액비율 (배)")
    ev_ebitda: Optional[float] = Field(None, description="EV/EBITDA (배)")


class TrendAnalysis(BaseModel):
    """트렌드 분석 결과"""
    metric_name: str = Field(..., description="지표명")
    values: List[float] = Field(..., description="시계열 값")
    periods: List[str] = Field(..., description="기간 (YYYY-MM-DD)")

    trend: str = Field(..., description="추세 (증가/감소/횡보)")
    cagr: Optional[float] = Field(None, description="연평균 성장률 (%)")

    # 통계 정보
    mean: float = Field(..., description="평균")
    std: float = Field(..., description="표준편차")
    min_value: float = Field(..., description="최솟값")
    max_value: float = Field(..., description="최댓값")

    # 이상치
    anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="이상치 목록 (기간, 값, 편차)"
    )


class PeerComparison(BaseModel):
    """경쟁사 비교 분석"""
    company_name: str = Field(..., description="기업명")
    peer_companies: List[str] = Field(..., description="비교 대상 기업 목록")

    # 비교 지표
    comparison_metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="비교 지표 {metric_name: {company: value}}"
    )

    # 상대적 포지션
    percentile_rank: Dict[str, float] = Field(
        ...,
        description="백분위 순위 {metric_name: percentile}"
    )

    # 강점/약점
    strengths: List[str] = Field(default_factory=list, description="경쟁사 대비 강점")
    weaknesses: List[str] = Field(default_factory=list, description="경쟁사 대비 약점")


class FinancialMetrics(BaseModel):
    """종합 재무 분석 결과"""
    company_name: str = Field(..., description="기업명")
    analysis_date: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="분석 일시"
    )

    # 재무 비율
    ratios: FinancialRatios = Field(..., description="재무 비율")

    # 트렌드 분석
    trends: List[TrendAnalysis] = Field(
        default_factory=list,
        description="트렌드 분석 결과"
    )

    # 경쟁사 비교
    peer_comparison: Optional[PeerComparison] = Field(
        None,
        description="경쟁사 비교 분석"
    )

    # 종합 점수
    overall_score: Dict[str, float] = Field(
        ...,
        description="카테고리별 점수 (수익성, 안정성, 성장성, 활동성, 종합)"
    )

    # 추천 사항
    recommendations: List[str] = Field(
        default_factory=list,
        description="개선 추천 사항"
    )


# === AI 분석 모델 ===

class BusinessAnalysis(BaseModel):
    """사업 모델 분석"""
    revenue_model: str = Field(..., description="수익 구조")
    customer_segments: List[str] = Field(..., description="고객 세그먼트")
    competitive_moat: str = Field(..., description="경쟁 우위 (Moat)")
    key_risks: List[str] = Field(..., description="비즈니스 리스크")
    growth_drivers: List[str] = Field(..., description="성장 동력")


class IndustryAnalysis(BaseModel):
    """산업 분석"""
    industry_name: str = Field(..., description="산업명")
    market_size: Optional[str] = Field(None, description="시장 규모")
    growth_rate: Optional[float] = Field(None, description="산업 성장률 (%)")

    # Porter's 5 Forces
    competitive_rivalry: str = Field(..., description="경쟁 강도")
    supplier_power: str = Field(..., description="공급자 교섭력")
    buyer_power: str = Field(..., description="구매자 교섭력")
    threat_of_substitutes: str = Field(..., description="대체재 위협")
    threat_of_new_entrants: str = Field(..., description="신규 진입 위협")

    # 산업 전망
    outlook: str = Field(..., description="산업 전망")
    key_trends: List[str] = Field(..., description="주요 트렌드")


class SWOTAnalysis(BaseModel):
    """SWOT 분석"""
    strengths: List[str] = Field(..., description="강점 (Strengths)")
    weaknesses: List[str] = Field(..., description="약점 (Weaknesses)")
    opportunities: List[str] = Field(..., description="기회 (Opportunities)")
    threats: List[str] = Field(..., description="위협 (Threats)")


class SentimentAnalysis(BaseModel):
    """뉴스 감성 분석"""
    overall_sentiment: Sentiment = Field(..., description="전체 감성")
    sentiment_score: float = Field(
        ...,
        ge=-10,
        le=10,
        description="감성 점수 (-10 ~ +10)"
    )

    key_themes: List[Dict[str, Any]] = Field(
        ...,
        description="주요 테마 (theme, sentiment, impact)"
    )

    major_events: List[Dict[str, Any]] = Field(
        ...,
        description="주요 이벤트 (date, event, impact)"
    )

    market_reaction: str = Field(..., description="시장 반응 예측")


class InvestmentThesis(BaseModel):
    """투자 논리"""
    recommendation: InvestmentRecommendation = Field(..., description="투자 의견")
    target_price: Optional[float] = Field(None, description="목표 주가")
    upside_potential: Optional[float] = Field(None, description="상승 여력 (%)")

    # 시나리오 분석
    bull_case: str = Field(..., description="낙관 시나리오")
    base_case: str = Field(..., description="기본 시나리오")
    bear_case: str = Field(..., description="비관 시나리오")

    # 핵심 포인트
    key_investment_points: List[str] = Field(
        ...,
        description="핵심 투자 포인트"
    )

    key_risks: List[str] = Field(..., description="주요 리스크 요인")

    # 밸류에이션
    valuation_assessment: str = Field(..., description="밸류에이션 평가")


class AIInsights(BaseModel):
    """AI 기반 종합 인사이트"""
    executive_summary: str = Field(..., description="Executive Summary")
    business_analysis: BusinessAnalysis = Field(..., description="사업 모델 분석")
    industry_analysis: IndustryAnalysis = Field(..., description="산업 분석")
    swot_analysis: SWOTAnalysis = Field(..., description="SWOT 분석")
    sentiment_analysis: Optional[SentimentAnalysis] = Field(
        None,
        description="뉴스 감성 분석"
    )
    investment_thesis: InvestmentThesis = Field(..., description="투자 논리")

    # ESG (선택적)
    esg_score: Optional[Dict[str, float]] = Field(
        None,
        description="ESG 점수 (E, S, G)"
    )


# === 분석 설정 모델 ===

class AnalysisConfig(BaseModel):
    """분석 설정"""
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.STANDARD,
        description="분석 깊이"
    )

    include_peer_analysis: bool = Field(
        default=True,
        description="경쟁사 비교 분석 포함 여부"
    )

    peer_companies: List[str] = Field(
        default_factory=list,
        description="비교 대상 기업 목록"
    )

    include_sentiment_analysis: bool = Field(
        default=True,
        description="뉴스 감성 분석 포함 여부"
    )

    include_esg: bool = Field(
        default=False,
        description="ESG 분석 포함 여부"
    )

    report_language: str = Field(
        default="ko",
        description="보고서 언어 (ko, en)"
    )

    template: str = Field(
        default="default",
        description="보고서 템플릿"
    )

    @field_validator('peer_companies')
    @classmethod
    def validate_peer_companies(cls, v):
        if len(v) > 10:
            raise ValueError("비교 대상 기업은 최대 10개까지 가능합니다.")
        return v


# === 보고서 모델 ===

class ReportSection(BaseModel):
    """보고서 섹션"""
    section_name: str = Field(..., description="섹션명")
    content: str = Field(..., description="내용")
    charts: List[str] = Field(
        default_factory=list,
        description="차트 파일 경로 목록"
    )
    order: int = Field(..., description="순서")


class GeneratedReport(BaseModel):
    """생성된 보고서"""
    report_id: str = Field(..., description="보고서 ID")
    company_name: str = Field(..., description="기업명")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="생성 시각"
    )

    # 섹션 목록
    sections: List[ReportSection] = Field(..., description="보고서 섹션 목록")

    # 파일 정보
    pdf_path: str = Field(..., description="PDF 파일 경로")
    file_size_mb: float = Field(..., description="파일 크기 (MB)")

    # 메타데이터
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="메타데이터"
    )


# === 작업 상태 모델 ===

class TaskStatus(str, Enum):
    """작업 상태"""
    QUEUED = "queued"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisTask(BaseModel):
    """분석 작업"""
    task_id: str = Field(..., description="작업 ID")
    status: TaskStatus = Field(default=TaskStatus.QUEUED, description="작업 상태")
    progress: int = Field(default=0, ge=0, le=100, description="진행률 (%)")

    # 시간 정보
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="생성 시각"
    )
    started_at: Optional[str] = Field(None, description="시작 시각")
    completed_at: Optional[str] = Field(None, description="완료 시각")

    # 설정 및 결과
    config: AnalysisConfig = Field(..., description="분석 설정")
    result_path: Optional[str] = Field(None, description="결과 파일 경로")
    error_message: Optional[str] = Field(None, description="에러 메시지")

    # 현재 단계
    current_step: str = Field(default="대기 중", description="현재 진행 단계")
    estimated_remaining_seconds: Optional[int] = Field(
        None,
        description="예상 남은 시간 (초)"
    )
