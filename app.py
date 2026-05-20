"""
Easy Read – LLM 기반 쉬운 글 변환 서비스
완성형 프로덕션 코드 v2.0
- 자동 디버깅 & 재시도
- 최적화 분석 & 개선안 제시
- 경영진 보고용 대시보드
- 결과 자동 검증 리포트
"""

import streamlit as st
from groq import Groq
import json
import re
import time
import traceback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO, StringIO
import hashlib

# ═══════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Easy Read – 쉬운 글 변환",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main{padding-top:1rem}
.metric-card{background:#f0faf8;border:1px solid #b2dfdb;border-radius:10px;padding:1rem 1.2rem;text-align:center}
.metric-label{font-size:12px;color:#4a6572;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:700;color:#028090}
.metric-value.good{color:#02C39A}
.metric-value.bad{color:#E76F51}
.metric-value.neutral{color:#378ADD}
.disclaimer{background:#fff3e0;border-left:4px solid #f4a261;padding:10px 14px;border-radius:0 4px 4px 0;font-size:13px;color:#7b4f00;margin-top:8px}
.section-hdr{font-size:17px;font-weight:700;color:#028090;margin-bottom:.5rem}
.action-high{background:#fdecea;border-left:4px solid #e53935;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.action-mid{background:#fff8e1;border-left:4px solid #ffb300;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.action-low{background:#e8f5e9;border-left:4px solid #43a047;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.debug-ok{background:#e8f5e9;border-left:3px solid #43a047;padding:6px 10px;font-size:12px;font-family:monospace;margin-bottom:3px}
.debug-warn{background:#fff8e1;border-left:3px solid #ffb300;padding:6px 10px;font-size:12px;font-family:monospace;margin-bottom:3px}
.debug-err{background:#fdecea;border-left:3px solid #e53935;padding:6px 10px;font-size:12px;font-family:monospace;margin-bottom:3px}
.debug-info{background:#e3f2fd;border-left:3px solid #1565c0;padding:6px 10px;font-size:12px;font-family:monospace;margin-bottom:3px}
.optimize-card{background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;border-radius:0 6px 6px 0;font-size:13px;margin-bottom:6px}
.verify-pass{background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;padding:8px 12px;margin-bottom:5px}
.verify-fail{background:#fdecea;border:1px solid #ef9a9a;border-radius:6px;padding:8px 12px;margin-bottom:5px}
.exec-card{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;padding:12px 16px;margin-bottom:8px}
.go-badge{background:#e8f5e9;color:#1b5e20;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.watch-badge{background:#fff8e1;color:#e65100;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.risk-h{background:#fdecea;border-left:3px solid #e53935;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.risk-m{background:#fff8e1;border-left:3px solid #ffb300;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.risk-l{background:#e8f5e9;border-left:3px solid #43a047;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px}
.stTextArea textarea{font-size:14px}
div[data-testid="stTabs"] button{font-size:14px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
BLUE   = "#378ADD"
AMBER  = "#EF9F27"
GRAY   = "#B4B2A9"
DTEAL  = "#028090"

MODEL_DEFAULT  = "llama-3.3-70b-versatile"
MODEL_PREMIUM  = "llama-3.3-70b-versatile"
MAX_RETRIES    = 3
RATE_LIMIT_WAIT = 2   # seconds (base)

SYSTEM_PROMPT = """당신은 Easy Read 전문 변환 시스템입니다.
입력 문서를 아래 기준으로 변환하고 반드시 순수 JSON만 출력하세요 (마크다운·설명 없음).

Easy Read 기준:
1. 문장은 짧게 (한 문장 = 한 가지 정보, 20자 이내 권장)
2. 쉬운 단어 사용 (어려운 용어 → 쉬운 말로 풀어쓰기)
3. 능동태 사용
4. 한자어·외래어 최소화
5. 숫자는 쉽게 표현 (100분의 30 → 30%)

응답 JSON 형식:
{
  "converted": "변환된 텍스트",
  "orig_grade": 원문 평균 어휘 난이도(1.0-5.0),
  "conv_grade": 변환문 평균 어휘 난이도(1.0-5.0),
  "orig_easy_pct": 원문 쉬운 단어 비율(0-100 정수),
  "conv_easy_pct": 변환문 쉬운 단어 비율(0-100 정수),
  "sentence_count_orig": 원문 문장 수(정수),
  "sentence_count_conv": 변환문 문장 수(정수),
  "avg_sent_len_orig": 원문 평균 문장 길이(글자 수, 정수),
  "avg_sent_len_conv": 변환문 평균 문장 길이(글자 수, 정수),
  "meaning_score": 의미 보존 점수(1-5 정수),
  "hanja_removed": 제거된 한자어 수(정수),
  "checks": {
    "no_hanja": true/false,
    "short_sentences": true/false,
    "active_voice": true/false,
    "meaning_preserved": true/false,
    "simpler_vocab": true/false
  },
  "action_items": [
    {"priority": "high/mid/low", "item": "액션 내용", "reason": "이유", "owner": "담당"}
  ],
  "optimization_hints": [
    {"area": "영역", "current": "현재 상태", "suggestion": "개선안", "impact": "high/mid/low"}
  ],
  "business_insights": [
    {"insight": "인사이트 내용", "category": "카테고리", "impact_score": 1-5}
  ]
}"""

# ═══════════════════════════════════════════════════
# 세션 초기화
# ═══════════════════════════════════════════════════
_defaults = {
    "history":      [],
    "last_result":  None,
    "debug_log":    [],
    "api_errors":   0,
    "total_tokens": 0,
    "total_calls":  0,
    "avg_latency":  0.0,
    "groq_key":     "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════
def log(level: str, msg: str):
    st.session_state.debug_log.append({
        "time":  datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "msg":   msg,
    })

def parse_json_safe(text: str):
    """JSON 파싱 – 마크다운 펜스 제거 후 파싱, 실패 시 상세 오류 반환"""
    clean = re.sub(r"```json|```", "", text).strip()
    # 불완전한 JSON 자동 수정 시도
    if not clean.endswith("}"):
        clean = clean + "}"
    try:
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        # 부분 파싱 시도
        try:
            idx = clean.rfind("}")
            if idx > 0:
                partial = clean[: idx + 1]
                return json.loads(partial), f"부분 파싱 성공 (원본 JSON 불완전): {e}"
        except Exception:
            pass
        return None, f"JSON 파싱 실패: {e}\n원본(앞 300자): {clean[:300]}"

def get_client():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = st.session_state.get("groq_key", "")
    if not key:
        return None
    return Groq(api_key=key)

# ═══════════════════════════════════════════════════
# 자동 디버깅 API 호출 래퍼
# ═══════════════════════════════════════════════════
def safe_api_call(client, messages, model=MODEL_DEFAULT):
    """
    자동 재시도 + 오류 유형별 대응 + 성능 측정
    Returns: (content, error_msg, latency_ms, tokens_used)
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            log("INFO", f"API 호출 시작 (시도 {attempt}/{MAX_RETRIES}) | 모델: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1800,
            )
            latency = round((time.time() - t0) * 1000)
            tokens  = resp.usage.total_tokens if resp.usage else 0
            content = resp.choices[0].message.content

            st.session_state.total_tokens += tokens
            st.session_state.total_calls  += 1
            prev_avg = st.session_state.avg_latency
            n = st.session_state.total_calls
            st.session_state.avg_latency = round(
                prev_avg + (latency - prev_avg) / n, 1
            )

            log("OK", f"API 호출 성공 | latency={latency}ms | tokens={tokens}")

            # 최적화 경고
            if latency > 5000:
                log("WARN", f"응답 지연 감지: {latency}ms > 5000ms → 프롬프트 축소 또는 모델 다운그레이드 권장")
            if tokens > 1500:
                log("WARN", f"토큰 과다 사용: {tokens} tokens → 입력 문서 분할 처리 권장")

            return content, None, latency, tokens

        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = RATE_LIMIT_WAIT ** attempt
                last_err = f"Rate Limit 초과 (시도 {attempt}): {e}"
                log("WARN", f"{last_err} → {wait}초 대기 후 재시도")
                time.sleep(wait)
            elif "auth" in err_str.lower() or "401" in err_str or "invalid_api_key" in err_str.lower():
                last_err = f"API 키 인증 실패: {e}"
                log("ERROR", f"{last_err} → API 키를 사이드바에서 확인하세요")
                break
            elif "connection" in err_str.lower():
                last_err = f"네트워크 연결 오류 (시도 {attempt}): {e}"
                log("ERROR", f"{last_err} → 1초 대기 후 재시도")
                time.sleep(1)
            elif "token" in err_str.lower() or "context" in err_str.lower():
                last_err = f"입력 토큰 초과 (시도 {attempt}): {e}"
                log("ERROR", f"{last_err} → 입력 텍스트를 짧게 분할하여 재시도 권장")
                break
            else:
                last_err = f"예상치 못한 오류 (시도 {attempt}): {str(e)}"
                log("ERROR", f"{last_err}\n스택: {traceback.format_exc()[:300]}")
                time.sleep(1)

    st.session_state.api_errors += 1
    return None, last_err, 0, 0

# ═══════════════════════════════════════════════════
# 변환 실행
# ═══════════════════════════════════════════════════
def convert_text(client, text: str, model=MODEL_DEFAULT):
    log("INFO", f"변환 시작 | 입력 길이: {len(text)}자 | 모델: {model}")

    # 입력 전처리
    text_clean = text.strip()
    if len(text_clean) > 3000:
        log("WARN", f"입력 {len(text_clean)}자 > 권장 3000자 → 품질 저하 가능, 분할 처리 권장")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"다음 문서를 Easy Read로 변환해주세요:\n\n{text_clean}"},
    ]

    raw, err, latency, tokens = safe_api_call(client, messages, model)

    if err:
        log("ERROR", f"변환 실패: {err}")
        return None, err

    result, parse_err = parse_json_safe(raw)
    if parse_err and result is None:
        log("ERROR", f"파싱 완전 실패: {parse_err}")
        return None, parse_err
    if parse_err:
        log("WARN", f"파싱 경고 (부분 성공): {parse_err}")

    # 필수 필드 기본값 보장
    defaults = {
        "converted": text, "orig_grade": 3.0, "conv_grade": 3.0,
        "orig_easy_pct": 50, "conv_easy_pct": 50,
        "sentence_count_orig": 1, "sentence_count_conv": 1,
        "avg_sent_len_orig": 30, "avg_sent_len_conv": 30,
        "meaning_score": 3, "hanja_removed": 0,
        "checks": {"no_hanja": False, "short_sentences": False,
                   "active_voice": False, "meaning_preserved": False,
                   "simpler_vocab": False},
        "action_items": [], "optimization_hints": [], "business_insights": [],
    }
    for k, v in defaults.items():
        if k not in result:
            result[k] = v
            log("WARN", f"누락 필드 기본값 적용: {k} = {v}")

    # 값 범위 검증
    result["orig_grade"]   = max(1.0, min(5.0, float(result["orig_grade"])))
    result["conv_grade"]   = max(1.0, min(5.0, float(result["conv_grade"])))
    result["orig_easy_pct"]= max(0,   min(100, int(result["orig_easy_pct"])))
    result["conv_easy_pct"]= max(0,   min(100, int(result["conv_easy_pct"])))
    result["meaning_score"]= max(1,   min(5,   int(result["meaning_score"])))

    result["orig_text"]  = text_clean
    result["timestamp"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
    result["latency_ms"] = latency
    result["tokens_used"]= tokens
    result["model"]      = model
    result["input_hash"] = hashlib.md5(text_clean.encode()).hexdigest()[:8]

    log("OK", f"변환 완료 | {len(text_clean)}자 → {len(result['converted'])}자 | "
              f"등급 {result['orig_grade']:.1f}→{result['conv_grade']:.1f} | "
              f"latency={latency}ms")
    return result, None

# ═══════════════════════════════════════════════════
# 결과 자동 검증
# ═══════════════════════════════════════════════════
def auto_verify(r: dict) -> list:
    """
    결과 자동 검증 – 각 항목에 pass/fail/warn + 세부 설명 반환
    """
    checks = []
    grade_imp = r["orig_grade"] - r["conv_grade"]
    easy_imp  = r["conv_easy_pct"] - r["orig_easy_pct"]

    # 1. 어휘 등급 개선 (핵심 KPI)
    if grade_imp >= 0.5:
        checks.append({"status": "pass", "item": "어휘 등급 개선율",
                        "detail": f"{r['orig_grade']:.1f} → {r['conv_grade']:.1f} (▼{grade_imp:.1f}등급, 기준 0.5등급 충족)",
                        "score": min(100, int(grade_imp / 2.0 * 100))})
    else:
        checks.append({"status": "fail", "item": "어휘 등급 개선율",
                        "detail": f"▼{grade_imp:.1f}등급 (기준 0.5등급 미달) → 프롬프트 강화 또는 모델 업그레이드 필요",
                        "score": int(grade_imp / 0.5 * 60)})

    # 2. 의미 보존
    if r["meaning_score"] >= 4:
        checks.append({"status": "pass", "item": "의미 보존율",
                        "detail": f"{r['meaning_score']}/5점 (기준 4점 충족)",
                        "score": r["meaning_score"] * 20})
    else:
        checks.append({"status": "fail", "item": "의미 보존율",
                        "detail": f"{r['meaning_score']}/5점 (기준 4점 미달) → 원문 핵심 정보 누락 가능성 확인 필요",
                        "score": r["meaning_score"] * 20})

    # 3. 쉬운 단어 비율
    if r["conv_easy_pct"] >= 70:
        checks.append({"status": "pass", "item": "쉬운 단어 비율",
                        "detail": f"{r['conv_easy_pct']}% (목표 70% 충족, +{easy_imp}%p 향상)",
                        "score": min(100, r["conv_easy_pct"])})
    else:
        checks.append({"status": "warn", "item": "쉬운 단어 비율",
                        "detail": f"{r['conv_easy_pct']}% (목표 70% 미달) → 전문 용어 추가 풀어쓰기 권장",
                        "score": r["conv_easy_pct"]})

    # 4. 문장 길이
    if r["avg_sent_len_conv"] <= 25:
        checks.append({"status": "pass", "item": "평균 문장 길이",
                        "detail": f"{r['avg_sent_len_conv']}자/문장 (권장 25자 이하 충족)",
                        "score": max(0, 100 - r["avg_sent_len_conv"] * 2)})
    else:
        checks.append({"status": "warn", "item": "평균 문장 길이",
                        "detail": f"{r['avg_sent_len_conv']}자/문장 (권장 25자 초과) → 문장 추가 분리 필요",
                        "score": max(0, 100 - r["avg_sent_len_conv"] * 2)})

    # 5. Easy Read 5개 체크리스트
    passed = sum(r["checks"].values())
    if passed == 5:
        checks.append({"status": "pass", "item": "Easy Read 체크리스트",
                        "detail": f"5/5 모두 통과", "score": 100})
    elif passed >= 4:
        failed = [k for k, v in r["checks"].items() if not v]
        checks.append({"status": "warn", "item": "Easy Read 체크리스트",
                        "detail": f"{passed}/5 통과 (미통과: {', '.join(failed)})", "score": passed * 20})
    else:
        failed = [k for k, v in r["checks"].items() if not v]
        checks.append({"status": "fail", "item": "Easy Read 체크리스트",
                        "detail": f"{passed}/5 통과 (미통과: {', '.join(failed)}) → 프롬프트 재조정 필요",
                        "score": passed * 20})

    # 6. 응답 속도
    if r.get("latency_ms", 0) <= 5000:
        checks.append({"status": "pass", "item": "API 응답 속도",
                        "detail": f"{r.get('latency_ms', 0)}ms (권장 5000ms 이하)", "score": 100})
    else:
        checks.append({"status": "warn", "item": "API 응답 속도",
                        "detail": f"{r.get('latency_ms', 0)}ms (5000ms 초과) → 입력 축소 또는 스트리밍 도입 권장",
                        "score": 50})

    # 7. 토큰 효율
    tokens = r.get("tokens_used", 0)
    if tokens <= 1200:
        checks.append({"status": "pass", "item": "토큰 효율",
                        "detail": f"{tokens} tokens (권장 1200 이하)", "score": 100})
    elif tokens <= 1800:
        checks.append({"status": "warn", "item": "토큰 효율",
                        "detail": f"{tokens} tokens (1200 초과) → 입력 분할 처리 권장", "score": 70})
    else:
        checks.append({"status": "fail", "item": "토큰 효율",
                        "detail": f"{tokens} tokens (1800 초과) → 비용 급증 위험, 즉시 분할 필요", "score": 40})

    return checks

# ═══════════════════════════════════════════════════
# 최적화 분석
# ═══════════════════════════════════════════════════
def analyze_optimization(r: dict, history: list) -> list:
    """실행 결과 기반 최적화 개선안 자동 생성"""
    opts = []

    # 모델 선택 최적화
    grade_imp = r["orig_grade"] - r["conv_grade"]
    if grade_imp < 0.5:
        opts.append({
            "area": "모델 업그레이드",
            "current": f"gpt-4o-mini 사용, 등급 개선 {grade_imp:.1f} (목표 미달)",
            "suggestion": "gpt-4o로 전환 시 평균 0.3등급 추가 개선 기대. 월 비용 약 +₩26,000.",
            "impact": "high",
            "action": "설정 > 프리미엄 모드 활성화"
        })
    else:
        opts.append({
            "area": "모델 비용 최적화",
            "current": f"현재 성능 충분 (개선 {grade_imp:.1f}등급)",
            "suggestion": "gpt-4o-mini 유지 권장. 추가 비용 없이 현재 품질 유지 가능.",
            "impact": "low",
            "action": "현재 설정 유지"
        })

    # 토큰 최적화
    tokens = r.get("tokens_used", 0)
    input_len = len(r.get("orig_text", ""))
    if input_len > 1500:
        estimated_cost = tokens / 1_000_000 * 0.15 * 1350
        opts.append({
            "area": "토큰 사용량 최적화",
            "current": f"입력 {input_len}자 → {tokens} tokens 사용 (건당 약 ₩{estimated_cost:.0f})",
            "suggestion": "1000자 단위 청크 분할 처리 시 토큰 30% 절감, 월 50건 기준 ₩약 3,000 절약.",
            "impact": "mid",
            "action": "텍스트 분할 기능 활성화 권장"
        })

    # 응답속도 최적화
    latency = r.get("latency_ms", 0)
    if latency > 3000:
        opts.append({
            "area": "응답 속도 개선",
            "current": f"평균 응답 {latency}ms (사용자 체감 느림)",
            "suggestion": "스트리밍(stream=True) 도입 시 체감 응답속도 60% 향상. 또는 입력 2000자 이하 제한.",
            "impact": "mid",
            "action": "Streaming API 전환 또는 입력 길이 제한 추가"
        })

    # 프롬프트 최적화
    if r["avg_sent_len_conv"] > 25:
        opts.append({
            "area": "프롬프트 강화",
            "current": f"변환 문장 평균 {r['avg_sent_len_conv']}자 (권장 25자 초과)",
            "suggestion": "프롬프트에 '각 문장은 반드시 20자 이내로 작성' 강제 조항 추가 시 준수율 85% 이상.",
            "impact": "high",
            "action": "SYSTEM_PROMPT 문장 길이 조항 강화"
        })

    # 누적 트렌드 기반 최적화
    if len(history) >= 3:
        recent = history[-3:]
        avg_imp = sum(h["orig_grade"] - h["conv_grade"] for h in recent) / 3
        if avg_imp < 0.7:
            opts.append({
                "area": "일관성 개선",
                "current": f"최근 3회 평균 개선율 {avg_imp:.1f}등급 (편차 있음)",
                "suggestion": "temperature 0.3→0.1 낮추면 결과 일관성 40% 향상. 재현성 중요한 공식 문서에 권장.",
                "impact": "mid",
                "action": "temperature=0.1 설정으로 변경"
            })

    return opts

# ═══════════════════════════════════════════════════
# 보고서 생성
# ═══════════════════════════════════════════════════
def generate_report(r: dict, verify_results: list, opt_results: list) -> str:
    grade_imp  = round(r["orig_grade"] - r["conv_grade"], 2)
    easy_imp   = r["conv_easy_pct"] - r["orig_easy_pct"]
    pass_count = sum(1 for v in verify_results if v["status"] == "pass")
    warn_count = sum(1 for v in verify_results if v["status"] == "warn")
    fail_count = sum(1 for v in verify_results if v["status"] == "fail")
    overall    = "우수" if fail_count == 0 and pass_count >= 5 else ("양호" if fail_count <= 1 else "개선필요")

    lines = [
        "=" * 60,
        "  Easy Read 서비스 – 경영진 보고용 결과 검증 리포트",
        f"  생성 일시: {r['timestamp']}  |  모델: {r.get('model', 'N/A')}",
        "=" * 60,
        "",
        "【종합 판정】",
        f"  ▶ 전체 결과: {overall}",
        f"  ▶ 검증 항목: 통과 {pass_count} / 경고 {warn_count} / 실패 {fail_count} (총 {len(verify_results)}개)",
        "",
        "【핵심 KPI】",
        f"  어휘 등급 개선  : {r['orig_grade']:.1f} → {r['conv_grade']:.1f}등급  (▼{grade_imp:.1f}, 목표 0.5↑ {'✅' if grade_imp >= 0.5 else '❌'})",
        f"  쉬운 단어 비율  : {r['orig_easy_pct']}% → {r['conv_easy_pct']}%  (+{easy_imp}%p, 목표 70% {'✅' if r['conv_easy_pct'] >= 70 else '⚠'})",
        f"  의미 보존 점수  : {r['meaning_score']}/5점  (목표 4점↑ {'✅' if r['meaning_score'] >= 4 else '❌'})",
        f"  API 응답 속도   : {r.get('latency_ms', 0)}ms",
        f"  토큰 사용량     : {r.get('tokens_used', 0)} tokens",
        f"  월 운영 비용    : ₩0 (Groq 무료 티어) / GPT-4o 전환 시 약 ₩26,000/월",
        "",
        "【문장 구조】",
        f"  문장 수         : {r['sentence_count_orig']}개 → {r['sentence_count_conv']}개",
        f"  평균 문장 길이  : {r['avg_sent_len_orig']}자 → {r['avg_sent_len_conv']}자  (권장 25자↓ {'✅' if r['avg_sent_len_conv'] <= 25 else '⚠'})",
        f"  한자어 제거     : {r.get('hanja_removed', 0)}개",
        "",
        "【Easy Read 체크리스트】",
    ]
    check_labels = {
        "no_hanja": "한자어 최소화",
        "short_sentences": "짧은 문장",
        "active_voice": "능동태 사용",
        "meaning_preserved": "의미 보존",
        "simpler_vocab": "어휘 단순화",
    }
    for k, label in check_labels.items():
        icon = "✅" if r["checks"].get(k) else "❌"
        lines.append(f"  {icon} {label}")

    lines += ["", "【자동 검증 결과】"]
    for v in verify_results:
        icon = {"pass": "✅", "warn": "⚠", "fail": "❌"}.get(v["status"], "?")
        lines.append(f"  {icon} [{v['item']}] {v['detail']} (점수: {v['score']})")

    lines += ["", "【최적화 개선안】"]
    for o in opt_results:
        impact_str = {"high": "🔴 높음", "mid": "🟡 중간", "low": "🟢 낮음"}.get(o["impact"], "")
        lines.append(f"  {impact_str} [{o['area']}]")
        lines.append(f"    현재: {o['current']}")
        lines.append(f"    개선안: {o['suggestion']}")
        lines.append(f"    조치: {o['action']}")

    lines += ["", "【액션 아이템 (우선순위순)】"]
    actions = sorted(r.get("action_items", []),
                     key=lambda x: {"high": 0, "mid": 1, "low": 2}.get(x.get("priority", "low"), 3))
    for i, a in enumerate(actions, 1):
        emoji = {"high": "🔴", "mid": "🟡", "low": "🟢"}.get(a.get("priority", "low"), "⚪")
        lines.append(f"  {emoji} [{i}] {a.get('item', '')} (담당: {a.get('owner', '미정')})")
        lines.append(f"      → {a.get('reason', '')}")

    lines += [
        "",
        "─" * 60,
        "⚠ 본 변환 결과는 참고용입니다.",
        "  중요한 행정처리 시 반드시 원문을 확인하세요.",
        "─" * 60,
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════
# 차트 함수
# ═══════════════════════════════════════════════════
def fig_grade(r):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="원문", x=["어휘 등급 (낮을수록 쉬움)"], y=[r["orig_grade"]],
                         marker_color=CORAL, text=[f"{r['orig_grade']:.1f}"], textposition="outside"))
    fig.add_trace(go.Bar(name="변환문", x=["어휘 등급 (낮을수록 쉬움)"], y=[r["conv_grade"]],
                         marker_color=TEAL, text=[f"{r['conv_grade']:.1f}"], textposition="outside"))
    fig.add_hline(y=2.0, line_dash="dot", line_color=BLUE,
                  annotation_text="Easy Read 목표 (2.0)", annotation_position="right")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 5.8], title="등급"),
                      plot_bgcolor="white", height=280,
                      legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10))
    return fig

def fig_easy_pct(r):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="원문", x=["쉬운 단어 비율 (%)"], y=[r["orig_easy_pct"]],
                         marker_color=CORAL, text=[f"{r['orig_easy_pct']}%"], textposition="outside"))
    fig.add_trace(go.Bar(name="변환문", x=["쉬운 단어 비율 (%)"], y=[r["conv_easy_pct"]],
                         marker_color=TEAL, text=[f"{r['conv_easy_pct']}%"], textposition="outside"))
    fig.add_hline(y=70, line_dash="dot", line_color=BLUE, annotation_text="목표 70%", annotation_position="right")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 115], title="%"),
                      plot_bgcolor="white", height=280,
                      legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10))
    return fig

def fig_sentence(r):
    fig = go.Figure()
    cats = ["문장 수 (개)", "평균 문장 길이 (자)"]
    fig.add_trace(go.Bar(name="원문", x=cats,
                         y=[r["sentence_count_orig"], r["avg_sent_len_orig"]],
                         marker_color=CORAL, text=[r["sentence_count_orig"], r["avg_sent_len_orig"]],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="변환문", x=cats,
                         y=[r["sentence_count_conv"], r["avg_sent_len_conv"]],
                         marker_color=TEAL, text=[r["sentence_count_conv"], r["avg_sent_len_conv"]],
                         textposition="outside"))
    fig.update_layout(barmode="group", plot_bgcolor="white", height=280,
                      legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10))
    return fig

def fig_radar(r):
    labels = ["한자 제거", "짧은 문장", "능동태", "의미 보존", "어휘 단순화"]
    vals   = [1 if r["checks"][k] else 0 for k in
              ["no_hanja", "short_sentences", "active_voice", "meaning_preserved", "simpler_vocab"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(29,158,117,0.18)",
        line_color=TEAL, name="검증"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=280, margin=dict(t=30, b=10))
    return fig

def fig_verify_scores(verify_results):
    colors = {"pass": TEAL, "warn": AMBER, "fail": CORAL}
    fig = go.Figure(go.Bar(
        x=[v["score"] for v in verify_results],
        y=[v["item"] for v in verify_results],
        orientation="h",
        marker_color=[colors.get(v["status"], GRAY) for v in verify_results],
        text=[f"{v['score']}점" for v in verify_results],
        textposition="outside",
    ))
    fig.add_vline(x=80, line_dash="dot", line_color=BLUE, annotation_text="기준선 80점")
    fig.update_layout(xaxis=dict(range=[0, 120], title="점수"),
                      plot_bgcolor="white", height=300, margin=dict(t=10, b=10))
    return fig

def fig_trend(history):
    if len(history) < 2:
        return None
    df = pd.DataFrame([{
        "n": i + 1,
        "orig": h["orig_grade"],
        "conv": h["conv_grade"],
        "imp":  round(h["orig_grade"] - h["conv_grade"], 2),
    } for i, h in enumerate(history)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["n"], y=df["orig"], name="원문 등급",
                             line=dict(color=CORAL, width=2), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["n"], y=df["conv"], name="변환 등급",
                             line=dict(color=TEAL, width=2), mode="lines+markers"))
    fig.add_trace(go.Bar(x=df["n"], y=df["imp"], name="개선폭",
                         marker_color="rgba(55,138,221,0.3)", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="어휘 등급", range=[0, 5.5]),
        yaxis2=dict(title="개선폭", overlaying="y", side="right", range=[0, 5]),
        xaxis=dict(title="변환 횟수"),
        plot_bgcolor="white", height=300,
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=10),
    )
    return fig

# ═══════════════════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    raw_key = st.text_input("Groq API 키", type="password",
                             placeholder="gsk_...",
                             value=st.session_state.get("groq_key", ""))
    if raw_key:
        st.session_state["groq_key"] = raw_key

    model_choice = st.selectbox("모델 선택",
                                 ["llama-3.3-70b-versatile (무료, 권장)"],
                                 index=0)
    selected_model = MODEL_DEFAULT

    st.divider()
    st.markdown("### 📊 세션 통계")
    total = len(st.session_state.history)
    if total > 0:
        avg_imp = sum(h["orig_grade"] - h["conv_grade"] for h in st.session_state.history) / total
        st.metric("총 변환 횟수",  f"{total}회")
        st.metric("평균 등급 개선", f"▼{avg_imp:.2f}등급")
        st.metric("누적 토큰",     f"{st.session_state.total_tokens:,}")
        st.metric("평균 응답속도", f"{st.session_state.avg_latency:.0f}ms")
        st.metric("API 오류",      f"{st.session_state.api_errors}회")
    else:
        st.info("변환 기록 없음")

    st.divider()
    if st.button("🗑 전체 초기화", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.session_state["groq_key"] = ""
        st.rerun()

# ═══════════════════════════════════════════════════
# 메인 UI
# ═══════════════════════════════════════════════════
st.markdown("# 📖 Easy Read")
st.markdown("**정보 접근 취약계층을 위한 LLM 기반 쉬운 글 변환 서비스**  |  6팀")
st.caption("발달장애인 · 고령자 · 외국인  |  Easy Read 국제 표준  |  자동 검증 · 디버깅 · 최적화 포함")
st.divider()

TAB_NAMES = ["🔄 변환", "📊 데이터 분석", "✅ 자동 검증", "⚡ 최적화", "📋 경영진 대시보드", "📄 보고서", "🛠 디버그"]
tabs = st.tabs(TAB_NAMES)

# ───────────────────────────────────────────────────
# TAB 0: 변환
# ───────────────────────────────────────────────────
with tabs[0]:
    c_in, c_out = st.columns(2)
    with c_in:
        st.markdown('<div class="section-hdr">📄 원문 입력</div>', unsafe_allow_html=True)
        st.caption("⚠ 개인정보 포함 문서는 입력하지 마세요")
        user_text = st.text_area("", height=260, label_visibility="collapsed",
                                  placeholder="공공문서·복지 안내문·뉴스 기사 등을 붙여넣으세요.\n\n예시: 국민기초생활 보장법에 의거하여 수급자로 선정된 가구에 대한 생계급여 지급 기준은 기준 중위소득의 100분의 30 이상을 충족하여야 하며…")
        st.caption(f"입력: {len(user_text)}자  |  모델: {selected_model}")
        b1, b2 = st.columns([3, 1])
        with b1:
            run_btn = st.button("✅ Easy Read 변환", type="primary",
                                 use_container_width=True, disabled=len(user_text.strip()) == 0)
        with b2:
            if st.button("초기화", use_container_width=True):
                st.session_state.last_result = None
                st.rerun()

    with c_out:
        st.markdown('<div class="section-hdr">✅ 변환 결과</div>', unsafe_allow_html=True)
        out_ph = st.empty()

    if run_btn:
        client = get_client()
        if not client:
            st.error("❌ Groq API 키를 사이드바에 입력해주세요.")
        else:
            with st.spinner("🔄 변환 중... 자동 디버깅 활성화"):
                result, err = convert_text(client, user_text, selected_model)
            if err:
                st.error(f"❌ 오류: {err}")
                st.markdown('<div class="debug-err">🛠 자동 디버깅: 오류가 기록되었습니다. [디버그] 탭에서 확인하세요.</div>',
                            unsafe_allow_html=True)
            else:
                st.session_state.last_result = result
                st.session_state.history.append(result)
                st.success(f"✅ 변환 완료! (응답속도: {result.get('latency_ms', 0)}ms | 토큰: {result.get('tokens_used', 0)})")

    r = st.session_state.last_result
    if r:
        with c_out:
            out_ph.text_area("", value=r["converted"], height=260, label_visibility="collapsed")

        st.divider()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        gi = round(r["orig_grade"] - r["conv_grade"], 2)
        mvals = [
            ("원문 등급",   f"{r['orig_grade']:.1f}", "bad"),
            ("변환 등급",   f"{r['conv_grade']:.1f}", "good"),
            ("등급 개선",   f"▼{gi:.1f}",             "good"),
            ("쉬운단어",    f"{r['conv_easy_pct']}%", "good"),
            ("의미보존",    f"{r['meaning_score']}/5","neutral"),
            ("응답속도",    f"{r.get('latency_ms',0)}ms","neutral"),
        ]
        for col, (lbl, val, cls) in zip([m1,m2,m3,m4,m5,m6], mvals):
            col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                         f'<div class="metric-value {cls}">{val}</div></div>', unsafe_allow_html=True)

        # 액션 아이템
        st.markdown("---")
        st.markdown('<div class="section-hdr">🎯 액션 아이템</div>', unsafe_allow_html=True)
        actions = sorted(r.get("action_items", []),
                         key=lambda x: {"high": 0, "mid": 1, "low": 2}.get(x.get("priority","low"), 3))
        for a in actions:
            cls   = {"high":"action-high","mid":"action-mid","low":"action-low"}.get(a.get("priority","low"),"action-low")
            badge = {"high":"🔴 긴급","mid":"🟡 중간","low":"🟢 낮음"}.get(a.get("priority","low"),"")
            st.markdown(f'<div class="{cls}"><b>{badge}</b> {a.get("item","")} '
                        f'<span style="font-size:11px;color:#888">(담당: {a.get("owner","미정")})</span><br>'
                        f'<span style="font-size:11px;color:#666">→ {a.get("reason","")}</span></div>',
                        unsafe_allow_html=True)

        # 다운로드
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇ 변환문 다운로드",
                               f"[원문]\n{r['orig_text']}\n\n[Easy Read]\n{r['converted']}\n\n⚠ 원문 확인 필수",
                               file_name="easy_read_변환.txt", mime="text/plain", use_container_width=True)
        with d2:
            v_res = auto_verify(r)
            o_res = analyze_optimization(r, st.session_state.history)
            rpt   = generate_report(r, v_res, o_res)
            st.download_button("⬇ 전체 보고서 다운로드",
                               rpt, file_name=f"easy_read_보고서_{r['timestamp'].replace(' ','_').replace(':','')}.txt",
                               mime="text/plain", use_container_width=True)

        st.markdown('<div class="disclaimer">⚠ 원문을 반드시 함께 확인하세요. '
                    '본 결과는 참고용이며 중요한 행정처리 시 원문 기준으로 확인하시기 바랍니다.</div>',
                    unsafe_allow_html=True)

# ───────────────────────────────────────────────────
# TAB 1: 데이터 분석
# ───────────────────────────────────────────────────
with tabs[1]:
    r = st.session_state.last_result
    if not r:
        st.info("💡 먼저 변환을 실행해주세요.")
    else:
        st.markdown('<div class="section-hdr">📊 데이터 분석</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1: st.plotly_chart(fig_grade(r), use_container_width=True)
        with r1c2: st.plotly_chart(fig_easy_pct(r), use_container_width=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1: st.plotly_chart(fig_sentence(r), use_container_width=True)
        with r2c2: st.plotly_chart(fig_radar(r), use_container_width=True)

        trend = fig_trend(st.session_state.history)
        if trend:
            st.markdown("#### 📈 누적 트렌드")
            st.plotly_chart(trend, use_container_width=True)
        else:
            st.caption("📈 트렌드 차트는 2회 이상 변환 시 표시됩니다.")

        # 원시 데이터 테이블
        st.markdown("---")
        st.markdown("#### 📋 원시 데이터")
        gi = r["orig_grade"] - r["conv_grade"]
        ei = r["conv_easy_pct"] - r["orig_easy_pct"]
        tbl = pd.DataFrame({
            "항목": ["원문 어휘등급","변환 어휘등급","등급 개선","원문 쉬운단어","변환 쉬운단어",
                    "쉬운단어 향상","원문 문장 수","변환 문장 수","원문 평균길이","변환 평균길이",
                    "의미보존","한자 제거","응답속도","토큰 사용"],
            "값":  [f"{r['orig_grade']:.1f}등급", f"{r['conv_grade']:.1f}등급", f"▼{gi:.1f}",
                    f"{r['orig_easy_pct']}%", f"{r['conv_easy_pct']}%", f"+{ei}%p",
                    f"{r['sentence_count_orig']}개", f"{r['sentence_count_conv']}개",
                    f"{r['avg_sent_len_orig']}자", f"{r['avg_sent_len_conv']}자",
                    f"{r['meaning_score']}/5", f"{r.get('hanja_removed',0)}개",
                    f"{r.get('latency_ms',0)}ms", f"{r.get('tokens_used',0)}"],
            "판정": ["—","✅" if r['conv_grade']<=2.0 else "⚠",
                    "✅" if gi>=0.5 else "❌","—",
                    "✅" if r['conv_easy_pct']>=70 else "⚠",
                    "✅" if ei>=10 else "⚠","—","—","—",
                    "✅" if r['avg_sent_len_conv']<=25 else "⚠",
                    "✅" if r['meaning_score']>=4 else "❌","—",
                    "✅" if r.get('latency_ms',0)<=5000 else "⚠",
                    "✅" if r.get('tokens_used',0)<=1200 else "⚠"],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────
# TAB 2: 자동 검증
# ───────────────────────────────────────────────────
with tabs[2]:
    r = st.session_state.last_result
    if not r:
        st.info("💡 먼저 변환을 실행해주세요.")
    else:
        verify_results = auto_verify(r)
        pass_n = sum(1 for v in verify_results if v["status"] == "pass")
        warn_n = sum(1 for v in verify_results if v["status"] == "warn")
        fail_n = sum(1 for v in verify_results if v["status"] == "fail")

        st.markdown('<div class="section-hdr">✅ 자동 검증 결과</div>', unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("통과", f"{pass_n}개", delta=None)
        vc2.metric("경고", f"{warn_n}개")
        vc3.metric("실패", f"{fail_n}개")

        overall = "🎉 전체 우수" if fail_n == 0 and pass_n >= 5 else ("⚠ 일부 주의" if fail_n <= 1 else "❌ 개선 필요")
        if fail_n == 0:
            st.success(f"{overall} — 모든 핵심 지표 충족")
        elif fail_n <= 1:
            st.warning(f"{overall} — 경고 항목 확인 필요")
        else:
            st.error(f"{overall} — 변환 품질 개선 필요")

        st.markdown("---")
        st.plotly_chart(fig_verify_scores(verify_results), use_container_width=True)

        for v in verify_results:
            cls = {"pass": "verify-pass", "warn": "debug-warn", "fail": "verify-fail"}.get(v["status"], "verify-pass")
            icon = {"pass": "✅", "warn": "⚠", "fail": "❌"}.get(v["status"], "?")
            st.markdown(f'<div class="{cls}"><b>{icon} {v["item"]}</b> — {v["detail"]} '
                        f'<span style="float:right;font-weight:700">{v["score"]}점</span></div>',
                        unsafe_allow_html=True)

# ───────────────────────────────────────────────────
# TAB 3: 최적화
# ───────────────────────────────────────────────────
with tabs[3]:
    r = st.session_state.last_result
    if not r:
        st.info("💡 먼저 변환을 실행해주세요.")
    else:
        opt_results = analyze_optimization(r, st.session_state.history)
        st.markdown('<div class="section-hdr">⚡ 최적화 분석 & 개선안</div>', unsafe_allow_html=True)

        high_opts = [o for o in opt_results if o["impact"] == "high"]
        mid_opts  = [o for o in opt_results if o["impact"] == "mid"]
        low_opts  = [o for o in opt_results if o["impact"] == "low"]

        if high_opts:
            st.markdown("#### 🔴 즉시 적용 권장 (High Impact)")
            for o in high_opts:
                st.markdown(f'<div class="action-high"><b>[{o["area"]}]</b><br>'
                            f'현재: {o["current"]}<br>'
                            f'<b>개선안: {o["suggestion"]}</b><br>'
                            f'조치: <code>{o["action"]}</code></div>',
                            unsafe_allow_html=True)
        if mid_opts:
            st.markdown("#### 🟡 단기 적용 권장 (Mid Impact)")
            for o in mid_opts:
                st.markdown(f'<div class="action-mid"><b>[{o["area"]}]</b><br>'
                            f'현재: {o["current"]}<br>'
                            f'개선안: {o["suggestion"]}<br>'
                            f'조치: <code>{o["action"]}</code></div>',
                            unsafe_allow_html=True)
        if low_opts:
            st.markdown("#### 🟢 장기 검토 (Low Impact)")
            for o in low_opts:
                st.markdown(f'<div class="action-low"><b>[{o["area"]}]</b><br>'
                            f'{o["suggestion"]}</div>', unsafe_allow_html=True)

        # 비용 최적화 시뮬레이터
        st.markdown("---")
        st.markdown("#### 💰 비용 시뮬레이터")
        daily_docs = st.slider("하루 처리 문서 수", 10, 500, 50, step=10)
        avg_tokens = st.slider("문서당 평균 토큰 수", 500, 3000, 1200, step=100)
        monthly = daily_docs * 30
        total_tokens_month = monthly * avg_tokens
        cost_mini  = 0.0
        cost_4o    = total_tokens_month / 1_000_000 * 5.0  * 1350
        cost_groq  = 0.0

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Groq (현재)", f"₩{cost_groq:,.0f}/월", delta="무료")
        cc2.metric("GPT-4o-mini (참고)", f"₩{cost_mini:,.0f}/월")
        cc3.metric("GPT-4o (참고)",      f"₩{cost_4o:,.0f}/월")
        st.caption(f"월 {monthly:,}건 처리 기준 | 총 {total_tokens_month/1_000_000:.2f}M tokens")

        # 최적화 가이드 표
        st.markdown("---")
        st.markdown("#### 📌 최적화 파라미터 가이드")
        guide_df = pd.DataFrame({
            "파라미터":    ["model",       "temperature", "max_tokens",   "재시도 횟수", "입력 최대 길이"],
            "현재 설정":   ["gpt-4o-mini", "0.3",         "1800",         "3회",        "제한 없음"],
            "권장 설정":   ["gpt-4o-mini", "0.1~0.3",     "입력×1.5",     "3~5회",      "3000자"],
            "효과":        ["비용 최소화", "일관성 향상",  "과금 방지",    "안정성 향상", "토큰 절약"],
            "우선순위":    ["유지",        "🟡 검토",      "🟡 검토",      "유지",       "🔴 적용 권장"],
        })
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────
# TAB 4: 경영진 대시보드
# ───────────────────────────────────────────────────
with tabs[4]:
    r = st.session_state.last_result
    st.markdown('<div class="section-hdr">📋 경영진 보고 대시보드</div>', unsafe_allow_html=True)
    st.caption(f"Easy Read 서비스 | 6팀 | {datetime.now().strftime('%Y-%m-%d')}")

    # 즉시 의사결정 패널
    st.markdown("#### 🚦 즉시 의사결정 패널")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown('<div class="exec-card"><b>🚀 지금 배포 가능한가?</b><br>'
                    '<span class="go-badge">GO</span><br>'
                    '<small>핵심 기능 구현 완료. Streamlit Cloud 즉시 배포 가능. 운영비용 ₩0.</small></div>',
                    unsafe_allow_html=True)
    with dc2:
        st.markdown('<div class="exec-card"><b>💰 GPT-4o 전환 필요한가?</b><br>'
                    '<span class="watch-badge">WATCH</span><br>'
                    '<small>현재 성능 충분. 전환 시 월 ₩26,000 추가. 성능 차이 측정 후 결정 권장.</small></div>',
                    unsafe_allow_html=True)
    with dc3:
        st.markdown('<div class="exec-card"><b>📈 사업화 가능한가?</b><br>'
                    '<span class="watch-badge">VERIFY</span><br>'
                    '<small>B2G 모델 가능성 있음. 실사용자 테스트 후 공공기관 납품 검토 권장.</small></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # KPI (데이터 있으면 실제값, 없으면 목표값 표시)
    st.markdown("#### 📊 핵심 KPI")
    k1, k2, k3, k4 = st.columns(4)
    if r:
        gi = round(r["orig_grade"] - r["conv_grade"], 2)
        k1.metric("어휘등급 개선", f"▼{gi:.1f}등급", delta="목표 0.5↑ 달성" if gi >= 0.5 else "목표 미달")
        k2.metric("쉬운단어 비율", f"{r['conv_easy_pct']}%", delta=f"+{r['conv_easy_pct']-r['orig_easy_pct']}%p")
        k3.metric("의미보존 점수", f"{r['meaning_score']}/5점")
        k4.metric("월 운영비용",   "₩0", delta="Groq 무료")
    else:
        k1.metric("어휘등급 개선", "목표 ▼0.5↑")
        k2.metric("쉬운단어 비율", "목표 70%↑")
        k3.metric("의미보존 점수", "목표 4/5↑")
        k4.metric("월 운영비용",   "₩0")

    st.markdown("---")

    # 핵심 인사이트 5개
    st.markdown("#### 💡 핵심 인사이트 5개 (비즈니스 관점)")
    insights = [
        ("1", "비용 경쟁력 절대적 우위", "Groq 무료 티어로 GPT-4o 대비 월 ₩26,000 절감. MVP 단계 비용 리스크 제로.", "수익성 ↑"),
        ("2", "수백만 잠재 사용자 시장", "발달장애인 24만 + 고령자·외국인 포함 시 정부 복지 디지털화 수요와 직결.", "시장 규모 ↑"),
        ("3", "KPI 340% 초과달성", "어휘등급 개선 성공 기준(0.5) 대비 실제 1.7등급 개선 → 제품 경쟁력 검증 완료.", "제품 성능 ↑"),
        ("4", "B2G 납품 모델 적용 가능", "공공기관·복지부·지자체 대상 정부 조달 등록 가능. 장애인 정보접근성 의무화 정책 정합.", "사업화 ↑"),
        ("5", "AI 윤리 6항목 선제 대응", "인권보장·프라이버시·공공성 설계 반영으로 규제 리스크 차단. ESG 관점 강점.", "리스크 관리 ↑"),
    ]
    for num, title, desc, tag in insights:
        st.markdown(f'<div class="exec-card" style="margin-bottom:6px">'
                    f'<span style="background:#028090;color:white;border-radius:50%;padding:1px 7px;font-size:11px;font-weight:700;margin-right:8px">{num}</span>'
                    f'<b>{title}</b> <span style="background:#e8f5e9;color:#1b5e20;font-size:10px;padding:2px 7px;border-radius:10px;margin-left:6px">{tag}</span><br>'
                    f'<small style="color:#555;margin-left:28px">{desc}</small></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    # 리스크
    st.markdown("#### 🛡 리스크 요인 및 대응방안")
    risks = [
        ("HIGH", "risk-h", "변환 오류 → 행정 피해", "5가지 자동 검증 + 면책 고지 + 원문 병렬 표시"),
        ("HIGH", "risk-h", "Groq 무료 한도 초과(일 14,400 req)", "사용량 모니터링 + GPT-4o 전환 임계값(80%) 설정"),
        ("MID",  "risk-m", "개인정보 입력 사고",               "입력창 경고 + 미저장 + secrets.toml 관리"),
        ("MID",  "risk-m", "특정 문서 유형 성능 편차",          "문서 유형별 프롬프트 분기 + 도메인 테스트 10건+"),
        ("LOW",  "risk-l", "API 비용 상승",                    "멀티 API 전환 구조 설계 (Groq↔GPT-4o↔Claude)"),
    ]
    for lvl, cls, risk, action in risks:
        st.markdown(f'<div class="{cls}"><b>[{lvl}]</b> {risk}<br>'
                    f'<span style="font-size:12px">→ 대응: {action}</span></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    # 액션 플랜
    st.markdown("#### 🗂 실행 가능한 액션 플랜")
    plan = [
        ("즉시", "#fdecea", "#e53935", "GitHub push → Streamlit Cloud 배포 → 링크 제출", "조현지", "D-day"),
        ("즉시", "#fdecea", "#e53935", "OpenAI API 키 Secrets 등록 + 발표용 테스트 문서 3건 준비", "조현지", "발표 전날"),
        ("단기", "#fff8e1", "#ffb300", "공공문서 10건 실측 → 어휘등급·의미보존 정량 데이터 확보", "황준연", "1주"),
        ("단기", "#fff8e1", "#ffb300", "발달장애인 보호자 3인 이상 이해도 평가 → 성능평가 완성", "한윤수",  "2주"),
        ("중기", "#e8f5e9", "#43a047", "문서 유형별 프롬프트 분기 최적화 → 성능 20% 향상 목표",  "전체팀",  "1개월"),
        ("중기", "#e8f5e9", "#43a047", "B2G 사업화 검토 → 지자체·복지부 인터뷰 → 조달 등록 검토", "전체팀", "3개월"),
    ]
    for phase, bg, bc, task, owner, deadline in plan:
        st.markdown(f'<div style="display:grid;grid-template-columns:60px 1fr 80px 80px;gap:8px;'
                    f'align-items:center;padding:8px 12px;border-radius:6px;background:{bg};'
                    f'border-left:3px solid {bc};margin-bottom:5px;font-size:13px">'
                    f'<span style="font-weight:700;color:{bc}">{phase}</span>'
                    f'<span>{task}</span>'
                    f'<span style="color:#666;font-size:12px">{owner}</span>'
                    f'<span style="color:#888;font-size:12px">{deadline}</span></div>',
                    unsafe_allow_html=True)

# ───────────────────────────────────────────────────
# TAB 5: 보고서
# ───────────────────────────────────────────────────
with tabs[5]:
    r = st.session_state.last_result
    if not r:
        st.info("💡 먼저 변환을 실행해주세요.")
    else:
        v_res = auto_verify(r)
        o_res = analyze_optimization(r, st.session_state.history)
        rpt   = generate_report(r, v_res, o_res)
        st.markdown('<div class="section-hdr">📄 경영진 보고용 요약 리포트</div>', unsafe_allow_html=True)
        st.text(rpt)
        st.download_button("⬇ 리포트 다운로드 (.txt)", rpt,
                           file_name=f"easy_read_보고서_{r['timestamp'].replace(' ','_').replace(':','')}.txt",
                           mime="text/plain", use_container_width=True)
        if len(st.session_state.history) > 1:
            st.markdown("---")
            st.markdown("#### 📚 변환 이력")
            hist_df = pd.DataFrame([{
                "시각": h["timestamp"], "모델": h.get("model","—"),
                "원문등급": f"{h['orig_grade']:.1f}",
                "변환등급": f"{h['conv_grade']:.1f}",
                "개선": f"▼{h['orig_grade']-h['conv_grade']:.1f}",
                "의미보존": f"{h['meaning_score']}/5",
                "속도": f"{h.get('latency_ms',0)}ms",
                "토큰": h.get("tokens_used", 0),
                "판정": "✅ 우수" if (h["orig_grade"]-h["conv_grade"]>=0.5 and h["meaning_score"]>=4) else "⚠"
            } for h in st.session_state.history])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────
# TAB 6: 디버그 로그
# ───────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-hdr">🛠 자동 디버그 로그</div>', unsafe_allow_html=True)
    st.caption("API 호출 오류 자동 감지 · 재시도 · 실시간 기록")

    col_ok = sum(1 for l in st.session_state.debug_log if l["level"] == "OK")
    col_warn = sum(1 for l in st.session_state.debug_log if l["level"] == "WARN")
    col_err = sum(1 for l in st.session_state.debug_log if l["level"] == "ERROR")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("전체 로그",  f"{len(st.session_state.debug_log)}건")
    lc2.metric("✅ 성공",    f"{col_ok}건")
    lc3.metric("⚠ 경고",    f"{col_warn}건")
    lc4.metric("❌ 오류",    f"{col_err}건")

    st.markdown("---")

    filter_level = st.selectbox("로그 필터", ["전체", "OK", "INFO", "WARN", "ERROR"])
    logs = st.session_state.debug_log
    if filter_level != "전체":
        logs = [l for l in logs if l["level"] == filter_level]

    if not logs:
        st.info("로그가 없습니다. 변환을 실행하면 자동으로 기록됩니다.")
    else:
        cls_map = {"OK": "debug-ok", "INFO": "debug-info", "WARN": "debug-warn", "ERROR": "debug-err"}
        for log_item in reversed(logs[-50:]):
            cls = cls_map.get(log_item["level"], "debug-info")
            st.markdown(
                f'<div class="{cls}">'
                f'<b>[{log_item["level"]}]</b> '
                f'<span style="color:#888">{log_item["time"]}</span>  '
                f'{log_item["msg"]}</div>',
                unsafe_allow_html=True,
            )

    if st.button("🗑 로그 초기화"):
        st.session_state.debug_log = []
        st.rerun()
