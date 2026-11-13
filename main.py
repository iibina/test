import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="요일별 전력 소비 예측 실습실", page_icon="⚡", layout="wide")

DATA_FILE = Path(__file__).parent / "전력소비_전처리.csv"
REQUIRED_COLUMNS = ["월", "요일특성", "시간", "전력소비계수"]
DAY_ORDER = ["월요일", "근무일(화~금)", "토요일", "일요일"]
TARGET = "전력소비계수"

HOUR_OPTIONS = ["사용 안 함", "숫자 그대로 (직선)", "곡선 (다항식)", "시간대별 따로 (원-핫)"]
MONTH_OPTIONS = ["사용 안 함", "숫자 그대로 (직선)", "월별 따로 (원-핫)"]


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
COLUMN_ALIASES = {
    "월": ["월", "month", "달"],
    "요일특성": ["요일특성", "요일", "요일구분", "요일별", "구분", "daytype"],
    "시간": ["시간", "시간대", "시", "hour"],
    "전력소비계수": ["전력소비계수", "소비계수", "계수", "전력소비", "value"],
}


def clean_name(name) -> str:
    """열 이름의 BOM, 공백, 따옴표를 없애고 한글 유니코드를 통일한다."""
    text = unicodedata.normalize("NFC", str(name))
    for ch in ("\ufeff", "\u200b", "\xa0", '"', "'"):
        text = text.replace(ch, "")
    return "".join(text.split()).lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """열 이름이 조금 달라도 표준 이름(월, 요일특성, 시간, 전력소비계수)으로 맞춘다."""
    alias_lookup = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lookup[clean_name(alias)] = standard

    rename = {}
    for col in df.columns:
        standard = alias_lookup.get(clean_name(col))
        if standard and standard not in rename.values():
            rename[col] = standard
    return df.rename(columns=rename)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as err:
            last_error = err
    else:
        raise last_error
    return standardize_columns(df)


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """'1월', '1시' 같은 글자가 섞여 있어도 숫자로 바꾸고, 요일특성 글자도 정리한다."""
    df = df.copy()
    for col in ("월", "시간"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce"
        )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    day = df["요일특성"].astype(str).map(lambda x: unicodedata.normalize("NFC", x).strip())
    df["요일특성"] = day
    return df.dropna(subset=REQUIRED_COLUMNS)


def get_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error("데이터 파일 `전력소비_전처리.csv`를 찾을 수 없어요.")
        st.info(
            "깃허브 저장소에서 `main.py`와 **같은 폴더**에 `전력소비_전처리.csv` 파일이 "
            "올라가 있는지 확인해 주세요. (파일 인코딩: UTF-8)"
        )
        st.stop()

    df = load_data(str(DATA_FILE))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"데이터에 필요한 열이 없어요: {', '.join(missing)}")
        st.info(f"필요한 열: {', '.join(REQUIRED_COLUMNS)}")
        st.warning(
            "지금 파일에서 읽은 열 이름: "
            + ", ".join(f"`{c}`" for c in df.columns)
            + "\n\n올라가 있는 CSV가 변환된 `전력소비_전처리.csv`(열이 4개, 1,152행)인지 확인해 주세요."
        )
        st.stop()

    df = clean_values(df[REQUIRED_COLUMNS])

    unknown = sorted(set(df["요일특성"]) - set(DAY_ORDER))
    if unknown:
        st.error("요일특성에 예상하지 못한 값이 있어요: " + ", ".join(unknown))
        st.info("요일특성 값은 `" + "`, `".join(DAY_ORDER) + "` 네 가지여야 해요.")
        st.stop()
    if df.empty:
        st.error("사용할 수 있는 행이 없어요. CSV 내용을 확인해 주세요.")
        st.stop()

    df["요일특성"] = pd.Categorical(df["요일특성"], categories=DAY_ORDER, ordered=True)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 모델 관련 함수
# ------------------------------------------------------------------
def make_features(df, use_day, month_mode, hour_mode, combo):
    """선택한 옵션에 따라 입력 변수 표를 만든다."""
    X = pd.DataFrame(index=df.index)
    cat_cols, num_cols, poly_cols = [], [], []

    if use_day:
        X["요일특성"] = df["요일특성"].astype(str)
        cat_cols.append("요일특성")

    if month_mode == "숫자 그대로 (직선)":
        X["월"] = df["월"]
        num_cols.append("월")
    elif month_mode == "월별 따로 (원-핫)":
        X["월(범주)"] = df["월"].astype(str)
        cat_cols.append("월(범주)")

    if hour_mode == "숫자 그대로 (직선)":
        X["시간"] = df["시간"]
        num_cols.append("시간")
    elif hour_mode == "곡선 (다항식)":
        X["시간"] = df["시간"]
        poly_cols.append("시간")
    elif hour_mode == "시간대별 따로 (원-핫)":
        X["시간(범주)"] = df["시간"].astype(str)
        cat_cols.append("시간(범주)")

    if combo:
        X["요일×시간"] = df["요일특성"].astype(str) + "_" + df["시간"].astype(str)
        cat_cols.append("요일×시간")

    return X, cat_cols, num_cols, poly_cols


def build_pipeline(cat_cols, num_cols, poly_cols, degree):
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if poly_cols:
        poly = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ]
        )
        transformers.append(("poly", poly, poly_cols))

    pre = ColumnTransformer(transformers, sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def evaluate(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


# ------------------------------------------------------------------
# 화면 시작
# ------------------------------------------------------------------
st.title("⚡ 요일별 전력 소비 예측 실습실")
st.caption("선형회귀로 '요일'과 '시간대'가 전력소비계수를 얼마나 설명하는지 직접 실험해 봐요.")

df = get_data()

tab_data, tab_lab = st.tabs(["📊 데이터 살펴보기", "🧪 모델 실습"])

# ------------------------------------------------------------------
# 탭 1: 데이터 살펴보기
# ------------------------------------------------------------------
with tab_data:
    st.subheader("데이터 소개")
    st.markdown(
        """
- **월**: 1~12
- **요일특성**: 월요일 / 근무일(화~금) / 토요일 / 일요일
- **시간**: 1~24시
- **전력소비계수**: 요일특성별 평균이 약 1000이 되도록 맞춘 값
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("행 수", f"{len(df):,}")
    c2.metric("열 수", len(df.columns))
    c3.metric("전체 평균", f"{df[TARGET].mean():,.1f}")

    st.markdown("**앞부분 미리보기**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**요일특성별 평균 / 표준편차**")
    summary = (
        df.groupby("요일특성", observed=True)[TARGET]
        .agg(평균="mean", 표준편차="std", 최소="min", 최대="max")
        .round(1)
    )
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### 시간대별 평균 전력소비계수")
    hourly = df.groupby(["요일특성", "시간"], observed=True)[TARGET].mean().reset_index()
    fig_line = px.line(
        hourly,
        x="시간",
        y=TARGET,
        color="요일특성",
        markers=True,
        category_orders={"요일특성": DAY_ORDER},
    )
    fig_line.update_layout(xaxis=dict(dtick=1), legend_title_text="요일특성")
    st.plotly_chart(fig_line, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 요일특성별 분포")
        fig_box = px.box(
            df,
            x="요일특성",
            y=TARGET,
            color="요일특성",
            category_orders={"요일특성": DAY_ORDER},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with col_b:
        st.markdown("#### 월 × 시간 히트맵")
        day_pick = st.selectbox("요일특성 선택", DAY_ORDER, key="heat_day")
        sub = df[df["요일특성"] == day_pick]
        pivot = sub.pivot_table(index="월", columns="시간", values=TARGET, aggfunc="mean")
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels=dict(x="시간", y="월", color=TARGET),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------------
# 탭 2: 모델 실습
# ------------------------------------------------------------------
with tab_lab:
    st.subheader("선형회귀 모델 만들기")
    with st.expander("💡 이렇게 실습해 보세요", expanded=False):
        st.markdown(
            """
1. 처음에는 **시간대별 따로(원-핫)** 로 학습된 결과를 먼저 살펴보세요.
2. 시간 사용 방식을 **사용 안 함**으로 바꿔서 요일특성만 쓰면 설명력(R²)이 얼마나 떨어지는지 확인해 보세요.
3. 시간을 **숫자 그대로(직선)** 로 넣어 보세요. 하루 모양이 직선일까요?
4. 시간을 **곡선(다항식)** 으로 바꾸고 차수를 올려 보세요. 훈련/테스트 점수 차이도 같이 보세요.
5. 마지막으로 **요일×시간 조합**을 켜서, 요일마다 다른 하루 모양을 학습시켜 보세요.
"""
        )

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**① 입력 변수 선택**")
        use_day = st.checkbox("요일특성 사용", value=True)
        month_mode = st.radio("월 사용 방식", MONTH_OPTIONS, index=0)
        hour_mode = st.radio("시간 사용 방식", HOUR_OPTIONS, index=3)
        degree = 2
        if hour_mode == "곡선 (다항식)":
            degree = st.slider("다항식 차수", 2, 10, 4)
        combo = st.checkbox(
            "요일특성 × 시간 조합 사용",
            value=False,
            help="요일특성마다 시간대 모양을 따로 학습해요.",
        )

        st.markdown("**② 데이터 나누기**")
        test_pct = st.slider("테스트 데이터 비율 (%)", 10, 40, 20, step=5)
        seed = st.number_input("무작위 시드", min_value=0, max_value=9999, value=42, step=1)

    X_all, cat_cols, num_cols, poly_cols = make_features(df, use_day, month_mode, hour_mode, combo)

    with right:
        if X_all.shape[1] == 0:
            st.warning("입력 변수를 하나 이상 선택해 주세요.")
        else:
            y_all = df[TARGET]
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_pct / 100, random_state=int(seed)
            )

            pipe = build_pipeline(cat_cols, num_cols, poly_cols, degree)
            pipe.fit(X_train, y_train)

            pred_train = pipe.predict(X_train)
            pred_test = pipe.predict(X_test)
            m_train = evaluate(y_train, pred_train)
            m_test = evaluate(y_test, pred_test)

            n_features = pipe.named_steps["pre"].transform(X_train.head(1)).shape[1]

            st.markdown("**③ 결과**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("테스트 R²", f"{m_test['R²']:.3f}")
            k2.metric("훈련 R²", f"{m_train['R²']:.3f}")
            k3.metric("테스트 MAE", f"{m_test['MAE']:.1f}")
            k4.metric("테스트 RMSE", f"{m_test['RMSE']:.1f}")
            st.caption(
                f"모델이 실제로 사용한 입력 열 개수: {n_features}개 · "
                f"훈련 {len(X_train):,}행 / 테스트 {len(X_test):,}행"
            )

            gap = m_train["R²"] - m_test["R²"]
            if m_test["R²"] < 0.1:
                st.warning("설명력이 매우 낮아요. '시간 사용 방식'을 '시간대별 따로(원-핫)'이나 '곡선(다항식)'으로 바꿔 보세요.")
            elif gap > 0.1:
                st.warning("훈련 점수가 테스트 점수보다 꽤 높아요. 과적합이 아닌지 살펴보세요.")
            else:
                st.success("훈련과 테스트 점수가 비슷해요. 안정적으로 학습됐어요.")

            # 시간대별 실제 vs 예측 (전체 데이터 기준)
            all_pred = pipe.predict(X_all)
            curve = df[["요일특성", "시간"]].copy()
            curve["실제"] = y_all.values
            curve["예측"] = all_pred
            curve = (
                curve.groupby(["요일특성", "시간"], observed=True)[["실제", "예측"]]
                .mean()
                .reset_index()
                .melt(id_vars=["요일특성", "시간"], var_name="구분", value_name=TARGET)
            )
            fig_curve = px.line(
                curve,
                x="시간",
                y=TARGET,
                color="요일특성",
                line_dash="구분",
                category_orders={"요일특성": DAY_ORDER, "구분": ["실제", "예측"]},
            )
            fig_curve.update_layout(
                title="시간대별 평균: 실제(실선) vs 예측(점선) · 전체 데이터 기준",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # 실제 vs 예측 산점도 (테스트 데이터)
            lo = float(min(y_test.min(), pred_test.min()))
            hi = float(max(y_test.max(), pred_test.max()))
            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=y_test,
                    y=pred_test,
                    mode="markers",
                    name="테스트 데이터",
                    marker=dict(size=6, opacity=0.6),
                )
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    name="완벽한 예측",
                    line=dict(dash="dash"),
                )
            )
            fig_sc.update_layout(
                title="테스트 데이터: 실제값 vs 예측값",
                xaxis_title="실제 전력소비계수",
                yaxis_title="예측 전력소비계수",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
