import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objs as go
import joblib

# Streamlit 레이아웃 설정
st.set_page_config(layout="wide")

# 사전 학습된 모델 로드
model = joblib.load(r"xgboost_model2.pkl")  # 모델 객체 로드

# 학습에 사용한 피처 이름
feature_columns = ['국민임대', '공공임대', '영구임대', '행복주택', '임대료',
                '버스추정교통량', '승용차추정교통량', '대중교통접근성_상', '대중교통접근성_중', '대중교통접근성_하',
                '생활서비스형상권', '소매외식형상권', '이동여가형상권', 'pop_20g_pop', 'pop_30g_pop',
                'pop_40g_pop', 'pop_50g_pop', 'pop_60g_pop', 'pop_70g_pop',
                'pop_80g_pop', 'pop_90g_pop', 'pop_100g_pop']

# 조정할 변수 설정
continuous_vars = ['pop_20g_pop', 'pop_30g_pop','pop_60g_pop', 'pop_70g_pop',
                   '소매외식형상권', '이동여가형상권']
categorical_vars = ['대중교통접근성_상', '대중교통접근성_중', '대중교통접근성_하']

# 데이터 읽기
merged_data = pd.read_csv(r"merged_data.csv")

# 주택유형별 색상 매핑 (RGB 값)
housing_color_map = {
    "국민임대": [255, 165, 0],  # 주황색
    "공공임대": [0, 128, 255],  # 파란색
    "영구임대": [34, 139, 34],  # 초록색
    "행복주택": [220, 20, 60],  # 빨간색
}

# 데이터프레임에 색상 추가
merged_data["color"] = merged_data["유형별"].map(housing_color_map)

# 변수 이름 매핑
variable_display_names = {
    "pop_20g_pop": "20대 인구",
    "pop_30g_pop": "30대 인구",
    "pop_60g_pop": "60대 인구",
    "pop_70g_pop": "70대 이상 인구",
    "소매외식형상권": "소매/외식 상권",
    "이동여가형상권": "이동/여가형 상권",
    "대중교통접근성_상": "대중교통 접근성 (상)",
    "대중교통접근성_중": "대중교통 접근성 (중)",
    "대중교통접근성_하": "대중교통 접근성 (하)"
}

# Streamlit 대시보드
st.title("공공주택 내 공유차량 이용에 영향을 미치는 요인 분석")
st.sidebar.header("변수 설정")

# '단지명' 선택
selected_complex = st.sidebar.selectbox(
    "단지 선택",
    options=merged_data["단지명"].unique(),
    index=0  # 기본값
)

# 선택된 단지명으로 데이터 필터링
filtered_data = merged_data[merged_data["단지명"] == selected_complex]

# 범주형 변수 선택
selected_categorical_var = st.sidebar.selectbox(
    "대중교통 접근성 선택",
    options=categorical_vars,
    format_func=lambda x: variable_display_names[x],  # 매핑된 이름으로 표시
    index=0
)

# 범주형 변수 매핑 (선택된 값 -> 숫자)
categorical_mapping = {cat: (1 if cat == selected_categorical_var else 0) for cat in categorical_vars}

# 변수 초기화 및 상태 관리
if "selected_vars" not in st.session_state or st.session_state.get("current_complex") != selected_complex:
    # 새로운 단지가 선택되었을 때 초기값 설정
    st.session_state.selected_vars = {
        var: int(filtered_data[var].iloc[0]) if not filtered_data.empty and not pd.isna(filtered_data[var].iloc[0]) else 0
        for var in continuous_vars
    }
    st.session_state.initial_values = st.session_state.selected_vars.copy()
    st.session_state.current_complex = selected_complex
    # 초기 예측값 설정
    if not filtered_data.empty:
        filtered_data_filled = filtered_data[feature_columns].fillna(0)  # NaN 값을 0으로 대체
        st.session_state.initial_predicted_value = model.predict(filtered_data_filled)[0]
    else:
        st.session_state.initial_predicted_value = 0  # 비어있는 경우 초기 예측값은 0

# 슬라이더를 통해 변수 조정
selected_vars = {}
for var in continuous_vars:
    # 데이터프레임의 실제 최소값과 최대값 계산
    original_min = int(filtered_data[var].min() if filtered_data[var].min() >= 0 else 0)
    original_max = int(filtered_data[var].max() + 1)

    # 최대값 설정
    if var == "버스추정교통량":
        max_val = 250000  # 버스추정교통량의 최대값 설정
    elif var == "소매외식형상권":
        max_val = 2200  # 소매외식형상권의 최대값 설정
    else:
        max_val = 1500  # 나머지 변수의 최대값 설정

    # 초기값은 세션 상태의 초기값으로 설정
    initial_value = st.session_state.initial_values[var]

    # 슬라이더 생성 시 타입 일치
    selected_vars[var] = st.sidebar.slider(
        label=variable_display_names[var],  # 매핑된 이름으로 표시
        min_value=0,  # 최소값은 항상 0으로 고정
        max_value=max_val,  # 변수별 최대값 설정
        value=int(initial_value),  # 초기값은 세션 상태의 초기값을 정수로 변환
        step=1,  # step을 정수로 설정
        key=f"slider_{var}"
    )
    st.session_state.selected_vars[var] = selected_vars[var]  # 상태 업데이트

# 입력 데이터 생성 (연속형 + 범주형 변수를 포함)
input_data = pd.DataFrame([{
    **selected_vars,
    **categorical_mapping  # 범주형 변수 매핑 추가
}], columns=feature_columns)

# 모델 예측
predicted_value = model.predict(input_data)[0]

# 예측값 출력
st.sidebar.subheader("예측 결과")
st.sidebar.write(f"선택된 단지: {selected_complex}")
st.sidebar.write(f"예상 공유차량 이용건수: {predicted_value:.2f}")

# 지도 시각화를 위한 데이터 업데이트
filtered_data = filtered_data.copy()  # 복사를 통해 독립적으로 관리
if not filtered_data.empty:
    filtered_data_filled = filtered_data[feature_columns].fillna(0)  # NaN 값을 0으로 대체
    filtered_data["예측값"] = model.predict(filtered_data_filled)  # 실시간 업데이트
else:
    filtered_data["예측값"] = 0

# 변수 변화량 계산 및 출력
st.subheader("📈 변수 변화 및 결과 요약")
cols = st.columns(len(selected_vars) + 1)  # 변수 + 예측값

for i, (var, value) in enumerate(selected_vars.items()):
    base_value = st.session_state.initial_values[var]  # 초기값
    change = value - base_value  # 변화량 계산

    if change > 0:
        arrow = "▲"
        color = "red"
        change_display = f"(+{change})"
    elif change < 0:
        arrow = "▼"
        color = "blue"
        change_display = f"({change})"
    else:
        arrow = "➖"
        color = "gray"
        change_display = "(-)"

    # 변수 이름을 매핑된 이름으로 표시
    variable_name = variable_display_names[var]
    if change == 0:
        display_text = f"{base_value} {change_display}"
    else:
        display_text = f"{value} {change_display}"

    cols[i].markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; 
                    padding: 20px; text-align: center; background-color: #f9f9f9; font-size: 15px;">
            <span style='color:{color}; font-weight:bold;'>{variable_name}</span><br>
            <span style='color:{color};'>{display_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# 예측값 변화량 계산 및 출력
initial_predicted_value = st.session_state["initial_predicted_value"]
predicted_change = predicted_value - initial_predicted_value

# 변화량 시각화 색상 설정
if predicted_change > 0:
    predicted_color = "red"
    arrow = "▲"
    change_display = f"(+{predicted_change:.2f})"
elif predicted_change < 0:
    predicted_color = "blue"
    arrow = "▼"
    change_display = f"({predicted_change:.2f})"
else:
    predicted_color = "gray"
    arrow = "➖"
    change_display = "(-)"

# 최종 예측값과 변화량 출력
cols[-1].markdown(
    f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; 
                padding: 20px; text-align: center; background-color: #f9f9f9; font-size: 15px;">
        <span style='color:{predicted_color}; font-weight:bold;'>예상 공유차량 이용건수</span><br>
        <span style='color:{predicted_color};'>{predicted_value:.2f} {change_display}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Pydeck 지도 생성 함수
def create_pydeck_map(predicted_value, selected_complex):

    # sumID에서 연도 제거
    merged_data["cleaned_sumID"] = merged_data["sumID"].str.extract(r"(\D+.*)")  # 연도 제외한 나머지 추출

    # 연도 제거된 sumID로 그룹화하여 평균 계산
    averaged_data = (
        merged_data.groupby(["cleaned_sumID", "lon", "lat"], as_index=False)["공유차량 이용건수"]
        .mean()
        .rename(columns={"공유차량 이용건수": "평균공유차량이용건수"})
    )

    # 선택된 단지 데이터 가져오기
    selected_data = merged_data[merged_data["단지명"] == selected_complex]
    if not selected_data.empty:
        selected_row = selected_data.iloc[0]  # 선택된 단지의 첫 번째 행
    else:
        # 기본값 설정
        selected_row = pd.Series({"lon": 0, "lat": 0, "color": [0, 0, 0]})

    # 선택되지 않은 단지 데이터 필터링
    unselected_data = merged_data[merged_data["단지명"] != selected_complex]

    # 히트맵 데이터를 위해 "공유차량 이용건수" 컬럼 확인
    if "공유차량 이용건수" not in merged_data.columns:
        st.error("데이터에 '공유차량 이용건수' 컬럼이 없습니다.")
        return None

    # 히트맵 레이어: 평균 공유차량 이용건수를 반영
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=averaged_data,
        get_position=["lon", "lat"],  # 위치
        get_weight="평균공유차량이용건수",  # 히트맵의 가중치 (평균 공유차량 이용건수)
        radiusPixels=50,  # 각 데이터 포인트 반경
        intensity=1,  # 강도 조정
        threshold=0.1,  # 최소 표시 강도
    )

    # 레이어 1: 선택되지 않은 단지 (점으로 표시)
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=unselected_data,
        get_position=["lon", "lat"],
        get_color="color",  # RGB 값 사용
        get_radius=200,  # 점의 크기
    )

    # 레이어 2: 선택된 단지 (3D 막대)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=pd.DataFrame([{
            "lon": selected_row["lon"],  # 선택된 단지의 경도
            "lat": selected_row["lat"],  # 선택된 단지의 위도
            "elevation": predicted_value,  # 막대 높이를 예측값으로 설정
            "color": selected_row["color"],  # 막대 색상
        }]),
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=10,  # 높이 스케일 설정
        radius=500,  # 막대 반경
        get_fill_color="color",  # RGB 색상
        pickable=True,
        auto_highlight=True,
    )

    # 뷰포트 설정 (지도의 기본 위치와 확대 수준)
    view_state = pdk.ViewState(
        latitude=selected_row["lat"],  # 선택된 단지의 위도
        longitude=selected_row["lon"],  # 선택된 단지의 경도
        zoom=13,  # 확대 수준
        pitch=45,  # 기울기
    )

    # 지도 반환 (두 레이어 사용)
    return pdk.Deck(
        layers=[heatmap_layer, scatter_layer, column_layer],
        initial_view_state=view_state,
    )

# Plotly 그래프 생성 함수 (세 개로 분리)
def update_plots():
    # 인구수 그룹
    population_data = {variable_display_names[var]: selected_vars[var] for var in continuous_vars if var in [
        'pop_20g_pop', 'pop_30g_pop','pop_60g_pop', 'pop_70g_pop'
    ]}

    # 상권 그룹
    commercial_data = {variable_display_names[var]: selected_vars[var] for var in continuous_vars if var in [
        '소매외식형상권', '이동여가형상권'
    ]}
    # 추정 교통량 그룹
    #traffic_data = {variable_display_names[var]: selected_vars[var] for var in continuous_vars if var == "버스추정교통량"}

    # Plotly 그래프 생성 (인구수)
    population_fig = go.Figure(data=[go.Bar(
        x=list(population_data.keys()),
        y=list(population_data.values()),
        text=list(population_data.values()),
        textposition='auto'
    )])
    population_fig.update_layout(
        title='인구수',
        xaxis_title='변수',
        yaxis_title='값',
        yaxis=dict(
            range=[0, max(population_data.values()) * 1.2],  # y축 범위를 데이터에 맞게 설정
            tick0=0,          # y축 시작점
            dtick=100         # y축 눈금 간격 설정
        )
    )

    # Plotly 그래프 생성 (상권)
    commercial_fig = go.Figure(data=[go.Bar(
        x=list(commercial_data.keys()),
        y=list(commercial_data.values()),
        text=list(commercial_data.values()),
        textposition='auto'
    )])
    commercial_fig.update_layout(
        title='상권',
        xaxis_title='변수',
        yaxis_title='값',
        yaxis=dict(
            range=[0, max(commercial_data.values()) * 1.2],  # y축 범위를 데이터에 맞게 설정
            tick0=0,          # y축 시작점
            dtick=200         # y축 눈금 간격 설정 (200씩 증가)
        )
    )

    return population_fig, commercial_fig

# 레이아웃: Pydeck 지도 + Plotly 그래프들 + 이미지를 나란히 배치
col1, col2 = st.columns([3, 1])  # col1: 너비 비율 3, col2: 너비 비율 1

# 왼쪽 열: Pydeck 지도와 Plotly 그래프
with col1:
    # Pydeck 지도
    st.subheader("🗺️ Pydeck 지도")
    deck_map = create_pydeck_map(predicted_value, selected_complex)
    if deck_map:
        st.pydeck_chart(deck_map, use_container_width=True)

    # Plotly 그래프를 수평으로 배치
    # Plotly 그래프 생성
    population_fig, commercial_fig = update_plots()

    st.subheader("📊 Plotly 그래프들")
    graph_cols = st.columns(2)  # 두 개의 열로 나누어 그래프 배치

    with graph_cols[0]:  # 첫 번째 그래프
        st.plotly_chart(population_fig, use_container_width=True)

    with graph_cols[1]:  # 두 번째 그래프
        st.plotly_chart(commercial_fig, use_container_width=True)

# 오른쪽 열: 이미지를 수직으로 배치
with col2:

    st.subheader("📈 xgboost 그래프")
    st.image("image.png", caption="XGBoost 예측값", use_container_width=True)
    st.image("image2.png", caption="XGBoost 변수 중요도", use_container_width=True)

    # 세션 상태 업데이트 (이미는 필요없어 보임)
    st.session_state["previous_predicted_value"] = predicted_value

# streamlit run my_app.py
