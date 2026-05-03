import os
import glob
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# shap 改成可选依赖
try:
    import shap
except Exception:
    shap = None


# =========================================================
# 0. 页面配置
# =========================================================
st.set_page_config(
    page_title="LBP Recurrence Prediction Model",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 1. 模型路径
#    如果你想固定模型文件路径，直接改 MODEL_PATH
# =========================================================
MODEL_PATH = None
MODEL_PATTERNS = [
    "best_model/BestModel_*.joblib",
    "BestModel_*.joblib"
]


# =========================================================
# 2. 变量显示名映射
#    这里尽量适配你 LBP 数据里的常见变量
# =========================================================
NAME_MAP = {
    "age": "Age",
    "sex": "sex",
    "height": "Height",
    "weight": "Weight",
    "BMI": "BMI",
    "edu": "Education",
    "career": "Career",
    "Medication_history": "Medication history",
    "clbp_time": "CLBP duration",
    "root_pain": "root pain",
    "surgical_history": "surgical history",
    "therapy_number": "Therapy number",
    "Sedentary_time": "Sedentary time",
    "Physical_activity": "Physical activity",
    "smoken": "smoken",
    "drink": "drink",
    "V0_NRS_now": "V0 NRS",
    "NRS_mean_week": "NRS mean week",
    "NRS_most_week": "NRS most week",
    "V0_BPI_pain_severity": "V0 BPI",
    "BPI_pain_count": "BPI pain count",
    "BPI_pain_interference": "BPI pain interference",
    "V0_RMDQ": "V0 RMDQ",
    "Depression": "Depression",
    "Anxiety": "Anxiety",
    "Stress": "Stress",
    "MMSE": "MMSE",
    "PCS": "PCS",
    "CSI": "CSI",
    "CSI_symptoms": "CSI symptoms"
}


# =========================================================
# 3. 页面样式
# =========================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #f7f7f8;
        color: #2d2d2d;
        font-family: "Times New Roman", Georgia, serif;
    }

    .block-container {
        max-width: 760px;
        padding-top: 1.6rem;
        padding-bottom: 2.8rem;
    }

    .app-title {
        font-size: 34px;
        font-weight: 700;
        color: #202020;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }

    .app-subtitle {
        font-size: 15px;
        color: #6a6a6a;
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }

    .app-divider {
        border-top: 1px solid #cfcfcf;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }

    .stNumberInput > label,
    .stSelectbox > label {
        font-size: 15px !important;
        color: #313131 !important;
        font-weight: 500 !important;
        margin-bottom: 0.25rem !important;
    }

    div[data-baseweb="input"] > div {
        background: #ececef !important;
        border: 1px solid #ececef !important;
        border-radius: 6px !important;
        min-height: 46px !important;
    }

    div[data-baseweb="select"] > div {
        background: #ececef !important;
        border: 1px solid #ececef !important;
        border-radius: 6px !important;
        min-height: 46px !important;
    }

    div[data-baseweb="input"] input {
        background: transparent !important;
        color: #404040 !important;
        font-size: 18px !important;
        font-family: "Times New Roman", Georgia, serif !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        font-family: "Times New Roman", Georgia, serif !important;
        font-size: 16px !important;
        color: #4a4a4a !important;
    }

    .stButton > button {
        background-color: #ffffff !important;
        color: #ff6d73 !important;
        border: 2px solid #ff9296 !important;
        border-radius: 7px !important;
        padding: 0.42rem 1.05rem !important;
        font-size: 18px !important;
        font-family: "Times New Roman", Georgia, serif !important;
        box-shadow: 0 2px 8px rgba(255, 105, 120, 0.08) !important;
    }

    .stButton > button:hover {
        background-color: #fff5f6 !important;
        color: #ff5864 !important;
        border-color: #ff7c83 !important;
    }

    .result-sentence {
        font-size: 22px;
        font-style: italic;
        font-weight: 600;
        color: #2d2d2d;
        line-height: 1.5;
        margin-top: 1.3rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 18px 14px 18px;
        box-shadow: 0 6px 20px rgba(50, 50, 93, 0.06);
        border: 1px solid #f0f0f0;
        margin-top: 0.25rem;
        margin-bottom: 0.8rem;
    }

    .mini-caption {
        font-size: 13px;
        color: #7c7c7c;
        margin-top: 0.35rem;
    }

    .feature-tip {
        font-size: 13px;
        color: #7c7c7c;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. 工具函数
# =========================================================
def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores, dtype=float)
        return (scores - s_min) / (s_max - s_min)

    raise ValueError(f"模型 {type(model).__name__} 不支持概率输出")


def auto_find_model_path():
    if MODEL_PATH is not None and os.path.exists(MODEL_PATH):
        return MODEL_PATH

    for pattern in MODEL_PATTERNS:
        files = glob.glob(pattern)
        if files:
            files = sorted(files, key=os.path.getmtime, reverse=True)
            return files[0]

    raise FileNotFoundError(
        "未找到 BestModel_*.joblib，请把模型文件放到当前目录或 best_models 文件夹中。"
    )


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


def pretty_label(name: str) -> str:
    return NAME_MAP.get(name, name.replace("_", " "))


def infer_step_and_format(value):
    try:
        value = float(value)
    except Exception:
        return 1.0, "%.2f"

    if abs(value - round(value)) < 1e-9:
        return 1.0, "%.2f"
    return 0.1, "%.2f"


def parse_preprocessor_schema(preprocessor):
    schema = {
        "numeric_cols": [],
        "categorical_cols": [],
        "numeric_defaults": {},
        "categorical_defaults": {},
        "categorical_options": {},
        "processed_to_raw": {}
    }

    for trans_name, trans_obj, cols in preprocessor.transformers_:
        if trans_name == "remainder" or trans_obj == "drop":
            continue

        cols = list(cols)

        if trans_name == "num":
            schema["numeric_cols"] = cols
            imputer = trans_obj.named_steps["imputer"]
            stats = list(imputer.statistics_)

            for c, v in zip(cols, stats):
                if pd.isna(v):
                    v = 0.0
                schema["numeric_defaults"][c] = float(v)
                schema["processed_to_raw"][c] = c

        elif trans_name == "cat":
            schema["categorical_cols"] = cols

            imputer = trans_obj.named_steps["imputer"]
            onehot = trans_obj.named_steps["onehot"]

            stats = list(imputer.statistics_)
            out_names = list(onehot.get_feature_names_out(cols))

            idx = 0
            for i, c in enumerate(cols):
                default_val = stats[i]
                if pd.isna(default_val):
                    default_val = 0

                options = []
                for x in list(onehot.categories_[i]):
                    if hasattr(x, "item"):
                        x = x.item()
                    if not pd.isna(x):
                        options.append(x)

                if default_val not in options:
                    options = [default_val] + options

                schema["categorical_defaults"][c] = default_val
                schema["categorical_options"][c] = options

                n_cat = len(onehot.categories_[i])
                for _ in range(n_cat):
                    schema["processed_to_raw"][out_names[idx]] = c
                    idx += 1

    return schema


def extract_selected_raw_features(selected_processed_features, processed_to_raw):
    raw_features = []
    seen = set()

    for f in selected_processed_features:
        raw_name = processed_to_raw.get(f, f)
        if raw_name not in seen:
            raw_features.append(raw_name)
            seen.add(raw_name)

    return raw_features


def aggregate_processed_scores_to_raw(processed_scores, processed_to_raw):
    raw_scores = {}

    for feat, score in processed_scores.items():
        raw_name = processed_to_raw.get(feat, feat)
        raw_scores[raw_name] = raw_scores.get(raw_name, 0.0) + float(score)

    return raw_scores


def _extract_shap_matrix(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3:
            if vals.shape[2] > 1:
                return vals[:, :, 1]
            return vals[:, :, 0]
        return vals

    vals = np.asarray(shap_values)
    if vals.ndim == 3:
        if vals.shape[2] > 1:
            return vals[:, :, 1]
        return vals[:, :, 0]
    return vals


def get_display_feature_order(bundle, schema):
    selected_processed = list(bundle["selected_features"])
    processed_to_raw = schema["processed_to_raw"]

    default_order = extract_selected_raw_features(selected_processed, processed_to_raw)

    # 1. 如果模型里已经保存了排序，优先用
    for key in ["raw_shap_order", "shap_feature_order_raw", "display_feature_order"]:
        if key in bundle:
            saved_order = [x for x in bundle[key] if x in default_order]
            remain = [x for x in default_order if x not in saved_order]
            if saved_order:
                return saved_order + remain

    # 2. 如果有 shap，就运行时计算一次原始变量级别的重要性
    if shap is not None and "X_train_final" in bundle:
        try:
            model = bundle["model"]
            model_name = bundle.get("model_name", "")
            X_train_final = bundle["X_train_final"].copy()

            if len(X_train_final) > 120:
                X_ref = X_train_final.sample(n=120, random_state=42)
            else:
                X_ref = X_train_final.copy()

            tree_models = {"DT", "RF", "ET", "GBM", "XGBoost", "LightGBM"}
            linear_models = {"LR"}

            if model_name in tree_models:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_ref)
            elif model_name in linear_models:
                background = X_ref.sample(n=min(80, len(X_ref)), random_state=42)
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_ref)
            else:
                raise ValueError("当前模型类型跳过 shap 排序")

            sv = _extract_shap_matrix(shap_values)
            mean_abs = np.abs(sv).mean(axis=0)

            processed_scores = dict(zip(X_ref.columns, mean_abs))
            raw_scores = aggregate_processed_scores_to_raw(processed_scores, processed_to_raw)

            ordered = sorted(default_order, key=lambda x: raw_scores.get(x, 0.0), reverse=True)
            return ordered

        except Exception:
            pass

    # 3. 如果有 feature_table，用其顺序聚合一下
    feature_table = bundle.get("feature_table", None)
    if isinstance(feature_table, pd.DataFrame) and "Feature" in feature_table.columns:
        processed_scores = {}
        for i, row in feature_table.reset_index(drop=True).iterrows():
            feat = row["Feature"]
            if "Abs_Coefficient" in feature_table.columns:
                score = float(row["Abs_Coefficient"])
            else:
                score = float(len(feature_table) - i)
            processed_scores[feat] = score

        raw_scores = aggregate_processed_scores_to_raw(processed_scores, processed_to_raw)
        ordered = sorted(default_order, key=lambda x: raw_scores.get(x, 0.0), reverse=True)
        return ordered

    # 4. 最后回退到最终模型顺序
    return default_order


def transform_input(raw_df, preprocessor, selected_features):
    X_processed = pd.DataFrame(
        preprocessor.transform(raw_df),
        columns=preprocessor.get_feature_names_out()
    )
    X_final = X_processed.loc[:, selected_features]
    return X_final


def format_option(x):
    if hasattr(x, "item"):
        x = x.item()
    return str(x)


def build_probability_bar(prob):
    prob = float(np.clip(prob, 0.0, 1.0))
    percent = prob * 100

    fill_color = "#ff2d78"
    if percent < 33:
        fill_color = "#5aa9ff"
    elif percent < 66:
        fill_color = "#ff9f43"

    html = f"""
    <div class="result-card">
        <div style="font-size:14px;color:#666;margin-bottom:10px;">Predicted probability</div>

        <div style="position:relative; width:100%; height:18px; background:#ececef;
                    border-radius:999px; overflow:hidden;">
            <div style="width:{percent:.2f}%; height:100%; background:{fill_color};
                        border-radius:999px;"></div>
        </div>

        <div style="display:flex; justify-content:space-between; margin-top:8px;
                    font-size:12px; color:#7a7a7a;">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
        </div>

        <div style="margin-top:12px; font-size:28px; font-style:italic; font-weight:700;
                    color:#2d2d2d;">
            {percent:.2f}%
        </div>
    </div>
    """
    return html


# =========================================================
# 5. 加载模型
# =========================================================
try:
    model_path = auto_find_model_path()
    bundle = load_bundle(model_path)
except Exception as e:
    st.error(str(e))
    st.stop()

model = bundle["model"]
preprocessor = bundle["preprocessor"]
selected_features = bundle["selected_features"]
schema = parse_preprocessor_schema(preprocessor)

selected_raw_features_order = get_display_feature_order(bundle, schema)

numeric_cols = set(schema["numeric_cols"])
categorical_cols = set(schema["categorical_cols"])

selected_numeric_features = [f for f in selected_raw_features_order if f in numeric_cols]
selected_categorical_features = [f for f in selected_raw_features_order if f in categorical_cols]


# =========================================================
# 6. 页面头部
# =========================================================
st.markdown('<div class="app-title">LBP Recurrence Prediction Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter the final model variables below to estimate the probability of low back pain recurrence. Variables are displayed in the order of feature importance.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="feature-tip">Only variables included in the final model are displayed.</div>',
    unsafe_allow_html=True
)


# =========================================================
# 7. 输入区
# =========================================================
input_values = {}

for col in selected_numeric_features:
    default_val = schema["numeric_defaults"].get(col, 0.0)
    step, fmt = infer_step_and_format(default_val)

    input_values[col] = st.number_input(
        pretty_label(col),
        value=float(default_val),
        step=float(step),
        format=fmt,
        key=f"num_{col}"
    )

for col in selected_categorical_features:
    options = schema["categorical_options"].get(col, [0, 1])
    default_val = schema["categorical_defaults"].get(col, options[0])

    try:
        default_index = options.index(default_val)
    except Exception:
        default_index = 0

    input_values[col] = st.selectbox(
        pretty_label(col),
        options=options,
        index=default_index,
        format_func=format_option,
        key=f"cat_{col}"
    )


# =========================================================
# 8. 预测按钮和结果
# =========================================================
if st.button("Predict"):
    try:
        all_raw_cols = schema["numeric_cols"] + schema["categorical_cols"]

        raw_input_dict = {}
        for c in all_raw_cols:
            if c in input_values:
                raw_input_dict[c] = input_values[c]
            elif c in schema["numeric_defaults"]:
                raw_input_dict[c] = schema["numeric_defaults"][c]
            elif c in schema["categorical_defaults"]:
                raw_input_dict[c] = schema["categorical_defaults"][c]
            else:
                raw_input_dict[c] = 0

        raw_df = pd.DataFrame([raw_input_dict])

        X_final = transform_input(raw_df, preprocessor, selected_features)
        prob = float(safe_predict_proba(model, X_final)[0])

        st.markdown(
            f'<div class="result-sentence">Based on feature values, predicted possibility of LBP recurrence is {prob * 100:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.components.v1.html(build_probability_bar(prob), height=160, scrolling=False)

        if prob >= 0.5:
            tip_text = "Prediction result suggests a relatively high recurrence risk."
        else:
            tip_text = "Prediction result suggests a relatively low recurrence risk."

        st.markdown(
            f'<div class="mini-caption">{tip_text}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"预测失败: {e}")


# =========================================================
# 9. 可折叠信息区
# =========================================================
with st.expander("Model information"):
    st.write(f"Model file: {model_path}")
    st.write(f"Displayed raw variables count: {len(selected_raw_features_order)}")
    st.write("Displayed raw variables:")
    st.write(selected_raw_features_order)
    st.write("Final processed features used by the model:")
    st.write(selected_features)
