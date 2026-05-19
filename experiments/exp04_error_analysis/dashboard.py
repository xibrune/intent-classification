"""
Streamlit-дашборд для Эксперимента №4: «Анатомия ошибок».

Запуск из корня репозитория:
    streamlit run experiments/exp04_error_analysis/dashboard.py

Загружает заранее подготовленные артефакты из results/analysis/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ── Пути ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "analysis"

# ── Цветовая палитра ─────────────────────────────────────────────────
METHOD_COLORS = {
    "Fine-Tuning": "#2ecc71",
    "Zero-Shot":   "#e74c3c",
    "Few-Shot":    "#3498db",
}

ERROR_TYPE_DESCRIPTIONS = {
    "magnet_intent":          "слабый метод сваливается в «дежурный» интент-магнит",
    "synonymous_intents":     "интенты-близнецы, отличающиеся одним словом",
    "ambiguous_phrasing":     "формулировка реально неоднозначна",
    "semantic_adjacency":     "семантическая близость интентов",
    "hierarchical_inclusion": "один интент — подвид другого",
}

ACCURACIES = {
    "Fine-Tuning":  0.9424,
    "Zero-Shot":    0.1688,
    "Few-Shot":     0.5282,
}

METHOD_LABELS = {
    "Fine-Tuning":  "DistilBERT, full fine-tune",
    "Zero-Shot":    "Flan-T5-base, Prompt 3 «вопрос»",
    "Few-Shot":     "Flan-T5-base, diverse 10-shot",
}


# ── Конфиг страницы ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Анатомия ошибок · CLINC150",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts():
    return {
        "df":            pd.read_csv(ARTIFACTS_DIR / "unified_predictions.csv"),
        "method_a":      pd.read_csv(ARTIFACTS_DIR / "method_a_pairs.csv"),
        "method_b":      pd.read_csv(ARTIFACTS_DIR / "method_b_pairs.csv"),
        "final_pairs":   pd.read_csv(ARTIFACTS_DIR / "final_pairs.csv"),
        "error_samples": pd.read_csv(ARTIFACTS_DIR / "error_samples.csv"),
        "umap":          pd.read_csv(ARTIFACTS_DIR / "umap_coords.csv"),
    }


try:
    data = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Не удалось загрузить артефакты из `{ARTIFACTS_DIR}`")
    st.exception(e)
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🔍 Анатомия ошибок")
st.sidebar.markdown("**CLINC150 · 151 интент · 3578 тестов**")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Раздел",
    ["Обзор", "Методология", "Топ-5 пар", "Типы ошибок", "UMAP-карта"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Сравнительный анализ Fine-Tuning, Zero-Shot и Few-Shot. "
    "Эксперимент №4 дипломной работы."
)


# ════════════════════════════════════════════════════════════════════
# Раздел: Обзор
# ════════════════════════════════════════════════════════════════════
def show_overview(data):
    st.title("Сравнительный анализ ошибок")
    st.markdown(
        "Лучшие варианты каждого подхода к классификации интентов "
        "на тестовой выборке CLINC150."
    )

    df = data["df"]
    n_test = len(df)

    col1, col2, col3 = st.columns(3)
    for col, method in zip([col1, col2, col3], ["Fine-Tuning", "Zero-Shot", "Few-Shot"]):
        with col:
            st.metric(METHOD_LABELS[method], f"{ACCURACIES[method]:.2%}")

    st.markdown("---")
    st.subheader("Структура ошибок")

    n_ft = int((df["correct_ft"] == 0).sum())
    n_zs = int((df["correct_zs"] == 0).sum())
    n_fs = int((df["correct_fs"] == 0).sum())
    n_all = int(
        ((df["correct_ft"] == 0) &
         (df["correct_zs"] == 0) &
         (df["correct_fs"] == 0)).sum()
    )

    n_pairs_ft = df[df["correct_ft"] == 0].groupby(["true_intent", "pred_ft"]).ngroups
    n_pairs_zs = df[df["correct_zs"] == 0].groupby(["true_intent", "pred_zs"]).ngroups
    n_pairs_fs = df[df["correct_fs"] == 0].groupby(["true_intent", "pred_fs"]).ngroups

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Fine-Tuning", "Zero-Shot", "Few-Shot", "Все три"],
            y=[n_ft, n_zs, n_fs, n_all],
            marker_color=[
                METHOD_COLORS["Fine-Tuning"],
                METHOD_COLORS["Zero-Shot"],
                METHOD_COLORS["Few-Shot"],
                "#9b59b6",
            ],
            text=[f"<b>{n}</b><br>{n/n_test:.1%}" for n in [n_ft, n_zs, n_fs, n_all]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Число ошибочных примеров (из {n_test})",
            height=420,
            showlegend=False,
            yaxis_title="Ошибок",
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        ### Ключевые цифры

        - **Тестовых примеров:** {n_test}
        - **Уникальных пар ошибок:**
          - Fine-Tuning — {n_pairs_ft}
          - Zero-Shot — {n_pairs_zs}
          - Few-Shot — {n_pairs_fs}
        - **Пар, где ошиблись ВСЕ три метода:** {len(data["method_a"])}
        - **Финальный пул спорных пар:** {len(data["final_pairs"])}

        ### Подход к анализу

        Подробно — в разделе **«Методология»**. Кратко:

        - Использованы **два независимых метода** выявления спорных пар
        - **Финальный пул** = объединение топ-5 двух методов
        - Каждый из 64 ошибочных примеров отнесён вручную к одному из 5 типов
        """)


# ════════════════════════════════════════════════════════════════════
# Раздел: Методология
# ════════════════════════════════════════════════════════════════════
def show_methodology(data):
    st.title("Как выбраны спорные пары")
    st.markdown(
        "Из суммарно ~2000 уникальных пар ошибок нужно выделить те, "
        "что отражают **реальную семантическую сложность интентов**, "
        "а не случайные сбои одного метода. Для этого использованы "
        "два методологически независимых подхода — пары, на которых "
        "они согласны, считаются наиболее надёжными."
    )

    method_a = data["method_a"]
    method_b = data["method_b"]
    final    = data["final_pairs"].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Метод 1 — пересечение ошибок")
        st.markdown(
            "Пары `(true → pred)`, на которых ошибаются **все три метода** "
            "хотя бы по разу. Если даже сильный Fine-Tuning путает пару, "
            "сложность объективна. Ранжируем по суммарному числу ошибок."
        )

        top_a = method_a.head(10).copy()
        top_a["pair"] = top_a["true_intent"] + " → " + top_a["pred_intent"]

        fig = go.Figure()
        for col_key, name, color in [
            ("n_ft", "Fine-Tuning", METHOD_COLORS["Fine-Tuning"]),
            ("n_zs", "Zero-Shot",   METHOD_COLORS["Zero-Shot"]),
            ("n_fs", "Few-Shot",    METHOD_COLORS["Few-Shot"]),
        ]:
            fig.add_trace(go.Bar(
                y=top_a["pair"],
                x=top_a[col_key],
                name=name,
                orientation="h",
                marker_color=color,
            ))
        fig.update_layout(
            barmode="stack",
            height=460,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Число ошибок (стек по методам)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Всего пар в пересечении: {len(method_a)}. "
            "**Верхние 5 → в финальный пул**."
        )

    with col2:
        st.subheader("Метод 2 — агрегированная частота")
        st.markdown(
            "Confusion matrices нормализованы построчно "
            "(доля ошибок внутри класса), три CM суммируются. "
            "Учитываются пары, где ошибаются **≥2 методов из 3** — "
            "иначе топ занимают магнит-провалы одного Zero-Shot."
        )

        top_b = method_b.head(10).copy()
        top_b["pair"] = top_b["true_intent"] + " → " + top_b["pred_intent"]

        fig = go.Figure()
        for col_key, name, color in [
            ("rate_ft", "Fine-Tuning", METHOD_COLORS["Fine-Tuning"]),
            ("rate_zs", "Zero-Shot",   METHOD_COLORS["Zero-Shot"]),
            ("rate_fs", "Few-Shot",    METHOD_COLORS["Few-Shot"]),
        ]:
            fig.add_trace(go.Bar(
                y=top_b["pair"],
                x=top_b[col_key],
                name=name,
                orientation="h",
                marker_color=color,
            ))
        fig.update_layout(
            barmode="stack",
            height=460,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Доля ошибок (стек, max=3.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Фильтр «≥2 методов» отсеивает одиночные провалы. "
            "**Верхние 5 → в финальный пул**."
        )

    st.markdown("---")

    st.subheader("Финальный пул = объединение двух топ-5")

    n_robust   = int(final["is_robust"].sum())
    n_int_only = int((final["in_intersection"] & ~final["is_robust"]).sum())
    n_agg_only = int((final["in_aggregated"]  & ~final["is_robust"]).sum())

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "🔒 Согласованные",
            n_robust,
            help="В обоих топах одновременно — самые надёжные",
        )
    with cols[1]:
        st.metric(
            "📊 Только в пересечении",
            n_int_only,
            help="Универсально сложные, но не на доминирующей доле",
        )
    with cols[2]:
        st.metric(
            "📈 Только в агрегированной",
            n_agg_only,
            help="Большая доля ошибок, но не у всех трёх методов",
        )

    def get_source(row):
        if row["is_robust"]:
            return "🔒 Согласованная"
        if row["in_intersection"]:
            return "📊 Метод пересечения"
        return "📈 Метод агрегированной"

    final["Источник"] = final.apply(get_source, axis=1)
    final["Пара"] = final["true_intent"] + " → " + final["pred_intent"]
    final["_sort"] = (
        final["is_robust"].astype(int) * -10
        + final["in_intersection"].astype(int) * -1
    )

    st.dataframe(
        final.sort_values("_sort")[["Пара", "Источник"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Согласованные пары** — самые надёжные индикаторы реальной "
        "семантической сложности: их выделили два независимых метода. "
    )


# ════════════════════════════════════════════════════════════════════
# Раздел: Топ-5 пар
# ════════════════════════════════════════════════════════════════════
def show_top_pairs(data):
    st.title("Разбор финальных пар")
    st.markdown(
        "Восемь пар из финального пула (см. **Методология**). "
        "Выбери пару — увидишь статистику и конкретные ошибочные примеры из теста."
    )

    final_pairs = data["final_pairs"].copy()
    final_pairs["display"] = final_pairs.apply(
        lambda r: f"{r['true_intent']} → {r['pred_intent']}"
                  + ("  ✓ согласованная" if r["is_robust"] else ""),
        axis=1,
    )

    selected_display = st.selectbox(
        "Пара для разбора",
        final_pairs["display"].tolist(),
    )
    sel_row = final_pairs[final_pairs["display"] == selected_display].iloc[0]
    true_i, pred_i = sel_row["true_intent"], sel_row["pred_intent"]
    pair_str = f"{true_i} → {pred_i}"

    badges = []
    if sel_row["is_robust"]:
        badges.append("🔒 **Согласованная** — попала в оба топа")
    if sel_row["in_intersection"]:
        badges.append("· Метод пересечения ошибок")
    if sel_row["in_aggregated"]:
        badges.append("· Метод агрегированной частоты")
    st.markdown("  ".join(badges))

    st.markdown("---")

    df = data["df"]
    class_examples = df[df["true_intent"] == true_i]
    n_class = len(class_examples)
    n_ft_err = (class_examples["pred_ft"] == pred_i).sum()
    n_zs_err = (class_examples["pred_zs"] == pred_i).sum()
    n_fs_err = (class_examples["pred_fs"] == pred_i).sum()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Fine-Tuning", "Zero-Shot", "Few-Shot"],
            y=[n_ft_err, n_zs_err, n_fs_err],
            marker_color=[
                METHOD_COLORS["Fine-Tuning"],
                METHOD_COLORS["Zero-Shot"],
                METHOD_COLORS["Few-Shot"],
            ],
            text=[
                f"<b>{n}</b><br>{n/n_class:.0%}"
                for n in [n_ft_err, n_zs_err, n_fs_err]
            ],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Ошибок «{pair_str}» из {n_class} примеров класса {true_i}",
            height=420,
            showlegend=False,
            yaxis_title="Ошибок",
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        ### Сводка по паре

        | Метод | Ошибок | Доля класса |
        |---|---:|---:|
        | Fine-Tuning | {n_ft_err} | {n_ft_err/n_class:.1%} |
        | Zero-Shot | {n_zs_err} | {n_zs_err/n_class:.1%} |
        | Few-Shot | {n_fs_err} | {n_fs_err/n_class:.1%} |

        **Размер класса в тесте:** {n_class} примеров
        """)

        pair_samples = data["error_samples"][data["error_samples"]["pair"] == pair_str]
        if len(pair_samples) > 0 and pair_samples["error_type"].notna().any():
            type_counts = pair_samples["error_type"].value_counts()
            type_lines = [f"- **{t}**: {c}" for t, c in type_counts.items()]
            st.markdown("### Типы ошибок на этой паре\n" + "\n".join(type_lines))

    st.markdown("---")

    st.subheader(f"Ошибочные примеры для пары «{pair_str}»")

    pair_samples = data["error_samples"][data["error_samples"]["pair"] == pair_str].copy()

    if len(pair_samples) == 0:
        st.info("Для этой пары нет сэмплированных примеров.")
        return

    show_cols = ["text", "methods_made_error", "n_methods", "error_type", "notes",
                 "pred_ft", "pred_zs", "pred_fs"]
    show_cols = [c for c in show_cols if c in pair_samples.columns]

    st.dataframe(
        pair_samples[show_cols].rename(columns={
            "text":               "Текст",
            "methods_made_error": "Кто ошибся",
            "n_methods":          "N методов",
            "error_type":         "Тип ошибки",
            "notes":              "Заметки",
            "pred_ft":            "FT",
            "pred_zs":            "ZS",
            "pred_fs":            "FS",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════
# Раздел: Типы ошибок
# ════════════════════════════════════════════════════════════════════
def show_error_types(data):
    st.title("Типы ошибок: таксономия и распределение")
    st.markdown(
        "Каждая из 64 сэмплированных ошибок отнесена вручную к одному "
        "из 5 типов на основе анализа конкретного текста."
    )

    samples = data["error_samples"]

    if samples["error_type"].isna().all() or (samples["error_type"] == "").all():
        st.warning(
            "Колонка `error_type` пуста — ручная разметка ещё не выполнена. "
            "Заполни её в `results/analysis/error_samples.csv`."
        )
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        type_counts = samples["error_type"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.45,
            textinfo="label+percent+value",
            marker=dict(line=dict(color="white", width=2)),
        )])
        fig.update_layout(
            title=f"Распределение {len(samples)} ошибок по типам",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Расшифровка типов")
        for err_type, desc in ERROR_TYPE_DESCRIPTIONS.items():
            count = int(type_counts.get(err_type, 0))
            pct = count / len(samples) * 100 if len(samples) > 0 else 0
            st.markdown(
                f"**{err_type}** · {count} ({pct:.0f}%)  \n"
                f"<small>{desc}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")

    st.subheader("Какие типы ошибок встречаются в каких парах")
    pivot = (
        samples.groupby(["pair", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues",
        text=pivot.values,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorbar=dict(title="Примеров"),
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Тип ошибки",
        yaxis_title="Пара",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Все примеры с фильтрами")

    all_types = sorted(samples["error_type"].dropna().unique())
    selected_types = st.multiselect(
        "Тип ошибки",
        all_types,
        default=all_types,
    )

    filtered = samples[samples["error_type"].isin(selected_types)]
    st.caption(f"Показано {len(filtered)} из {len(samples)} примеров")

    show_cols = ["pair", "text", "methods_made_error", "error_type", "notes"]
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].rename(columns={
            "pair":               "Пара",
            "text":               "Текст",
            "methods_made_error": "Кто ошибся",
            "error_type":         "Тип ошибки",
            "notes":              "Заметки",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════
# Раздел: UMAP-карта
# ════════════════════════════════════════════════════════════════════
def show_umap(data):
    st.title("UMAP-карта вовлечённых интентов")
    st.markdown(
        "Эмбеддинги от `sentence-transformers/all-MiniLM-L6-v2`, "
        "спроектированные в 2D через UMAP. Каждая точка — utterance "
        "из train+val+test (~150 на интент). "
        "Наведи на точку, чтобы увидеть текст."
    )

    umap_df = data["umap"]
    final_pairs = data["final_pairs"]

    col1, col2 = st.columns([1, 2])

    with col1:
        mode = st.radio(
            "Что показать",
            ["Все интенты", "Подсветить пару"],
        )

    if mode == "Подсветить пару":
        with col2:
            pair_str_list = [
                f"{r['true_intent']} → {r['pred_intent']}"
                for _, r in final_pairs.iterrows()
            ]
            selected = st.selectbox("Пара", pair_str_list)
            true_i, _, pred_i = selected.partition(" → ")

        umap_df = umap_df.copy()
        umap_df["group"] = umap_df["intent_name"].apply(
            lambda x: x if x in (true_i, pred_i) else "прочие"
        )
        color_map = {
            true_i: "#e74c3c",
            pred_i: "#3498db",
            "прочие": "#cccccc",
        }
        fig = px.scatter(
            umap_df,
            x="umap_x",
            y="umap_y",
            color="group",
            color_discrete_map=color_map,
            hover_data={"text": True, "intent_name": True,
                        "umap_x": False, "umap_y": False, "group": False},
            height=700,
            category_orders={"group": ["прочие", true_i, pred_i]},
        )
    else:
        fig = px.scatter(
            umap_df,
            x="umap_x",
            y="umap_y",
            color="intent_name",
            hover_data={"text": True, "intent_name": True,
                        "umap_x": False, "umap_y": False},
            height=700,
            color_discrete_sequence=px.colors.qualitative.Light24,
        )

    fig.update_traces(marker=dict(
        size=7,
        opacity=0.75,
        line=dict(width=0.4, color="white"),
    ))
    fig.update_layout(
        legend=dict(title="Интент"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("Как читать эту карту"):
        st.markdown("""
        - **Перекрывающиеся кластеры** (`calendar` ↔ `calendar_update`,
          `cook_time` ↔ `food_last`) — пары, где интенты семантически
          очень близки. Объясняет ошибки `synonymous_intents`
          и `semantic_adjacency`.
        - **Удалённые кластеры** (`travel_suggestion` ↔ `fun_fact`) —
          пары, которые тем не менее путаются у Zero-Shot.
          Подтверждает гипотезу `magnet_intent`: ошибка обусловлена
          поведением модели, а не геометрией данных.
        - **`oil_change_when` ↔ `oil_change_how`** — эмбеддинг-модель
          различает их по синтаксической форме вопроса (when/how),
          хотя интенты семантически идентичны. Слабые классификаторы
          этот синтаксический сигнал игнорируют.
        """)


# ════════════════════════════════════════════════════════════════════
# Маршрутизация
# ════════════════════════════════════════════════════════════════════
if section == "Обзор":
    show_overview(data)
elif section == "Методология":
    show_methodology(data)
elif section == "Топ-5 пар":
    show_top_pairs(data)
elif section == "Типы ошибок":
    show_error_types(data)
elif section == "UMAP-карта":
    show_umap(data)