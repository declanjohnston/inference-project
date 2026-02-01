#!/usr/bin/env python3
"""Streamlit dashboard for comparing base vs fine-tuned model responses."""
import json
from pathlib import Path

import streamlit as st

OUTPUT_DIR = Path("data/outputs")


def load_responses(model_dir: Path) -> tuple[list[dict], list[dict]] | None:
    """Load base and finetuned responses from a model output directory."""
    base_path = model_dir / "base_responses.json"
    finetuned_path = model_dir / "finetuned_responses.json"

    if not base_path.exists() or not finetuned_path.exists():
        return None

    base = json.loads(base_path.read_text())
    finetuned = json.loads(finetuned_path.read_text())
    return base, finetuned


def get_available_models() -> list[str]:
    """Get list of model directories with cached outputs."""
    if not OUTPUT_DIR.exists():
        return []
    return [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]


def main():
    st.set_page_config(
        page_title="Model Comparison Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Base vs Fine-tuned Model Comparison")
    st.markdown("Compare responses from the base Gemma-2-2B model with fine-tuned versions.")

    models = get_available_models()

    if not models:
        st.error("No model outputs found. Run `python run_inference.py` first.")
        st.stop()

    selected_model = st.sidebar.selectbox(
        "Select Fine-tuned Model",
        models,
        help="Choose which fine-tuned model's outputs to compare",
    )

    model_dir = OUTPUT_DIR / selected_model
    data = load_responses(model_dir)

    if data is None:
        st.error(f"Could not load responses for {selected_model}")
        st.stop()

    base_responses, finetuned_responses = data

    if len(base_responses) != len(finetuned_responses):
        st.warning("Mismatch in number of responses between base and fine-tuned models.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Queries:** {len(base_responses)}")

    query_options = [
        f"Q{i+1}: {r['instruction'][:50]}..."
        for i, r in enumerate(base_responses)
    ]
    selected_idx = st.sidebar.radio(
        "Select Query",
        range(len(query_options)),
        format_func=lambda i: query_options[i],
    )

    query = base_responses[selected_idx]
    base_response = base_responses[selected_idx]["response"]
    finetuned_response = finetuned_responses[selected_idx]["response"]

    st.markdown("---")

    st.subheader(f"Query {selected_idx + 1}")
    st.markdown(f"**Instruction:** {query['instruction']}")
    if query.get("input"):
        st.markdown(f"**Input:** {query['input']}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Base Model")
        st.markdown("`google/gemma-2-2b`")
        st.markdown(base_response)

    with col2:
        st.markdown("### 🟢 Fine-tuned Model")
        st.markdown(f"`{selected_model}`")
        st.markdown(finetuned_response)

    st.markdown("---")

    with st.expander("View All Responses"):
        for i, (base, ft) in enumerate(zip(base_responses, finetuned_responses)):
            st.markdown(f"### Query {i + 1}: {base['instruction']}")
            if base.get("input"):
                st.markdown(f"*Input: {base['input']}*")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Base Model:**")
                st.text(base["response"][:500] + "..." if len(base["response"]) > 500 else base["response"])
            with c2:
                st.markdown("**Fine-tuned:**")
                st.text(ft["response"][:500] + "..." if len(ft["response"]) > 500 else ft["response"])
            st.markdown("---")


if __name__ == "__main__":
    main()
