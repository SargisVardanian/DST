from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "Common_code"
RUNNER = COMMON / "test_Ripper_DST.py"
INSPECTOR = COMMON / "sample_rule_inspector.py"


def run_cmd(args: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


def zip_dir_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))
    buf.seek(0)
    return buf.read()


st.set_page_config(page_title="DST Trainer", layout="wide")
st.title("DST Rule Trainer")
st.write("Train FOIL/RIPPER rule models on your CSV, save rules and learned weights, and inspect predictions on examples.")

tab_train, tab_inspect = st.tabs(["Train", "Inspect"])

with tab_train:
    st.subheader("Train On A CSV")
    uploaded = st.file_uploader("Dataset CSV", type=["csv"])
    default_out = str((ROOT / "artifacts" / "app_run").resolve())
    save_root = st.text_input("Output directory", value=default_out)
    inducers = st.multiselect("Inducers", ["RIPPER", "FOIL", "STATIC"], default=["RIPPER", "FOIL"])
    max_epochs = st.number_input("Max epochs", min_value=1, max_value=500, value=100)
    test_size = st.number_input("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=100000, value=42)
    paper_mode = st.checkbox("Paper mode", value=False)

    if st.button("Run Training", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Upload a CSV file first.")
        elif not inducers:
            st.error("Select at least one inducer.")
        else:
            out_root = Path(save_root).expanduser().resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            uploads_dir = out_root / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = uploads_dir / uploaded.name
            dataset_path.write_bytes(uploaded.getbuffer())

            cmd = [
                sys.executable,
                str(RUNNER),
                "--dataset-path",
                str(dataset_path),
                "--inducers",
                ",".join(inducers),
                "--save-root",
                str(out_root),
                "--max-epochs",
                str(int(max_epochs)),
                "--test-size",
                str(float(test_size)),
                "--seeds",
                str(int(seed)),
            ]
            if not paper_mode:
                cmd.append("--no-paper-mode")

            with st.spinner("Training..."):
                code, output = run_cmd(cmd)

            st.code(output or "(no output)", language="text")
            if code != 0:
                st.error("Training failed.")
            else:
                st.success("Training finished.")
                st.write(f"Saved to `{out_root}`")
                st.write("Main outputs:")
                st.write(f"- `{out_root / 'rules'}`")
                st.write(f"- `{out_root / 'models'}`")
                st.write(f"- `{out_root / 'benchmarks'}`")
                zip_bytes = zip_dir_bytes(out_root)
                st.download_button(
                    "Download Outputs ZIP",
                    data=zip_bytes,
                    file_name=f"{dataset_path.stem}_dst_outputs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

with tab_inspect:
    st.subheader("Inspect A Saved Model")
    dataset_path = st.text_input("Dataset path", value="")
    model_algo = st.selectbox("Algorithm", ["RIPPER", "FOIL", "STATIC"], index=0)
    model_path = st.text_input("Model path (.pkl or .dsb)", value="")
    row_index = st.number_input("Row index", min_value=0, max_value=10_000_000, value=0)
    split = st.selectbox("Split", ["test", "train", "full"], index=0)
    combine_rule = st.selectbox("Combine rule", ["dempster", "yager", "vote"], index=0)

    if st.button("Inspect Sample", use_container_width=True):
        if not dataset_path.strip() or not model_path.strip():
            st.error("Provide both dataset path and model path.")
        else:
            cmd = [
                sys.executable,
                str(INSPECTOR),
                "--dataset",
                dataset_path.strip(),
                "--algo",
                model_algo,
                "--model",
                model_path.strip(),
                "--row-index",
                str(int(row_index)),
                "--split",
                split,
                "--combine-rule",
                combine_rule,
            ]
            with st.spinner("Inspecting..."):
                code, output = run_cmd(cmd)
            st.code(output or "(no output)", language="text")
            if code != 0:
                st.error("Inspection failed.")
