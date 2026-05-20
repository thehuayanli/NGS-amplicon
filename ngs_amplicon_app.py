"""
Streamlit dashboard for processing amplicon-seq FASTQ reads from phage-display libraries.

Original script behavior reproduced and extended:
- Extracts DNA inserts between fixed upstream/downstream phage anchor sequences.
- Translates DNA inserts to amino-acid sequences.
- Counts unique peptide IDs and percentages.
- Reports QC categories: both anchors, one anchor, no anchor, short reads, bad FASTQ records.

Run locally:
    streamlit run ngs_streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st


# -----------------------------
# Constants from original script
# -----------------------------
DEFAULT_UPSTREAM = "CCTTTCTATTCTCACTCT"
DEFAULT_DOWNSTREAM = "GCCGAAACTGTTGAAAGT"
DEFAULT_MIN_READ_LENGTH = 100

CODON_TABLE: Dict[str, str] = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
    "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
    "TAC": "Y", "TAT": "Y", "TAA": "*", "TAG": "*",
    "TGC": "C", "TGT": "C", "TGA": "*", "TGG": "W",
}

DEMO_FASTQ = """@read_001
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTGGTGCTTGTGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_002
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTGGTGCTTGTGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_003
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTGATGCTTGTGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_004
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTGGTGCTTGNNNNGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_005
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTAAGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_006
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTAAGTGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_007
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTCCTTTCTATTCTCACTCTTGTGCTTGTGGTGCTTGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_008
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTGCCGAAACTGTTGAAAGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_009
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
@read_010
ACGTCCTTTCTATTCTCACTCTTGTGCTTGTGGTGCCGAAACTGTTGAAAGTACGT
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
"""


@dataclass
class ReadRecord:
    read_id: str
    sequence: str
    quality: str = ""


@dataclass
class ProcessResult:
    translation_df: pd.DataFrame
    unique_df: pd.DataFrame
    qc: Dict[str, int]
    bad_lines: List[str]
    read_class_df: pd.DataFrame
    insert_length_df: pd.DataFrame


def normalize_sequence(seq: str) -> str:
    """Uppercase sequence and remove whitespace."""
    return "".join(seq.upper().split())


def translate_dna(dna_seq: str) -> Tuple[str, str]:
    """
    Translate DNA to amino acid using the original codon table.

    Returns:
        (amino_acid_sequence, translation_status)
    """
    seq = normalize_sequence(dna_seq)
    if len(seq) % 3 != 0:
        return "failed", "out_of_frame"

    aa_seq = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if "N" in codon:
            aa_seq.append("X")
        elif codon in CODON_TABLE:
            aa_seq.append(CODON_TABLE[codon])
        else:
            aa_seq.append("?")

    aa = "".join(aa_seq)
    if "?" in aa:
        status = "invalid_codon"
    elif "*" in aa:
        status = "stop_codon"
    elif "X" in aa:
        status = "ambiguous_N"
    else:
        status = "translated"
    return aa, status


def parse_fastq_strict(text: str) -> Tuple[List[ReadRecord], List[str]]:
    """Parse standard 4-line FASTQ records."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reads: List[ReadRecord] = []
    bad_records: List[str] = []

    i = 0
    while i < len(lines):
        block = lines[i : i + 4]
        if len(block) < 4:
            bad_records.append(" | ".join(block))
            break

        header, sequence, plus, quality = block
        if header.startswith("@") and plus.startswith("+"):
            reads.append(
                ReadRecord(
                    read_id=header[1:] or f"read_{len(reads) + 1}",
                    sequence=normalize_sequence(sequence),
                    quality=quality,
                )
            )
        else:
            bad_records.append(" | ".join(block))
        i += 4

    return reads, bad_records


def parse_fastq_legacy_line_filter(text: str) -> Tuple[List[ReadRecord], List[str]]:
    """
    Reproduce the original script's line-filtering behavior.

    Caveat: this can accidentally treat quality lines as sequence lines if a quality line starts
    with A/T/C/G/N, so strict FASTQ parsing is recommended for production use.
    """
    sequence_starts = {"A", "T", "C", "G", "N"}
    ignored_starts = {":", "@", "F", "+", ",", "#"}

    reads: List[ReadRecord] = []
    bad_lines: List[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        first = line[0]
        if first in sequence_starts:
            reads.append(ReadRecord(read_id=f"line_{line_number}", sequence=normalize_sequence(line)))
        elif first in ignored_starts:
            continue
        else:
            bad_lines.append(line)
    return reads, bad_lines


def process_reads(
    text: str,
    upstream: str,
    downstream: str,
    min_read_length: int = DEFAULT_MIN_READ_LENGTH,
    parser_mode: str = "Strict FASTQ",
) -> ProcessResult:
    """Core NGS processing pipeline."""
    upstream = normalize_sequence(upstream)
    downstream = normalize_sequence(downstream)

    if parser_mode == "Legacy line filter":
        reads, bad_lines = parse_fastq_legacy_line_filter(text)
    else:
        reads, bad_lines = parse_fastq_strict(text)

    extracted_rows: List[Dict[str, object]] = []
    qc = {
        "total_reads": len(reads),
        "short_reads": 0,
        "both_anchors": 0,
        "one_anchor": 0,
        "no_anchor": 0,
        "bad_records_or_lines": len(bad_lines),
        "translated": 0,
        "ambiguous_N": 0,
        "stop_codon": 0,
        "invalid_codon": 0,
        "out_of_frame": 0,
    }

    for read in reads:
        seq = read.sequence
        if len(seq) <= min_read_length:
            qc["short_reads"] += 1
            continue

        up_index = seq.find(upstream)
        down_index = seq.find(downstream)
        has_upstream = up_index != -1
        has_downstream = down_index != -1

        if has_upstream and has_downstream and down_index > up_index + len(upstream):
            insert_start = up_index + len(upstream)
            insert = seq[insert_start:down_index]
            aa, status = translate_dna(insert)
            qc["both_anchors"] += 1
            qc[status] = qc.get(status, 0) + 1

            extracted_rows.append(
                {
                    "Number": len(extracted_rows) + 1,
                    "Read ID": read.read_id,
                    "Read length": len(seq),
                    "Nucleotide Sequence": insert,
                    "Insert length (nt)": len(insert),
                    "Amino Acid Sequence": aa,
                    "AA length": 0 if aa == "failed" else len(aa),
                    "Translation status": status,
                }
            )
        elif has_upstream or has_downstream:
            qc["one_anchor"] += 1
        else:
            qc["no_anchor"] += 1

    translation_df = pd.DataFrame(extracted_rows)
    if translation_df.empty:
        unique_df = pd.DataFrame(columns=["Number", "Unique Sequence", "Counts", "Percentage", "Translation status"])
        insert_length_df = pd.DataFrame(columns=["Insert length (nt)", "Count"])
    else:
        total_valid = len(translation_df)
        unique_df = (
            translation_df.groupby(["Amino Acid Sequence", "Translation status"], dropna=False)
            .size()
            .reset_index(name="Counts")
            .sort_values(["Counts", "Amino Acid Sequence"], ascending=[False, True])
            .reset_index(drop=True)
        )
        unique_df["Percentage"] = unique_df["Counts"] / total_valid * 100
        unique_df.insert(0, "Number", unique_df.index + 1)
        unique_df = unique_df.rename(columns={"Amino Acid Sequence": "Unique Sequence"})

        insert_length_df = (
            translation_df.groupby("Insert length (nt)")
            .size()
            .reset_index(name="Count")
            .sort_values("Insert length (nt)")
        )

    read_class_df = pd.DataFrame(
        [
            {"Read class": "Both anchors", "Count": qc["both_anchors"]},
            {"Read class": "One anchor only", "Count": qc["one_anchor"]},
            {"Read class": "No anchors", "Count": qc["no_anchor"]},
            {"Read class": "Short reads", "Count": qc["short_reads"]},
            {"Read class": "Bad records/lines", "Count": qc["bad_records_or_lines"]},
        ]
    )

    return ProcessResult(
        translation_df=translation_df,
        unique_df=unique_df,
        qc=qc,
        bad_lines=bad_lines,
        read_class_df=read_class_df,
        insert_length_df=insert_length_df,
    )


def dataframe_to_tsv(df: pd.DataFrame) -> bytes:
    return df.to_csv(sep="\t", index=False).encode("utf-8")


def clean_stem(filename: str) -> str:
    for suffix in [".fastq", ".fq", ".txt", ".gz"]:
        if filename.lower().endswith(suffix):
            filename = filename[: -len(suffix)]
    return filename or "ngs_output"


def render_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, tooltip: Iterable[str] | None = None) -> None:
    if df.empty:
        st.info("No data to plot.")
        return
    tooltip = list(tooltip or [x, y])
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y),
            tooltip=tooltip,
        )
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty:
        st.info("No data to plot.")
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x),
            y=alt.Y(y),
            tooltip=[x, y],
        )
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Amplicon NGS Phage Display Dashboard",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Amplicon NGS Phage Display Dashboard")
    st.caption(
        "Extract inserts between phage anchor sequences, translate DNA to peptide IDs, "
        "rank unique amino-acid sequences, and review QC metrics."
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload FASTQ", type=["fastq", "fq", "txt"])
        use_demo = st.checkbox("Use demo data when no file is uploaded", value=True)

        st.header("Processing settings")
        upstream = st.text_input("Upstream anchor", value=DEFAULT_UPSTREAM).upper()
        downstream = st.text_input("Downstream anchor", value=DEFAULT_DOWNSTREAM).upper()
        min_read_length = st.number_input("Minimum read length", min_value=0, value=DEFAULT_MIN_READ_LENGTH, step=1)
        parser_mode = st.radio(
            "FASTQ parsing mode",
            options=["Strict FASTQ", "Legacy line filter"],
            index=0,
            help="Strict FASTQ is safer. Legacy line filter reproduces the original script's behavior.",
        )

        st.header("Display settings")
        top_n = st.slider("Top peptide IDs to plot", min_value=5, max_value=100, value=25, step=5)

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        try:
            fastq_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            fastq_text = raw_bytes.decode("latin-1")
        file_stem = clean_stem(uploaded_file.name)
    elif use_demo:
        fastq_text = DEMO_FASTQ
        file_stem = "demo"
    else:
        fastq_text = ""
        file_stem = "ngs_output"

    if not fastq_text:
        st.info("Upload a FASTQ file or enable demo data to start.")
        return

    result = process_reads(
        text=fastq_text,
        upstream=upstream,
        downstream=downstream,
        min_read_length=int(min_read_length),
        parser_mode=parser_mode,
    )

    st.subheader("QC summary")
    metric_cols = st.columns(5)
    total_reads = result.qc["total_reads"]
    both = result.qc["both_anchors"]
    one = result.qc["one_anchor"]
    unique_count = len(result.unique_df)
    warning_count = (
        result.qc["ambiguous_N"]
        + result.qc["stop_codon"]
        + result.qc["invalid_codon"]
        + result.qc["out_of_frame"]
    )

    metric_cols[0].metric("Parsed reads", f"{total_reads:,}")
    metric_cols[1].metric("Both anchors", f"{both:,}", f"{(both / total_reads * 100):.1f}%" if total_reads else "0.0%")
    metric_cols[2].metric("One anchor only", f"{one:,}")
    metric_cols[3].metric("Unique peptide IDs", f"{unique_count:,}")
    metric_cols[4].metric("Translation warnings", f"{warning_count:,}")

    if parser_mode == "Legacy line filter":
        st.warning(
            "Legacy mode reproduces the original script, but it can misclassify FASTQ quality lines "
            "as sequence lines if they start with A/T/C/G/N. Use Strict FASTQ mode for production analysis."
        )

    if not upstream or not downstream:
        st.error("Both upstream and downstream anchor sequences are required.")
        return

    if both == 0:
        st.error(
            "No reads with both anchors were found. Check anchor sequences, read orientation, minimum read length, "
            "or whether the uploaded file is the correct FASTQ."
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_bar_chart(
            result.read_class_df,
            x="Read class",
            y="Count",
            title="Read classification",
            tooltip=["Read class", "Count"],
        )
    with chart_col2:
        render_line_chart(
            result.insert_length_df,
            x="Insert length (nt)",
            y="Count",
            title="Insert length distribution",
        )

    st.subheader("Top unique peptide IDs")
    if result.unique_df.empty:
        st.info("No unique peptide IDs to display.")
    else:
        top_df = result.unique_df.head(top_n).copy()
        top_df["Percentage"] = top_df["Percentage"].round(4)
        render_bar_chart(
            top_df,
            x="Unique Sequence",
            y="Counts",
            title=f"Top {min(top_n, len(top_df))} unique amino-acid sequences",
            tooltip=["Number", "Unique Sequence", "Counts", "Percentage", "Translation status"],
        )

    tab_unique, tab_translation, tab_bad, tab_raw = st.tabs(
        ["Unique IDs", "All translations", "Bad records/lines", "Raw FASTQ preview"]
    )

    with tab_unique:
        st.write("Equivalent to the original `*_uniqueIDs_output.txt`, with an added translation-status column.")
        search = st.text_input("Search unique sequence or status", key="unique_search").strip().upper()
        unique_view = result.unique_df.copy()
        if not unique_view.empty:
            unique_view["Percentage"] = unique_view["Percentage"].round(6)
            if search:
                mask = (
                    unique_view["Unique Sequence"].str.upper().str.contains(search, regex=False)
                    | unique_view["Translation status"].str.upper().str.contains(search, regex=False)
                )
                unique_view = unique_view[mask]
        st.dataframe(unique_view, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download unique IDs TSV",
            data=dataframe_to_tsv(result.unique_df),
            file_name=f"{file_stem}_uniqueIDs_output.tsv",
            mime="text/tab-separated-values",
            disabled=result.unique_df.empty,
        )

    with tab_translation:
        st.write("Equivalent to the original `*_translation_output.txt`, with added read ID, length, and status fields.")
        st.dataframe(result.translation_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download translation TSV",
            data=dataframe_to_tsv(result.translation_df),
            file_name=f"{file_stem}_translation_output.tsv",
            mime="text/tab-separated-values",
            disabled=result.translation_df.empty,
        )

    with tab_bad:
        if result.bad_lines:
            st.write(f"Found {len(result.bad_lines)} bad FASTQ records/lines.")
            bad_df = pd.DataFrame({"Bad record/line": result.bad_lines})
            st.dataframe(bad_df, use_container_width=True, hide_index=True)
        else:
            st.success("No bad FASTQ records/lines detected by the selected parser.")

    with tab_raw:
        st.text_area("FASTQ text", value=fastq_text[:20000], height=350, help="Preview is truncated to first 20,000 characters.")

    with st.expander("Pipeline notes"):
        st.markdown(
            f"""
            **What this app does**

            1. Reads FASTQ records.
            2. Keeps reads longer than the minimum read length. Current cutoff: `{int(min_read_length)}` nt.
            3. Finds the upstream anchor `{upstream}` and downstream anchor `{downstream}`.
            4. Extracts the insert between the anchors.
            5. Translates DNA codons to amino acids.
            6. Counts unique amino-acid sequences and reports percentages among reads with both anchors.

            **Translation behavior**

            - Codons containing `N` become `X`.
            - Inserts not divisible by 3 are marked `failed` / `out_of_frame`.
            - Stop codons are retained as `*` and flagged as `stop_codon`.
            - Invalid codons are translated as `?` and flagged as `invalid_codon`.
            """
        )


if __name__ == "__main__":
    main()
