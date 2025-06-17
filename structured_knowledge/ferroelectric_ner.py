import arxiv
import fitz  # PyMuPDF
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import sqlite3
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time

# Define the directory containing the database (same as script directory)
DB_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(DB_DIR, "ferroelectric_knowledge.db")

# Initialize logging
logging.basicConfig(filename='ferroelectric_ner.log', level=logging.ERROR)

# Initialize Streamlit app
st.set_page_config(page_title="Ferroelectric Materials Research Tool", layout="wide")
st.title("Ferroelectric Materials Research Tool")
st.markdown("""
This tool supports research on ferroelectric materials, focusing on parameters like gradient energy coefficients, spontaneous polarization, and Curie temperature. Use the **arXiv Query** tab to search for relevant papers, download PDFs, extract parameters, and save to a database. Use the **NER Analysis** tab to analyze parameters stored in the database (`ferroelectric_knowledge.db`). Results can be visualized and exported as CSV or JSON.

**Note**: The spaCy model may miss complex scientific terms (e.g., tensor notations like G_{ijkl}). For better accuracy, consider training a custom spaCy model: [spaCy Training Guide](https://spacy.io/usage/training).
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `arxiv`, `pymupdf`, `spacy`, `pandas`, `streamlit`, `matplotlib`, `numpy`, `pyarrow` (for Parquet)
- Install with: `pip install arxiv pymupdf spacy pandas streamlit matplotlib numpy pyarrow`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
        st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
    except Exception as e2:
        st.error(f"Failed to load spaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
        st.stop()

# Custom NER patterns for ferroelectricity
matcher = Matcher(nlp.vocab)
patterns = [
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["j m³/c²", "j·m³/c²", "j m^3/c^2"]}}],  # GRADIENT_ENERGY_COEFFICIENT
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["c/m²", "c/m2"]}}],  # SPONTANEOUS_POLARIZATION
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["°c", "celsius", "k", "kelvin"]}}, {"LOWER": {"IN": ["curie", "temperature"]}, "OP": "?"}],  # CURIE_TEMPERATURE
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["v/m", "v/cm"]}}, {"LOWER": {"IN": ["coercive", "field"]}, "OP": "?"}],  # COERCIVE_FIELD
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["nm"]}}, {"LOWER": {"IN": ["domain", "wall", "width"]}, "OP": "?"}],  # DOMAIN_WALL_WIDTH
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["j/m²", "j/m2", "mj/m²", "mj/m2"]}}, {"LOWER": {"IN": ["domain", "wall", "energy"]}, "OP": "?"}],  # DOMAIN_WALL_ENERGY
    [{"TEXT": {"REGEX": r"^(BaTiO3|PbTiO3|LiNbO3|KNbO3|PZT|PVDF|TGS)$"}}, {"LOWER": {"IN": ["material", "ferroelectric"]}, "OP": "?"}],  # FERROELECTRIC_MATERIAL
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["f/m", "farad/m"]}}, {"LOWER": {"IN": ["dielectric", "permittivity"]}, "OP": "?"}],  # DIELECTRIC_PERMITTIVITY
]

param_types = [
    "GRADIENT_ENERGY_COEFFICIENT",
    "SPONTANEOUS_POLARIZATION",
    "CURIE_TEMPERATURE",
    "COERCIVE_FIELD",
    "DOMAIN_WALL_WIDTH",
    "DOMAIN_WALL_ENERGY",
    "FERROELECTRIC_MATERIAL",
    "DIELECTRIC_PERMITTIVITY"
]
for i, pattern in enumerate(patterns):
    matcher.add(f"FERROELECTRIC_PARAM_{param_types[i]}", [pattern])

# Parameter validation ranges
valid_ranges = {
    "GRADIENT_ENERGY_COEFFICIENT": (1e-11, 1e-9, "J m³/C²"),
    "SPONTANEOUS_POLARIZATION": (0.01, 1.0, "C/m²"),
    "CURIE_TEMPERATURE": (0, 1000, "°C"),  # Also accept Kelvin
    "COERCIVE_FIELD": (1e3, 1e8, "V/m"),  # V/m or V/cm
    "DOMAIN_WALL_WIDTH": (0.1, 10, "nm"),
    "DOMAIN_WALL_ENERGY": (1e-3, 100, "mJ/m²"),
    "FERROELECTRIC_MATERIAL": (None, None, ""),  # No numerical range for materials
    "DIELECTRIC_PERMITTIVITY": (1, 10000, "F/m")
}

# Color map for parameter histograms
param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Perform NER
def extract_ferroelectric_parameters(text):
    try:
        doc = nlp(text)
        entities = []
        
        # Capture general material entities
        for ent in doc.ents:
            if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or any(mat in ent.text.lower() for mat in ["batio3", "pzt", "linbo3", "kno3", "pvdf", "tgs"]):
                entities.append({
                    "text": ent.text,
                    "label": "FERROELECTRIC_MATERIAL",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "value": None,
                    "unit": None,
                    "outcome": None
                })
        
        # Custom matcher for ferroelectric parameters
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id].replace("FERROELECTRIC_PARAM_", "")
            match_text = span.text
            value_match = re.match(r"(\d+\.?\d*([eE][+-]?\d+)?)", match_text)
            value = float(value_match.group(1)) if value_match else None
            unit = match_text[value_match.end():].strip() if value_match else None
            
            # Unit conversions
            if unit and value is not None:
                if label == "GRADIENT_ENERGY_COEFFICIENT":
                    if unit.lower() in ["j·m^3/c^2", "j m^3/c^2"]:
                        unit = "J m³/C²"
                elif label == "SPONTANEOUS_POLARIZATION":
                    if unit.lower() == "c/m2":
                        unit = "C/m²"
                elif label == "CURIE_TEMPERATURE":
                    if unit.lower() in ["k", "kelvin"]:
                        value -= 273.15  # Convert Kelvin to Celsius
                        unit = "°C"
                elif label == "COERCIVE_FIELD":
                    if unit.lower() == "v/cm":
                        value *= 100  # Convert V/cm to V/m
                        unit = "V/m"
                elif label == "DOMAIN_WALL_ENERGY":
                    if unit.lower() in ["j/m2", "j/m²"]:
                        value *= 1000  # Convert J/m² to mJ/m²
                        unit = "mJ/m²"
            
            # Validate numerical parameters
            if label in valid_ranges and value is not None:
                min_val, max_val, expected_unit = valid_ranges[label]
                if not (min_val <= value <= max_val and (unit == expected_unit or unit is None or (label == "FERROELECTRIC_MATERIAL" and unit == ""))):
                    continue
            
            # Extract outcome (contextual properties)
            outcome = None
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context_text = text[context_start:context_end].lower()
            outcome_terms = ["domain wall energy", "polarization switching", "hysteresis loop", "piezoelectric coefficient", "dielectric constant"]
            for term in outcome_terms:
                if term in context_text:
                    outcome = term
                    break
            
            entities.append({
                "text": span.text,
                "label": label,
                "start": start,
                "end": end,
                "value": value,
                "unit": unit,
                "outcome": outcome
            })
        
        return entities
    except Exception as e:
        logging.error(f"NER failed: {str(e)}")
        return [{"text": f"Error: {str(e)}", "label": "ERROR", "start": 0, "end": 0, "value": None, "unit": None, "outcome": None}]

# Save to SQLite database (papers and parameters)
def save_to_sqlite(papers_df, params_list, db_file):
    try:
        conn = sqlite3.connect(db_file)
        # Save papers metadata
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        # Save parameters
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        return f"Saved metadata and parameters to {db_file}"
    except Exception as e:
        logging.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Save to Parquet
def save_to_parquet(df, parquet_file):
    try:
        df.to_parquet(parquet_file, index=False)
        return f"Saved metadata to {parquet_file}"
    except Exception as e:
        logging.error(f"Parquet save failed: {str(e)}")
        return f"Failed to save to Parquet: {str(e)}"

# Tabs for arXiv Query and NER Analysis
tab1, tab2 = st.tabs(["arXiv Query", "NER Analysis"])

# --- arXiv Query Tab ---
with tab1:
    st.header("arXiv Query for Ferroelectric Materials Papers")
    st.markdown("Search arXiv, download PDFs, extract ferroelectric parameters, and save to `ferroelectric_knowledge.db` for NER analysis.")

    # Query arXiv function
    def query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases=[]):
        try:
            query_terms = query.strip().split()
            formatted_terms = []
            synonyms = {
                "ferroelectric": ["ferroelectricity", "polarization"],
                "gradient energy coefficient": ["ginzburg-landau coefficient", "gradient term"],
                "batio3": ["barium titanate"],
                "pzt": ["lead zirconate titanate"]
            }
            for term in query_terms:
                if term.startswith('"') and term.endswith('"'):
                    formatted_terms.append(term.strip('"').replace(" ", "+"))
                else:
                    formatted_terms.append(term)
                    for key, syn_list in synonyms.items():
                        if term.lower() == key:
                            formatted_terms.extend(syn_list)
            api_query = " ".join(formatted_terms)
            for phrase in exact_phrases:
                api_query += f' "{phrase.replace(" ", "+")}"'
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=api_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            papers = []
            for result in client.results(search):
                if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                    abstract = result.summary.lower()
                    title = result.title.lower()
                    query_words = set(word.lower() for word in re.split(r'\s+|\".*?\"', query) if word and not word.startswith('"'))
                    for key, syn_list in synonyms.items():
                        if key in query_words:
                            query_words.update(syn_list)
                    matched_terms = [word for word in query_words if word in abstract or word in title]
                    match_score = len(matched_terms) / max(1, len(query_words))
                    
                    abstract_highlighted = abstract
                    for term in matched_terms:
                        abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                    
                    papers.append({
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "year": result.published.year,
                        "categories": ", ".join(result.categories),
                        "abstract": abstract,
                        "abstract_highlighted": abstract_highlighted,
                        "pdf_url": result.pdf_url,
                        "download_status": "Not downloaded",
                        "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                        "match_score": round(match_score * 100),
                        "pdf_path": None
                    })
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            logging.error(f"arXiv query failed: {str(e)}")
            st.error(f"Error querying arXiv: {str(e)}")
            return []

    # Download PDF and extract parameters
    def download_pdf_and_extract(pdf_url, paper_id):
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        params = []
        try:
            urllib.request.urlretrieve(pdf_url, pdf_path)
            file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
            text = extract_text_from_pdf(pdf_path)
            if not text.startswith("Error"):
                entities = extract_ferroelectric_parameters(text)
                for entity in entities:
                    start = max(0, entity["start"] - 50)
                    end = min(len(text), entity["end"] + 50)
                    context = text[start:end].replace("\n", " ")
                    params.append({
                        "paper_id": paper_id,
                        "entity_text": entity["text"],
                        "entity_label": entity["label"],
                        "value": entity["value"],
                        "unit": entity["unit"],
                        "outcome": entity["outcome"],
                        "context": context
                    })
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, params
        except Exception as e:
            logging.error(f"PDF download failed for {paper_id}: {str(e)}")
            params.append({
                "paper_id": paper_id,
                "entity_text": f"Failed: {str(e)}",
                "entity_label": "ERROR",
                "value": None,
                "unit": None,
                "outcome": None,
                "context": ""
            })
            return f"Failed: {str(e)}", None, params

    # Sidebar for search inputs
    with st.sidebar:
        st.subheader("arXiv Search Parameters")
        st.markdown("Customize your search. Use exact phrases (e.g., `\"gradient energy coefficient\"`) for precision.")
        
        query_option = st.radio(
            "Select Query Type",
            ["Default Query", "Custom Query", "Suggested Queries"],
            help="Choose how to specify the search query."
        )
        exact_phrases = []
        if query_option == "Default Query":
            query = "ferroelectric gradient energy coefficient polarization BaTiO3 PZT"
            st.write("Using default query: **" + query + "**")
        elif query_option == "Custom Query":
            query = st.text_input("Enter Custom Query", value="ferroelectric BaTiO3 polarization domain wall")
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"gradient energy coefficient\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Custom query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        else:
            suggested_queries = [
                "ferroelectric gradient energy coefficient BaTiO3",
                "PZT spontaneous polarization domain wall",
                "ferroelectric thin film Curie temperature",
                "multiferroic gradient energy coefficient"
            ]
            query = st.selectbox("Choose Suggested Query", suggested_queries)
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"ferroelectric polarization\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Selected query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        
        default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.mes-hall"]
        extra_categories = ["physics.optics", "cond-mat.other"]
        categories = st.multiselect(
            "Select arXiv Categories",
            default_categories + extra_categories,
            default=default_categories,
            help="Filter papers by categories."
        )
        
        max_results = st.slider(
            "Maximum Number of Papers",
            min_value=1,
            max_value=500,
            value=10,
            help="Set the maximum number of papers to retrieve."
        )
        
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=1900,
                max_value=current_year,
                value=2000,
                help="Earliest publication year."
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=start_year,
                max_value=current_year,
                value=current_year,
                help="Latest publication year."
            )
        
        output_formats = st.multiselect(
            "Select Output Formats",
            ["CSV", "SQLite (.db)", "Parquet (.parquet)", "JSON"],
            default=["CSV", "SQLite (.db)"],
            help="Choose formats for saving metadata."
        )
        
        search_button = st.button("Search arXiv")

    if search_button:
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not categories:
            st.error("Please select at least one category.")
        elif start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        else:
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases)
            
            if not papers:
                st.warning("No papers found matching your criteria.")
                st.markdown("""
                **Suggestions to find more papers:**
                - Broaden the query (e.g., 'ferroelectric polarization').
                - Use fewer terms (e.g., 'ferroelectric BaTiO3').
                - Add exact phrases (e.g., "gradient energy coefficient").
                - Include more categories (e.g., 'cond-mat.other').
                - Expand the year range (e.g., 1990–2025).
                - Increase the maximum number of papers.
                """)
            else:
                st.success(f"Found **{len(papers)}** papers matching your query!")
                exact_display = ', '.join(f'"{p}"' for p in exact_phrases) if exact_phrases else 'None'
                st.write(f"Query: **{query}** | Exact Phrases: **{exact_display}**")
                st.write(f"Categories: **{', '.join(categories)}** | Years: **{start_year}–{end_year}**")
                
                st.subheader("Downloading PDFs and Extracting Parameters")
                progress_bar = st.progress(0)
                all_params = []
                for i, paper in enumerate(papers):
                    if paper["pdf_url"]:
                        status, pdf_path, params = download_pdf_and_extract(paper["pdf_url"], paper["id"])
                        paper["download_status"] = status
                        paper["pdf_path"] = pdf_path
                        all_params.extend(params)
                    else:
                        paper["download_status"] = "No PDF URL"
                        paper["pdf_path"] = None
                        all_params.append({
                            "paper_id": paper["id"],
                            "entity_text": "No PDF URL",
                            "entity_label": "ERROR",
                            "value": None,
                            "unit": None,
                            "outcome": None,
                            "context": ""
                        })
                    progress_bar.progress((i + 1) / len(papers))
                    time.sleep(0.1)
                
                df = pd.DataFrame(papers)
                st.subheader("Paper Details")
                st.dataframe(
                    df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "match_score", "download_status", "pdf_path"]],
                    use_container_width=True,
                    column_config={
                        "abstract_highlighted": st.column_config.TextColumn("Abstract (Highlighted)", help="Matched terms in bold orange.")
                    }
                )
                
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    csv_path = "ferroelectric_papers_metadata.csv"
                    with open(csv_path, "w") as f:
                        f.write(csv)
                    st.info(f"Metadata CSV saved as {csv_path}. Automatic download starting...")
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="Download Paper Metadata CSV (Automatic)",
                            data=f,
                            file_name="ferroelectric_papers_metadata.csv",
                            mime="text/csv",
                            key=f"auto_download_{time.time()}"
                        )
                    st.download_button(
                        label="Download Paper Metadata CSV (Manual)",
                        data=csv,
                        file_name="ferroelectric_papers_metadata.csv",
                        mime="text/csv",
                        key="manual_download"
                    )
                
                if "SQLite (.db)" in output_formats:
                    sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), all_params, DB_FILE)
                    st.info(sqlite_status)
                
                if "Parquet (.parquet)" in output_formats:
                    parquet_status = save_to_parquet(df.drop(columns=["abstract_highlighted"]), "ferroelectric_papers_metadata.parquet")
                    st.info(parquet_status)
                
                if "JSON" in output_formats:
                    json_path = "ferroelectric_papers_metadata.json"
                    df.drop(columns=["abstract_highlighted"]).to_json(json_path, orient="records", lines=True)
                    st.info(f"Saved metadata to {json_path}")
                    with open(json_path, "rb") as f:
                        st.download_button(
                            label="Download Paper Metadata JSON",
                            data=f,
                            file_name="ferroelectric_papers_metadata.json",
                            mime="application/json",
                            key="json_download"
                        )
                
                downloaded = sum(1 for p in papers if "Downloaded" in p["download_status"])
                st.write(f"**Summary**: {len(papers)} papers found, {downloaded} PDFs downloaded successfully, {len([p for p in all_params if p['entity_label'] != 'ERROR'])} parameters extracted.")
                if downloaded < len(papers):
                    st.warning("Some PDFs failed to download. Check 'download_status' for details.")
                common_terms = set()
                for terms in df["matched_terms"]:
                    if terms and terms != "None":
                        common_terms.update(terms.split(", "))
                if common_terms:
                    st.markdown(f"**Query Refinement Tip**: Common matched terms: {', '.join(common_terms)}. Try focusing on these (e.g., '{' '.join(list(common_terms)[:3])}').")

# --- NER Analysis Tab ---
with tab2:
    st.header("NER Analysis for Ferroelectric Parameters")
    st.markdown("Analyze ferroelectric parameters stored in `ferroelectric_knowledge.db`. No PDFs or CSV metadata required.")

    # Validate database
    def validate_db(db_file):
        try:
            conn = sqlite3.connect(db_file)
            # Check papers table
            df_papers = pd.read_sql_query("SELECT * FROM papers LIMIT 1", conn)
            required_columns = ["id", "title", "year", "abstract"]
            missing_columns = [col for col in required_columns if col not in df_papers.columns]
            if missing_columns:
                conn.close()
                return False, f"Database 'papers' table missing required columns: {', '.join(missing_columns)}"
            # Check parameters table
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
            if not cursor.fetchone():
                conn.close()
                return False, "Database missing 'parameters' table."
            df_params = pd.read_sql_query("SELECT * FROM parameters LIMIT 1", conn)
            required_param_columns = ["paper_id", "entity_text", "entity_label"]
            missing_param_columns = [col for col in required_param_columns if col not in df_params.columns]
            if missing_param_columns:
                conn.close()
                return False, f"Database 'parameters' table missing required columns: {', '.join(missing_param_columns)}"
            conn.close()
            return True, "Database format is valid."
        except Exception as e:
            return False, f"Error reading database: {str(e)}"

    # Process parameters from database
    def process_params_from_db(db_file):
        if not os.path.isabs(db_file):
            db_file = os.path.join(DB_DIR, db_file)
        
        if not os.path.exists(db_file):
            st.error(f"Database file {db_file} not found. Run the arXiv Query tab first.")
            return None
        
        is_valid, validation_message = validate_db(db_file)
        if not is_valid:
            st.error(validation_message)
            return None
        st.info(validation_message)
        
        try:
            conn = sqlite3.connect(db_file)
            # Load parameters
            params_df = pd.read_sql_query("SELECT * FROM parameters", conn)
            # Load papers metadata for title and year
            papers_df = pd.read_sql_query("SELECT id, title, year FROM papers", conn)
            conn.close()
            
            if params_df.empty:
                st.warning("No parameters found in the database.")
                return None
            
            # Merge to get title and year
            df = params_df.merge(papers_df, left_on="paper_id", right_on="id", how="left")
            df = df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]]
            
            relevant_papers = len(df["paper_id"].unique())
            st.info(f"Found parameters from {relevant_papers} papers.")
            return df
        except Exception as e:
            st.error(f"Error reading database: {str(e)}")
            return None

    # Sidebar for NER inputs
    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        st.markdown("Configure the analysis to extract ferroelectric parameters from the database.")
        
        db_file_input = st.text_input("SQLite Database File", value=DB_FILE, key="ner_db_file")
        entity_types = st.multiselect(
            "Parameter Types to Display",
            ["GRADIENT_ENERGY_COEFFICIENT", "SPONTANEOUS_POLARIZATION", "CURIE_TEMPERATURE", "COERCIVE_FIELD", "DOMAIN_WALL_WIDTH", "DOMAIN_WALL_ENERGY", "FERROELECTRIC_MATERIAL", "DIELECTRIC_PERMITTIVITY"],
            default=["GRADIENT_ENERGY_COEFFICIENT", "SPONTANEOUS_POLARIZATION", "FERROELECTRIC_MATERIAL"],
            help="Select parameter types to filter results."
        )
        sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
        analyze_button = st.button("Run NER Analysis")

    if analyze_button:
        if not db_file_input:
            st.error("Please specify the SQLite database file.")
        else:
            with st.spinner("Processing parameters from database..."):
                df = process_params_from_db(db_file_input)
            
            if df is None or df.empty:
                st.warning("No parameters extracted. Ensure the database contains parameters from the arXiv Query tab.")
            else:
                st.success(f"Extracted **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                
                if entity_types:
                    df = df[df["entity_label"].isin(entity_types)]
                
                if sort_by == "entity_label":
                    df = df.sort_values(["entity_label", "value"])
                else:
                    df = df.sort_values(["value", "entity_label"], na_position="last")
                
                st.subheader("Extracted Parameters")
                st.dataframe(
                    df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
                    use_container_width=True,
                    column_config={
                        "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
                        "value": st.column_config.NumberColumn("Value", help="Numerical value of the parameter."),
                        "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., hysteresis loop).")
                    }
                )
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Ferroelectric Parameters CSV",
                    csv,
                    "ferroelectric_params.csv",
                    "text/csv"
                )
                
                json_data = df.to_json("ferroelectric_params.json", orient="records", lines=True)
                with open("ferroelectric_params.json", "rb") as f:
                    st.download_button(
                        "Download Ferroelectric Parameters JSON",
                        f,
                        "ferroelectric_params.json",
                        "application/json"
                    )
                
                st.subheader("Parameter Distribution Analysis")
                for param_type in entity_types:
                    if param_type in param_types:
                        param_df = df[df["entity_label"] == param_type]
                        if not param_df.empty:
                            values = param_df["value"].dropna()
                            if not values.empty:
                                fig, ax = plt.subplots()
                                ax.hist(values, bins=10, edgecolor="black", color=param_colors[param_type])
                                unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                                ax.set_xlabel(f"{param_type} ({unit})")
                                ax.set_ylabel("Count")
                                ax.set_title(f"Distribution of {param_type}")
                                st.pyplot(fig)
                
                st.write(f"**Summary**: {len(df)} parameters extracted, including {len(df[df['entity_label'] == 'GRADIENT_ENERGY_COEFFICIENT'])} gradient energy coefficients, {len(df[df['entity_label'] == 'SPONTANEOUS_POLARIZATION'])} spontaneous polarization, {len(df[df['entity_label'] == 'CURIE_TEMPERATURE'])} Curie temperature parameters.")
                st.markdown("""
                **Next Steps**:
                - Filter by parameter types to focus on specific ferroelectric properties.
                - Review outcomes to link parameters to properties like hysteresis or piezoelectricity.
                - Use CSV/JSON for further analysis or computational studies.
                """)

# Footer
st.markdown("---")
st.write("Developed for ferroelectric materials research, focusing on gradient energy coefficients and related properties.")

