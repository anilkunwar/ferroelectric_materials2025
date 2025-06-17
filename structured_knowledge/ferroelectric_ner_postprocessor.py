import pandas as pd
import streamlit as st
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sqlite3
import numpy as np
import logging
import io
import tempfile
import time
from PIL import Image

# Define the directory containing the database (same as script directory)
DB_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(DB_DIR, "ferroelectric_knowledge.db")

# Initialize logging
logging.basicConfig(filename='ferroelectric_ner.log', level=logging.DEBUG)  # Use DEBUG for Cloud diagnosis

# Log debugging information
logging.debug(f"Current working directory: {os.getcwd()}")
logging.debug(f"Script directory: {DB_DIR}")
logging.debug(f"Database file path: {DB_FILE}")
logging.debug(f"Database file exists: {os.path.exists(DB_FILE)}")
if os.path.exists(DB_FILE):
    logging.debug(f"Database file size: {os.path.getsize(DB_FILE)} bytes")

# Initialize Streamlit app
st.set_page_config(page_title="Ferroelectric NER Analysis Tool", layout="wide")
st.title("Ferroelectric Materials NER Analysis Tool")
st.markdown("""
This tool analyzes ferroelectric parameters (e.g., gradient energy coefficients, spontaneous polarization, Curie temperature) stored in a SQLite database (`ferroelectric_knowledge.db` or an uploaded `.db` file). The database should contain metadata and pre-extracted parameters from a prior arXiv query on ferroelectric materials. Results are visualized and exportable as CSV or JSON for further analysis or computational studies.

**Note**: For large datasets in Streamlit Cloud, enable histograms selectively to avoid performance issues.
""")

# Initialize session state for histogram toggle
if 'show_histograms' not in st.session_state:
    st.session_state.show_histograms = False

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `pandas`, `streamlit`, `matplotlib`, `numpy`, `sqlite3`, `io`, `tempfile`, `pillow`
- Install with: `pip install pandas streamlit matplotlib numpy pillow`
""")

# Parameter types for filtering and visualization
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

# Color map for parameter histograms
param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}

# Validate database
def validate_db(db_file):
    try:
        logging.debug(f"Validating database at: {db_file}")
        # Check SQLite header
        with open(db_file, 'rb') as f:
            header = f.read(16).decode('ascii', errors='ignore')
            if not header.startswith('SQLite format 3'):
                logging.error(f"File {db_file} is not a valid SQLite database (invalid header).")
                return False, f"File {db_file} is not a valid SQLite database (invalid header)."
        
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
        required_param_columns = ["paper_id", "entity_text", "entity_label", "value", "unit", "outcome", "context"]
        missing_param_columns = [col for col in required_param_columns if col not in df_params.columns]
        if missing_param_columns:
            conn.close()
            return False, f"Database 'parameters' table missing required columns: {', '.join(missing_param_columns)}"
        conn.close()
        return True, "Database format is valid."
    except sqlite3.DatabaseError as e:
        logging.error(f"Database validation failed: {str(e)}")
        return False, f"Error reading database: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error during database validation: {str(e)}")
        return False, f"Unexpected error reading database: {str(e)}"

# Process parameters from database
def process_params_from_db(db_file):
    # Resolve relative path to absolute path if not already absolute
    if not os.path.isabs(db_file):
        db_file = os.path.join(DB_DIR, db_file)
    
    logging.debug(f"Processing database at: {db_file}")
    if not os.path.exists(db_file):
        st.error(f"Database file {db_file} not found. Ensure it exists and contains metadata and parameters from a prior arXiv query.")
        logging.error(f"Database file not found: {db_file}")
        return None
    
    is_valid, validation_message = validate_db(db_file)
    if not is_valid:
        st.error(validation_message)
        return None
    st.info(validation_message)
    
    try:
        conn = sqlite3.connect(db_file)
        # Optimize queries with specific columns
        params_df = pd.read_sql_query("SELECT paper_id, entity_text, entity_label, value, unit, outcome, context FROM parameters", conn)
        papers_df = pd.read_sql_query("SELECT id, title, year, abstract FROM papers", conn)
        conn.close()
        
        if params_df.empty:
            st.warning("No parameters found in the database.")
            return None
        
        # Filter papers with relevant abstracts
        papers_df['abstract_lower'] = papers_df['abstract'].str.lower()
        relevant_papers = papers_df[papers_df['abstract_lower'].str.contains('ferroelectric|ferroelectricity|polarization|batio3|pzt|domain wall', na=False)]
        
        # Merge parameters with relevant papers
        df = params_df.merge(relevant_papers[['id', 'title', 'year']], left_on="paper_id", right_on="id", how="inner")
        df = df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]]
        
        if df.empty:
            st.warning("No parameters found for papers with relevant content (ferroelectricity, polarization, etc.).")
            return None
        
        relevant_papers_count = len(df["paper_id"].unique())
        st.info(f"Found parameters from {relevant_papers_count} relevant papers.")
        return df
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        logging.error(f"Database processing failed: {str(e)}")
        return None

# Save histogram to temporary file
def save_histogram_to_file(param_type, values, unit, param_colors, bins=10):
    try:
        logging.debug(f"Generating histogram for {param_type}")
        fig, ax = plt.subplots()
        ax.hist(values, bins=bins, edgecolor="black", color=param_colors[param_type])
        ax.set_xlabel(f"{param_type} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {param_type}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            plt.savefig(temp_file.name, format="png", bbox_inches="tight")
            plt.close(fig)
            logging.debug(f"Histogram saved to {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logging.error(f"Failed to generate histogram for {param_type}: {str(e)}")
        plt.close(fig)
        return None

# Save histogram data to CSV (in-memory)
def save_histogram_to_csv(param_type, values, unit, bins=10):
    try:
        counts, bin_edges = np.histogram(values, bins=bins)
        histogram_data = pd.DataFrame({
            'bin_start': bin_edges[:-1],
            'bin_end': bin_edges[1:],
            'count': counts
        })
        histogram_data['parameter_type'] = param_type
        histogram_data['unit'] = unit if unit else "None"
        csv_buffer = io.StringIO()
        histogram_data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), f"histogram_{param_type.lower()}.csv"
    except Exception as e:
        logging.error(f"Failed to save histogram CSV for {param_type}: {str(e)}")
        return None, None

# Sidebar for NER inputs
st.sidebar.header("NER Analysis Parameters")
st.sidebar.markdown("""
Configure the analysis to extract ferroelectric parameters from the SQLite database. **Uploading a `.db` file is recommended** for large datasets in Streamlit Cloud.
""")

# File uploader for .db files
uploaded_db = st.sidebar.file_uploader("Upload SQLite Database (.db)", type=["db", "sqlite", "sqlite3"], key="db_uploader")
db_file_input = st.sidebar.text_input("Or Specify SQLite Database Path", value=DB_FILE, key="ner_db_file")
if not uploaded_db and not os.path.exists(DB_FILE):
    st.sidebar.warning(f"No file uploaded and `ferroelectric_knowledge.db` not found at {DB_FILE}. Upload a valid SQLite database.")
entity_types = st.multiselect(
    "Parameter Types to Display",
    param_types,
    default=param_types,
    help="Select parameter types to filter results."
)
sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
show_histograms = st.sidebar.checkbox("Show Histograms", value=False, key="show_histograms")
if show_histograms:
    st.session_state.show_histograms = True
else:
    st.session_state.show_histograms = False
analyze_button = st.sidebar.button("Run NER Analysis")

# Handle file upload
if uploaded_db:
    try:
        with st.spinner("Processing uploaded file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
                temp_file.write(uploaded_db.read())
                st.session_state.db_path = temp_file.name
            st.info(f"Uploaded database saved temporarily at: {st.session_state.db_path}")
            logging.debug(f"Uploaded database saved at: {st.session_state.db_path}")
    except Exception as e:
        st.error(f"Failed to process uploaded database: {str(e)}")
        logging.error(f"Uploaded database processing failed: {str(e)}")
        st.session_state.db_path = None

if analyze_button:
    try:
        if not uploaded_db and not db_file_input:
            st.error("Please upload a SQLite database file or specify a valid database path.")
        else:
            db_path = st.session_state.db_path if uploaded_db else db_file_input
            logging.debug(f"Using database path for analysis: {db_path}")

            if db_path:
                with st.spinner("Processing parameters from database..."):
                    df = process_params_from_db(db_path)
                
                # Clean up temporary file if uploaded
                if uploaded_db and st.session_state.get('db_path') and os.path.exists(st.session_state.db_path):
                    try:
                        os.unlink(st.session_state.db_path)
                        st.info("Temporary database file cleaned up.")
                        logging.debug(f"Temporary file cleaned up: {st.session_state.db_path}")
                        st.session_state.db_path = None
                    except Exception as e:
                        logging.error(f"Failed to clean up temporary file {st.session_state.db_path}: {str(e)}")

                if df is None or df.empty:
                    st.warning("No parameters extracted. Ensure the database contains parameters and relevant papers from a prior arXiv query.")
                else:
                    st.success(f"Retrieved **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                    
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
                    
                    # Main parameters CSV download (in-memory)
                    try:
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue().encode('utf-8')
                        st.download_button(
                            "Download Ferroelectric Parameters CSV",
                            csv_data,
                            "ferroelectric_params.csv",
                            "text/csv",
                            key=f"main_csv_download_{time.time()}"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate main CSV download: {str(e)}")
                        logging.error(f"Main CSV download failed: {str(e)}")
                    
                    # Main parameters JSON download (in-memory)
                    try:
                        json_buffer = io.StringIO()
                        df.to_json(json_buffer, orient="records", lines=True)
                        json_data = json_buffer.getvalue().encode('utf-8')
                        st.download_button(
                            "Download Ferroelectric Parameters JSON",
                            json_data,
                            "ferroelectric_params.json",
                            "application/json",
                            key=f"main_json_download_{time.time()}"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate main JSON download: {str(e)}")
                        logging.error(f"Main JSON download failed: {str(e)}")
                    
                    if st.session_state.show_histograms:
                        st.subheader("Parameter Distribution Analysis")
                        for param_type in entity_types:
                            if param_type in param_types:
                                param_df = df[df["entity_label"] == param_type]
                                if not param_df.empty:
                                    # Convert values to float, filter invalid entries
                                    try:
                                        values = param_df["value"].dropna().astype(float)
                                        values = values[np.isfinite(values)]
                                        if not values.empty:
                                            unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                                            logging.debug(f"Values for {param_type}: {values.tolist()[:10]}")
                                            plot_path = save_histogram_to_file(param_type, values, unit, param_colors)
                                            if plot_path:
                                                try:
                                                    st.image(Image.open(plot_path), caption=f"Distribution of {param_type}")
                                                    os.unlink(plot_path)
                                                except Exception as e:
                                                    st.error(f"Failed to display histogram for {param_type}: {str(e)}")
                                                    logging.error(f"Histogram display failed for {param_type}: {str(e)}")
                                            
                                            # Histogram CSV download (in-memory)
                                            try:
                                                csv_data, csv_filename = save_histogram_to_csv(param_type, values, unit)
                                                if csv_data:
                                                    st.download_button(
                                                        label=f"Download {param_type} Histogram CSV",
                                                        data=csv_data.encode('utf-8'),
                                                        file_name=csv_filename,
                                                        mime="text/csv",
                                                        key=f"histogram_download_{param_type.lower()}_{time.time()}"
                                                    )
                                            except Exception as e:
                                                st.error(f"Failed to generate histogram CSV for {param_type}: {str(e)}")
                                                logging.error(f"Histogram CSV download failed for {param_type}: {str(e)}")
                                        else:
                                            st.warning(f"No valid numerical values for {param_type} to plot.")
                                    except ValueError as e:
                                        st.error(f"Invalid data for {param_type}: {str(e)}")
                                        logging.error(f"Invalid values for {param_type}: {str(e)}")
                    else:
                        st.info("Histograms are disabled. Enable 'Show Histograms' in the sidebar to view parameter distributions.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}. Please try again or check the logs for details.")
        logging.error(f"Unexpected error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed for ferroelectric materials research, focusing on gradient energy coefficients and related properties.")
