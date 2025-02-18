from openai import OpenAI
import base64
import json
import os
import streamlit as st
from urllib.parse import urlparse
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
from io import StringIO
from pdf2image import convert_from_bytes
import tempfile
import pandas as pd

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "list_of_list_of_dicts_response",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True
                            }
                        }
                    }
                },
                "required": ["data"],
                "additionalProperties": False
            },
            "strict": False
        }
    }

def encode_image(uploaded_file):
    """Encodes an uploaded file into a base64 string."""
    image_bytes = uploaded_file.read()  # Read the file as bytes
    return base64.b64encode(image_bytes).decode("utf-8")

def encode_image2(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def combine_dfs_by_columns(dfs):
    from collections import defaultdict
    grouped_dfs = defaultdict(list)
    result_dfs = []

    # Group DataFrames by their column names
    for df in dfs:
        if list(df.columns) == ["Column_1", "Column_2"]:
            grouped_dfs["Column_1_Column_2"].append(df)
        else:
            result_dfs.append(df)

    # Combine only those DataFrames that have the specified columns
    if grouped_dfs["Column_1_Column_2"]:
        combined_df = pd.concat(grouped_dfs["Column_1_Column_2"], ignore_index=True)
        result_dfs.append(combined_df)

    return result_dfs

def extract_from_image(base64_img):
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""You will be given an image to analyze. Your task is to extract ALL meaningful information, including ALL metadata present in the image, and present it in a structured format as a list of lists of dictionaries.
                                                Final Output Structure:
                                                The final extracted data should be represented as a list of lists of dictionaries.
                                                Each list represents a category of extracted information (e.g., metadata, tables).
                                                Each inner list contains dictionaries where each dictionary represents a single record (row), mapping keys (column names) to their values.
                                                
                                                Metadata Extraction:
                                                Store metadata in a list of dictionaries, where:
                                                Each dictionary represents a metadata field, mapping "Key" to "Value".
                                                
                                                Table Extraction:
                                                If a table is present in the image, extract it as a separate list of dictionaries.
                                                Each table should be a separate list, where each dictionary represents a single row with column names as keys and corresponding values."""},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{base64_img}"}
                    },
                ]
            },
        ],
        response_format = FORMAT,
    )

    output = response1.choices[0].message.content.strip()
    output_dict = json.loads(output)
    data1 = output_dict["data"]
    # data1 = [[{'Exporter/Shipper': 'ABC Trading'}, {'Exporter/Shipper': 'Jane Smith'}, {'Exporter/Shipper': '2593 Hillview Street'}, {'Exporter/Shipper': 'Columbia, SC, 29201'}, {'Exporter/Shipper': '(123) 123-4657'}, {'Exporter/Shipper': 'jane@abctrading123.com'}], [{'Importer/Consignee': 'Personal Home'}, {'Importer/Consignee': 'John Doe'}, {'Importer/Consignee': '3219 Turnpike Drive'}, {'Importer/Consignee': 'Huntsville, AL, 35816'}, {'Importer/Consignee': '(123) 123-4567'}, {'Importer/Consignee': 'johndoe@personalhome.com'}], [{'Invoice Date': 'March 18, 2019'}], [{'Invoice #': '0123456789'}], [{'Country of Origin': 'United States'}], [{'Country Destination': 'Finland'}], [{'Item ID': '001', 'Description': 'Product A', 'Weight': '1', 'Quantity': '10', 'Price ($)': '100', 'Total ($)': '1000'}, {'Item ID': '002', 'Description': 'Product B', 'Weight': '2', 'Quantity': '10', 'Price ($)': '200', 'Total ($)': '2000'}, {'Item ID': '003', 'Description': 'Product C', 'Weight': '3', 'Quantity': '10', 'Price ($)': '300', 'Total ($)': '3000'}, {'Item ID': '004', 'Description': 'Product D', 'Weight': '1', 'Quantity': '10', 'Price ($)': '400', 'Total ($)': '4000'}, {'Item ID': '005', 'Description': 'Product E', 'Weight': '2', 'Quantity': '10', 'Price ($)': '500', 'Total ($)': '5000'}, {'Item ID': '006', 'Description': 'Product F', 'Weight': '3', 'Quantity': '10', 'Price ($)': '100', 'Total ($)': '1000'}, {'Item ID': '007', 'Description': 'Product G', 'Weight': '1', 'Quantity': '10', 'Price ($)': '200', 'Total ($)': '2000'}], [{'Purpose of export': 'Sale'}], [{'Sub total ($)': '18000'}, {'Shipping Fee ($)': '500'}, {'Sales Tax ($)': '500'}, {'Total Amount ($)': '19000'}]]
    print(data1)
    dfs = []
    for index, data in enumerate(data1):
        # Remove empty dictionaries
        data = [row for row in data if row]

        expected_keys = set(data[0].keys())
        same_keys = all(set(d.keys()) == expected_keys for d in data)

        if not same_keys:
            df = pd.DataFrame([(k, v) for d in data for k, v in d.items()], columns=['Key', 'Value'])
            dfs.append(df)
        elif (len(data) == 1):
            df = pd.DataFrame(list(data[0].items()), columns=['Column_1', 'Column_2'])
            dfs.append(df)
        else:
            # Convert to DataFrame normally
            column_names = list(data[0].keys())
            rows = [list(row.values()) for row in data]

            df = pd.DataFrame(rows, columns=column_names)

            # Display DataFrame
            dfs.append(df)

    result_dfs = combine_dfs_by_columns(dfs)
    return result_dfs

def dfs_of_image(base64_img):
    result_dfs = extract_from_image(base64_img)
    return result_dfs

def dfs_of_pdf(pdf_file):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    all_dfs = []
    images = convert_from_path("temp.pdf", dpi=300)
    for i, image in enumerate(images):
        image_path = f"{i+1}.png"
        image.save(image_path, "PNG")
        base64_img = f"data:image/png;base64,{encode_image2(image_path)}"
        result_dfs = extract_from_image(base64_img)
        all_dfs.extend(result_dfs)

    result_dfs = combine_dfs_by_columns(all_dfs)
    return result_dfs


st.title("Extract Everything from PDF/Image")
uploaded_file = st.file_uploader("Upload an image (PNG, JPG) or a PDF file", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    if "next_clicked" not in st.session_state:
        st.session_state.next_clicked = False

    # Make the button span the full width
    col = st.columns(1)  # Single column to stretch across

    with col[0]:
        if st.button("Next", use_container_width=True):  # This makes it full width
            st.session_state.next_clicked = True

# When "Next" is clicked, proceed with image processing
if st.session_state.get("next_clicked", False):
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.png'):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        base64_img = f"data:image/png;base64,{encode_image2(temp_path)}"
        result_dfs = dfs_of_image(base64_img)
    elif file_name.endswith(('.jpg', '.jpeg')):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        base64_img = f"data:image/jpeg;base64,{encode_image2(temp_path)}"
        result_dfs = dfs_of_image(base64_img)
    elif file_name.endswith('.pdf'):
        if uploaded_file is not None:
            binary_data = uploaded_file.getvalue()
            base64_pdf = base64.b64encode(binary_data).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="400" type="application/pdf"></iframe>'
            st.write("Preview of the uploaded PDF:")
            st.markdown(pdf_display, unsafe_allow_html=True)
        result_dfs = dfs_of_pdf(uploaded_file)
    else:
        st.error("Unsupported file format")
        result_dfs = []

    for i, df in enumerate(result_dfs):
        st.write(f"Table {i + 1}")
        st.dataframe(df)
