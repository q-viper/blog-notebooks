
# File Selection Drop Down
import streamlit as st
import os
from typing import Dict


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def main():
    fileslist = get_static_store()
    folderPath = st.text_input('Enter folder path:')
    if folderPath:    
        filename = file_selector(folderPath)
        if not filename in fileslist.values():
            fileslist[filename] = filename
    else:
        fileslist.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Select one or more files.")

    if st.button("Clear file list"):
        fileslist.clear()
    if st.checkbox("Show file list?", True):
        finalNames = list(fileslist.keys())
        st.write(list(fileslist.keys()))

main()

