from langchain_community.document_loaders import PyPDFLoader


def load_pdf_to_text(pdf_path: str, text_path:str) -> str:
    """Load PDF and return its text content."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in docs])

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    # return full_text


if __name__ == "__main__":
    input_path = '/Users/m2sm/Desktop/projects/Agentic-AI/Judgement_Bot/Data/odisha_judgement_files/displayjudgement25_pdf.pdf'
    output_path = '/Users/m2sm/Desktop/projects/Agentic-AI/Judgement_Bot/Data/Text_files/displayjudgement25_pdf.txt'
    # load_pdf_to_text(input_path, output_path)
    import os
    for files in os.listdir('/Users/m2sm/Desktop/projects/Agentic-AI/Judgement_Bot/Data/odisha_judgement_files/'):
        if files.endswith('.pdf'):
            pdf_path = os.path.join('/Users/m2sm/Desktop/projects/Agentic-AI/Judgement_Bot/Data/odisha_judgement_files/', files)
            text_path = pdf_path.replace('.pdf', '.txt')
            load_pdf_to_text(pdf_path, text_path)