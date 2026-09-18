from pathlib import Path
import ast, nbformat

def test_project_structure_and_notebook_syntax():
    root=Path(__file__).resolve().parents[1]
    required=['notebook/complete_ecommerce_rag_chatbot.ipynb','app/main.py','app/chatbot.py','README.md','RUN_ONE_NOTEBOOK.md','GITHUB_UPLOAD_GUIDE.md']
    assert all((root/p).exists() for p in required)
    nb=nbformat.read(root/required[0],as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code' and not cell.source.lstrip().startswith(('%','!')):
            ast.parse(cell.source)
