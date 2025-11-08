"""GUI für das Anerkennungstool mithilfe von Gradio."""
import gradio as gr
import asyncio
from .rag_manager import RAGManager
from .llm_interface import LLMInterface
from .similarity_checker import SimilarityChecker

def launch_gui() -> None:
    """Startet die Gradio-Benutzeroberfläche."""
    rag = RAGManager()
    llm = LLMInterface()
    checker = SimilarityChecker(rag, llm)

    async def process_files(own_file, external_file):
        await rag.process_document(own_file.name)
        with open(external_file.name, "r", encoding="utf-8") as f:
            external_text = f.read()
        results = await checker.compare_modules(external_text)
        return results

    def sync_process(own_file, external_file):
        return asyncio.run(process_files(own_file, external_file))

    interface = gr.Interface(
        fn=sync_process,
        inputs=[
            gr.File(label="Eigenes Modulhandbuch (PDF)"),
            gr.File(label="Externes Modul (PDF/Text)"),
        ],
        outputs="text",
        title="Modul-Anerkennungs-Tool für PAV",
        description="Prüft Ähnlichkeiten zwischen Modulbeschreibungen.",
    )

    interface.launch()
