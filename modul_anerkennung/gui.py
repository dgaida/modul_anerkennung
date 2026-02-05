"""GUI für das Anerkennungstool mithilfe von Gradio."""
import gradio as gr
import asyncio
import os
from .rag_manager import RAGManager
from .llm_interface import LLMInterface
from .similarity_checker import SimilarityChecker

def launch_gui() -> None:
    """Startet die Gradio-Benutzeroberfläche."""

    # Lazy initialization to avoid starting everything at import
    rag = None
    llm = None
    checker = None

    async def get_checker():
        nonlocal rag, llm, checker
        if checker is None:
            rag = RAGManager()
            llm = LLMInterface()
            checker = SimilarityChecker(rag, llm)
        return checker

    async def process_files(own_file, external_file):
        c = await get_checker()

        # Index internal handbook if provided
        if own_file:
            await rag.process_document(own_file.name)

        # Read external module description
        if external_file.name.lower().endswith(".pdf"):
            # If it's a PDF, we could also process it via RAG or extract text
            # For simplicity, if RAGAnything is working, we can just use the path
            # But compare_modules expects text currently.
            # Let's use a simple text extraction for external_file if it's text-based
            # or just use its content if it's already indexed?
            # Actually, let's just support text files for the 'external' part for now,
            # or extract text if it's PDF.
            try:
                from pypdf import PdfReader
                reader = PdfReader(external_file.name)
                external_text = ""
                for page in reader.pages:
                    external_text += page.extract_text()
            except Exception as e:
                return f"Fehler beim Lesen der externen PDF: {e}"
        else:
            with open(external_file.name, "r", encoding="utf-8", errors="ignore") as f:
                external_text = f.read()

        results = await c.compare_modules(external_text)

        if "explanation" in results:
            return results["explanation"]
        elif "explanations" in results:
            return "\n\n---\n\n".join(results["explanations"])
        else:
            return str(results)

    def sync_process(own_file, external_file):
        try:
            # We need a new event loop or use the existing one
            return asyncio.run(process_files(own_file, external_file))
        except Exception as e:
            return f"Fehler: {e}"

    with gr.Blocks() as demo:
        gr.Markdown("# 🎓 Modul-Anerkennungs-Tool (PAV Assistant)")
        gr.Markdown("Lade dein Modulhandbuch hoch und vergleiche es mit einer externen Modulbeschreibung.")

        with gr.Row():
            own_handbook = gr.File(label="Eigenes Modulhandbuch (PDF)")
            external_module = gr.File(label="Externes Modul (PDF/Text)")

        submit_btn = gr.Button("Vergleichen")
        output = gr.Textbox(label="Ergebnis / Begründung", lines=10)

        submit_btn.click(
            fn=sync_process,
            inputs=[own_handbook, external_module],
            outputs=output
        )

    demo.launch()

if __name__ == "__main__":
    launch_gui()
