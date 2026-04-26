import gradio as gr
import logging

from .services import RecognitionService
from .mcp_client import MocogiClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSS = """
.green { color: green !important; font-weight: bold; }
.red { color: red !important; font-weight: bold; }
.yellow { color: orange !important; font-weight: bold; }
"""

def launch_gui():
    service = RecognitionService()

    # State to store cumulative application data
    # format: {"requests": [], "reports": []}
    app_state = gr.State({"requests": [], "reports": []})

    async def get_study_programs():
        try:
            async with MocogiClient() as client:
                programs = await client.list_study_programs()
                choices = []
                for p in programs:
                    po_id = p.get("id")
                    name = p.get("name", po_id)
                    choices.append((name, po_id))
                return gr.update(choices=choices)
        except Exception as e:
            logger.error(f"Error fetching programs: {e}")
            return gr.update(choices=[("Fehler beim Laden", "error")])

    async def analyze_module_ui(text: str):
        try:
            analysis = await service.analyze_module(text)
            name = analysis.name
            ects = str(analysis.ects) if analysis.ects else ""
            keywords = ", ".join(analysis.keywords)
            return f"Analyse abgeschlossen für: {name}", name, ects, keywords
        except Exception as e:
            logger.error(f"Error analyzing module: {e}")
            return f"Fehler bei der Analyse: {e}", "", "", ""

    async def search_similar_modules_ui(
        po_id: str, keywords: str, max_ects: str, external_text: str
    ):
        try:
            results = await service.search_and_compare(
                po_id, keywords, max_ects, external_text
            )
            # results is List[Tuple[Dict, ComparisonReport]]
            # We need to format it for the UI (list of dicts or similar)
            return results
        except Exception as e:
            logger.error(f"Error during search/comparison: {e}")
            return []

    def add_to_application(state, ext_name, internal_module, report):
        m_title = internal_module.get("metadata", {}).get("title", "Unbekanntes Modul")
        req = f"- {ext_name} soll als {m_title} anerkannt werden."

        new_state = state.copy()
        new_state["requests"].append(req)
        new_state["reports"].append(
            f"Begründung für {ext_name} -> {m_title}:\n{report}\n\n---\n"
        )

        return (
            new_state,
            "\n".join(new_state["requests"]),
            "\n".join(new_state["reports"]),
        )

    with gr.Blocks(title="Modul-Anerkennungs-Tool (PAV Assistant)") as demo:
        gr.Markdown("# 🎓 Modul-Anerkennungs-Tool (PAV Assistant)")

        state = app_state  # Use the state defined above

        with gr.Row():
            with gr.Column(scale=2):
                external_desc = gr.Textbox(
                    label="Externe Modulbeschreibung hier reinkopieren", lines=10
                )
                analyze_btn = gr.Button(
                    "Analysiere Modulbeschreibung", variant="primary"
                )

                with gr.Row():
                    ext_name = gr.Textbox(label="Extrahierter Name")
                    ext_ects = gr.Textbox(label="Extrahierte ECTS")
                ext_keywords = gr.Textbox(label="Suchbegriffe (kommagetrennt)")

                po_dropdown = gr.Dropdown(label="Dein Studiengang bei uns", choices=[])
                search_btn = gr.Button("Suche nach ähnlichen Modulen")
                status_msg = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=3):
                gr.Markdown("### Suchergebnisse und Vergleich")
                results_output = gr.State([])

                @gr.render(inputs=[results_output, ext_name])
                def render_results(comps, name_val):
                    if not comps:
                        gr.Markdown("Noch keine Ergebnisse. Bitte Suche starten.")
                        return

                    with gr.Tabs():
                        for i, (module, comp) in enumerate(comps):
                            decision = comp.decision
                            m_meta = module.get("metadata", {})
                            m_title = m_meta.get("title", "Modul")

                            # Emoji indicators for tabs
                            icon = (
                                "✅"
                                if decision == "Ja"
                                else ("❌" if decision == "Nein" else "⚠️")
                            )
                            tab_title = f"{icon} {m_title}"

                            with gr.Tab(tab_title):
                                color_class = (
                                    "green"
                                    if decision == "Ja"
                                    else ("red" if decision == "Nein" else "yellow")
                                )
                                gr.Markdown(
                                    f"**Vorschlag:** {m_title} ({m_meta.get('ects', '?')} ECTS)"
                                )
                                gr.HTML(
                                    f"<b>Entscheidung:</b> <span class='{color_class}'>{decision}</span>"
                                )
                                gr.Markdown(f"**Kurzbegründung:** {comp.reasoning}")
                                gr.Markdown(f"**Vergleichsbericht:**\n{comp.report}")

                                add_btn = gr.Button(f"Antrag für {m_title} vormerken")

                                add_btn.click(
                                    fn=add_to_application,
                                    inputs=[
                                        state,
                                        ext_name,
                                        gr.State(module),
                                        gr.State(comp.report),
                                    ],
                                    outputs=[state, final_list, final_reports],
                                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Zusammenfassung der Anträge")
                with gr.Row():
                    final_list = gr.Textbox(
                        label="Geplante Anerkennungen", lines=5, interactive=False
                    )
                    final_reports = gr.Textbox(
                        label="Zugehörige Begründungen", lines=10, interactive=False
                    )

        # Event handlers
        analyze_btn.click(
            analyze_module_ui,
            inputs=[external_desc],
            outputs=[status_msg, ext_name, ext_ects, ext_keywords],
        )

        search_btn.click(
            lambda: "Suche läuft und Vergleiche werden erstellt...",
            outputs=[status_msg],
        ).then(
            search_similar_modules_ui,
            inputs=[po_dropdown, ext_keywords, ext_ects, external_desc],
            outputs=[results_output],
        ).then(lambda: "Suche abgeschlossen.", outputs=[status_msg])

        demo.load(get_study_programs, outputs=[po_dropdown])

    return demo

if __name__ == "__main__":
    demo = launch_gui()
    demo.launch(css=CSS)
