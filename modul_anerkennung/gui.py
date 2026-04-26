import gradio as gr
import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple
from .llm_interface import LLMInterface
from .mcp_client import MocogiClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def launch_gui():
    llm = LLMInterface()

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

    async def analyze_module(text: str):
        if not text:
            return "Bitte Modulbeschreibung eingeben.", "", "", ""

        prompt = f"""Analysiere die folgende Modulbeschreibung und extrahiere:
1. Modulname
2. Anzahl ECTS (nur die Zahl)
3. 3-4 prägnante Suchbegriffe für eine semantische Suche.

Antworte ausschließlich im JSON-Format:
{{
  "name": "...",
  "ects": 5,
  "keywords": ["...", "...", "..."]
}}

Modulbeschreibung:
{text}"""

        try:
            response = await llm.achat([{"role": "user", "content": prompt}])
            # Basic JSON extraction
            start = response.find('{')
            end = response.rfind('}') + 1
            data = json.loads(response[start:end])

            name = data.get("name", "")
            ects = str(data.get("ects", ""))
            keywords = ", ".join(data.get("keywords", []))

            return f"Analyse abgeschlossen für: {name}", name, ects, keywords
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return f"Fehler bei der Analyse: {e}", "", "", ""

    async def search_similar_modules(po_id: str, keywords: str, max_ects: str, external_text: str):
        if not po_id:
            return []

        try:
            ects_val = float(max_ects) if max_ects else None
        except ValueError:
            ects_val = None

        try:
            async with MocogiClient() as client:
                modules = await client.call_tool("search_modules", {
                    "po_id": po_id,
                    "search_term": keywords,
                    "max_ects": ects_val
                })

            # For each module, perform a comparison
            comparisons = []
            for m in modules[:5]: # Top 5
                comp = await perform_comparison(external_text, m)
                comparisons.append((m, comp))

            return comparisons
        except Exception as e:
            logger.error(f"Error during search/comparison: {e}")
            return []

    async def perform_comparison(external_text: str, internal_module: Dict[str, Any]):
        internal_text = json.dumps(internal_module, indent=2)

        prompt = f"""Vergleiche die folgende externe Modulbeschreibung mit unserem internen Modul.

Externe Beschreibung:
{external_text}

Internes Modul:
{internal_text}

Erstelle einen detaillierten Vergleichsbericht.
Bestimme, ob das Modul anerkannt werden kann (Ja, Nein, Vielleicht).
Antworte im JSON-Format:
{{
  "decision": "Ja" | "Nein" | "Vielleicht",
  "reasoning": "Kurze Begründung",
  "report": "Ausführlicher Bericht"
}}
"""
        try:
            response = await llm.achat([{"role": "user", "content": prompt}])
            start = response.find('{')
            end = response.rfind('}') + 1
            return json.loads(response[start:end])
        except Exception as e:
            return {"decision": "Vielleicht", "reasoning": f"Fehler: {e}", "report": response}

    def add_to_application(state, ext_name, internal_module, report):
        m_title = internal_module.get('metadata', {}).get('title', 'Unbekanntes Modul')
        req = f"- {ext_name} soll als {m_title} anerkannt werden."

        new_state = state.copy()
        new_state["requests"].append(req)
        new_state["reports"].append(f"Begründung für {ext_name} -> {m_title}:\n{report}\n\n---\n")

        return new_state, "\n".join(new_state["requests"]), "\n".join(new_state["reports"])

    css = """
    .green { color: green !important; font-weight: bold; }
    .red { color: red !important; font-weight: bold; }
    .yellow { color: orange !important; font-weight: bold; }
    """
    with gr.Blocks(title="Modul-Anerkennungs-Tool (PAV Assistant)") as demo:
        gr.Markdown("# 🎓 Modul-Anerkennungs-Tool (PAV Assistant)")

        state = app_state # Use the state defined above

        with gr.Row():
            with gr.Column(scale=2):
                external_desc = gr.Textbox(label="Externe Modulbeschreibung hier reinkopieren", lines=10)
                analyze_btn = gr.Button("Analysiere Modulbeschreibung", variant="primary")

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

                    with gr.Tabs() as tabs:
                        for i, (module, comp) in enumerate(comps):
                            decision = comp.get("decision", "Vielleicht")
                            m_meta = module.get("metadata", {})
                            m_title = m_meta.get("title", "Modul")

                            # Emoji indicators for tabs
                            icon = "✅" if decision == "Ja" else ("❌" if decision == "Nein" else "⚠️")
                            tab_title = f"{icon} {m_title}"

                            with gr.Tab(tab_title):
                                color_class = "green" if decision == "Ja" else ("red" if decision == "Nein" else "yellow")
                                gr.Markdown(f"**Vorschlag:** {m_title} ({m_meta.get('ects', '?')} ECTS)")
                                gr.HTML(f"<b>Entscheidung:</b> <span class='{color_class}'>{decision}</span>")
                                gr.Markdown(f"**Kurzbegründung:** {comp.get('reasoning', '')}")
                                gr.Markdown(f"**Vergleichsbericht:**\n{comp.get('report', '')}")

                                add_btn = gr.Button(f"Antrag für {m_title} vormerken")

                                add_btn.click(
                                    fn=add_to_application,
                                    inputs=[state, ext_name, gr.State(module), gr.State(comp.get('report', ''))],
                                    outputs=[state, final_list, final_reports]
                                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Zusammenfassung der Anträge")
                with gr.Row():
                    final_list = gr.Textbox(label="Geplante Anerkennungen", lines=5, interactive=False)
                    final_reports = gr.Textbox(label="Zugehörige Begründungen", lines=10, interactive=False)

        # Event handlers
        analyze_btn.click(
            analyze_module,
            inputs=[external_desc],
            outputs=[status_msg, ext_name, ext_ects, ext_keywords]
        )

        search_btn.click(
            lambda: "Suche läuft und Vergleiche werden erstellt...",
            outputs=[status_msg]
        ).then(
            search_similar_modules,
            inputs=[po_dropdown, ext_keywords, ext_ects, external_desc],
            outputs=[results_output]
        ).then(
            lambda: "Suche abgeschlossen.",
            outputs=[status_msg]
        )

        demo.load(get_study_programs, outputs=[po_dropdown])

    return demo

if __name__ == "__main__":
    demo = launch_gui()
    demo.launch(css=css)
