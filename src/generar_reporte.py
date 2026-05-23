"""
generar_reporte.py
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
from pathlib import Path

# --- Configuración de Rutas de Producción ---
DIRECTORIO_RAIZ = Path(__file__).parent.parent
DIRECTORIO_FIGURAS = DIRECTORIO_RAIZ / "data" / "figures"
DIRECTORIO_PROCESADOS = DIRECTORIO_RAIZ / "data" / "processed"
ARCHIVO_SALIDA = DIRECTORIO_RAIZ / "Reporte_Final_CNN_Splicing.docx"

class GeneradorDocumento:
    """
    Clase responsable de la creación del reporte técnico en formato DOCX.
    Sigue estándares de ingeniería y diseño profesional.
    """

    def __init__(self):
        self.objeto_doc = Document()
        self._configurar_estilos_globales()

    def _configurar_estilos_globales(self):
        estilo_normal = self.objeto_doc.styles['Normal']
        estilo_normal.font.name = 'Calibri'
        estilo_normal.font.size = Pt(11)
        estilo_normal.paragraph_format.space_after = Pt(6)
        estilo_normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

        for i in range(1, 4):
            estilo_h = self.objeto_doc.styles[f'Heading {i}']
            estilo_h.font.name = 'Calibri'
            estilo_h.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
            estilo_h.font.bold = True

    def agregar_titulo(self, texto_titulo: str, nivel_jerarquia: int = 1):
        self.objeto_doc.add_heading(texto_titulo, level=nivel_jerarquia)

    def agregar_parrafo(self, texto_parrafo: str):
        self.objeto_doc.add_paragraph(texto_parrafo)

    def agregar_imagen(self, ruta_imagen: Path, pie_foto: str = ""):
        if ruta_imagen.exists():
            self.objeto_doc.add_picture(str(ruta_imagen), width=Inches(5.5))
            if pie_foto:
                parrafo_pie = self.objeto_doc.add_paragraph()
                parrafo_pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
                ejecucion_texto = parrafo_pie.add_run(pie_foto)
                ejecucion_texto.font.size = Pt(9)
                ejecucion_texto.font.italic = True
        else:
            self.agregar_parrafo(f"[Imagen no encontrada: {ruta_imagen.name}]")

    def agregar_tabla(self, encabezados_lista: list, filas_datos: list):
        tabla_nueva = self.objeto_doc.add_table(rows=len(filas_datos) + 1, cols=len(encabezados_lista))
        tabla_nueva.style = 'Table Grid'
        tabla_nueva.alignment = WD_TABLE_ALIGNMENT.CENTER
        celdas_cabecera = tabla_nueva.rows[0].cells
        for i, nombre_columna in enumerate(encabezados_lista):
            celdas_cabecera[i].text = nombre_columna
            sombreado = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F4E79" w:val="clear"/>')
            celdas_cabecera[i]._tc.get_or_add_tcPr().append(sombreado)
        for indice_fila, datos_fila in enumerate(filas_datos):
            celdas_datos = tabla_nueva.rows[indice_fila + 1].cells
            for indice_col, valor_celda in enumerate(datos_fila):
                celdas_datos[indice_col].text = str(valor_celda)

    def generar_portada(self):
        self.objeto_doc.add_paragraph("\n" * 3)
        p_titulo = self.objeto_doc.add_paragraph()
        p_titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_t = p_titulo.add_run("REPORTE FINAL DE IMPLEMENTACIÓN")
        run_t.font.size = Pt(26); run_t.font.bold = True; run_t.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        p_sub = self.objeto_doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_s = p_sub.add_run("Identificación de Sitios de Splicing mediante\nDeep Learning y Evolución Diferencial")
        run_s.font.size = Pt(16)
        self.objeto_doc.add_paragraph("\n" * 5)
        p_autores = self.objeto_doc.add_paragraph()
        p_autores.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_autores.add_run("César Alexander Martínez Pérez\nGuillermo Daniel Zaragoza Castro\nMayo 2026")
        self.objeto_doc.add_page_break()

    def escribir_introduccion(self):
        self.agregar_titulo("1. Introducción")
        self.agregar_parrafo("El splicing es un proceso fundamental en eucariotas...")
        self.agregar_parrafo("La predicción automatizada mediante CNNs permite...")

    def escribir_metodologia(self):
        self.agregar_titulo("2. Metodología")
        self.agregar_parrafo("Se implementó un pipeline modular con las siguientes etapas:")
        self.agregar_tabla(["Módulo", "Descripción"], [
            ["Preprocesamiento", "Limpieza y extracción con pyfaidx."],
            ["Optimización", "Algoritmo Evolutivo Diferencial."],
            ["Entrenamiento", "CNN 1D en PyTorch."]
        ])

    def escribir_resultados(self):
        self.agregar_titulo("3. Resultados")
        self.agregar_parrafo("A continuación se presentan las visualizaciones del análisis específico (Puntos 1-6):")
        
        figuras = [
            ("spec_01_balance_clases.png", "Punto 1: Balance de clases."),
            ("spec_02_dist_cromosomas.png", "Punto 2: Distribución por cromosoma."),
            ("spec_03_composicion_global.png", "Punto 3: Composición nucleotídica global."),
            ("spec_04_gc_content_violin.png", "Punto 4: Contenido GC por tipo de sitio."),
            ("spec_05_correlacion_heatmap.png", "Punto 5: Mapa de calor de correlación."),
            ("spec_06_longitud_exones.png", "Punto 6: Perfil de longitud de exones.")
        ]
        
        for nombre, descripcion in figuras:
            self.agregar_titulo(descripcion, nivel_jerarquia=2)
            self.agregar_imagen(DIRECTORIO_FIGURAS / nombre, descripcion)

    def escribir_conclusiones(self):
        self.agregar_titulo("4. Conclusiones e Interpretación")
        self.agregar_parrafo("La alta precisión obtenida sugiere que el modelo captura eficazmente...")

    def guardar_reporte(self):
        self.generar_portada()
        self.escribir_introduccion()
        self.escribir_metodologia()
        self.escribir_resultados()
        self.escribir_conclusiones()
        self.objeto_doc.save(str(ARCHIVO_SALIDA))
        print(f"✓ Reporte actualizado: {ARCHIVO_SALIDA.name}")

if __name__ == "__main__":
    GeneradorDocumento().guardar_reporte()
