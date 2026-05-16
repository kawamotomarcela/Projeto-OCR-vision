import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================

st.set_page_config(
    page_title="Azure Vision OCR",
    page_icon="📄",
    layout="wide"
)

# ======================================================
# CARREGAR .ENV
# ======================================================

load_dotenv()

endpoint = os.getenv("VISION_ENDPOINT")
key = os.getenv("VISION_KEY")

# ======================================================
# DEBUG
# ======================================================

st.sidebar.subheader("🔎 Debug")

st.sidebar.write("Endpoint carregado:")
st.sidebar.code(endpoint)

# ======================================================
# VALIDAÇÃO
# ======================================================

if not endpoint:
    st.error("❌ Endpoint não encontrado no .env")
    st.stop()

if not key:
    st.error("❌ Chave não encontrada no .env")
    st.stop()

# ======================================================
# CLIENT AZURE
# ======================================================

try:

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

except Exception as e:

    st.error(f"❌ Erro ao criar client Azure: {e}")
    st.stop()

# ======================================================
# TÍTULO
# ======================================================

st.title("📄 Azure AI Vision OCR")
st.markdown("### Extração Inteligente de Texto com Azure AI Vision")

st.divider()

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.header("⚙️ Tecnologias")

    st.success("Azure AI Vision conectado")

    st.markdown("""
    - Azure AI Vision
    - OCR READ
    - Python
    - Streamlit
    - Upload de Imagens
    - Dashboard
    """)

# ======================================================
# UPLOAD
# ======================================================

arquivo = st.file_uploader(
    "📤 Carregue uma imagem",
    type=["png", "jpg", "jpeg"]
)

# ======================================================
# PROCESSAMENTO
# ======================================================

if arquivo is not None:

    col1, col2 = st.columns([1, 1])

    # ==================================================
    # IMAGEM
    # ==================================================

    with col1:

        st.subheader("🖼️ Imagem")

        imagem = Image.open(arquivo)

        st.image(
            imagem,
            caption="Imagem carregada",
            use_column_width=True
        )

        st.divider()

        st.info(f"""
📁 Nome: {arquivo.name}

📦 Tipo: {arquivo.type}

📏 Tamanho: {round(arquivo.size / 1024, 2)} KB
        """)

    # ==================================================
    # OCR
    # ==================================================

    with col2:

        st.subheader("📖 OCR Azure")

        if st.button("🔍 Analisar Imagem"):

            with st.spinner("Analisando imagem com Azure AI Vision..."):

                try:

                    # ======================================
                    # RESETAR STREAM
                    # ======================================

                    arquivo.seek(0)

                    image_data = arquivo.read()

                    # ======================================
                    # OCR AZURE
                    # ======================================

                    result = client.analyze(
                        image_data=image_data,
                        visual_features=[VisualFeatures.READ]
                    )

                    texto_extraido = ""

                    # ======================================
                    # EXTRAÇÃO DE TEXTO
                    # ======================================

                    if result.read is not None:

                        for block in result.read.blocks:

                            for line in block.lines:

                                texto_extraido += line.text + "\n"

                    # ======================================
                    # RESULTADO
                    # ======================================

                    if texto_extraido.strip() != "":

                        st.success("✅ OCR realizado com sucesso!")

                        st.text_area(
                            "Texto Detectado",
                            texto_extraido,
                            height=350
                        )

                        st.divider()

                        # ==================================
                        # MÉTRICAS
                        # ==================================

                        colA, colB, colC = st.columns(3)

                        with colA:

                            st.metric(
                                "Linhas",
                                len(texto_extraido.splitlines())
                            )

                        with colB:

                            st.metric(
                                "Palavras",
                                len(texto_extraido.split())
                            )

                        with colC:

                            st.metric(
                                "Caracteres",
                                len(texto_extraido)
                            )

                        st.divider()

                        # ==================================
                        # DOWNLOAD TXT
                        # ==================================

                        st.download_button(
                            label="💾 Baixar TXT",
                            data=texto_extraido,
                            file_name="ocr_resultado.txt",
                            mime="text/plain"
                        )

                    else:

                        st.warning("⚠️ Nenhum texto encontrado.")

                except Exception as e:

                    st.error("❌ Erro ao analisar imagem")

                    st.code(str(e))