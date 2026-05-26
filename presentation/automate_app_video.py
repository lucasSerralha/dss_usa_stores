import subprocess
import time
import os
import signal
from playwright.sync_api import sync_playwright

# Configurações de Caminhos
BASE_DIR = "/Volumes/Extreme Pro/MEGSI/TIAPOSE/LUCAS/dss_usa_stores"
APP_PATH = os.path.join(BASE_DIR, "dss_app/app.py")
VIDEO_DIR = os.path.join(BASE_DIR, "presentation")
STREAMLIT_URL = "http://localhost:8501"

def clear_cache():
    print("🧹 Limpando cache do Streamlit...")
    subprocess.run(["streamlit", "cache", "clear"], check=True)

def start_app():
    print("🚀 Iniciando aplicação Streamlit...")
    # Iniciamos o processo em background
    process = subprocess.Popen(
        ["streamlit", "run", APP_PATH, "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    # Aguarda o servidor subir
    time.sleep(5)
    return process

def record_demo():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

    with sync_playwright() as p:
        print("🎥 Iniciando gravação com Playwright (Full HD)...")
        browser = p.chromium.launch(headless=True)
        
        # Criamos o contexto com gravação de vídeo habilitada
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            record_video_dir=VIDEO_DIR,
            record_video_size={'width': 1920, 'height': 1080}
        )
        
        page = context.new_page()
        page.goto(STREAMLIT_URL)
        
        # Aguarda o carregamento inicial (Hero section)
        page.wait_for_selector(".hero-title", timeout=15000)
        print("✅ Aplicação carregada. Gravando...")

        # --- Comandos de Navegação ---
        print("🏠 Exibindo Home...")
        page.wait_for_timeout(5000)
        
        print("📊 Navegando para Auditoria de Predições...")
        page.get_by_text("01 Auditoria Predicoes").click()
        page.wait_for_selector("text=Auditoria Científica de Modelos Preditivos", timeout=15000)
        page.wait_for_timeout(6000)

        print("⚖️ Navegando para Otimização Monobjetivo...")
        page.get_by_text("02 Otimizacao Monobjetivo").click()
        page.wait_for_selector("text=Otimizacao Monobjetivo — O1 e O2", timeout=15000)
        page.wait_for_timeout(7000)

        print("🧬 Navegando para Otimização Multiobjetivo...")
        page.get_by_text("03 Otimizacao Multiobjetivo").click()
        page.wait_for_selector("text=O3: Laboratorio de Decisao Multiobjetivo", timeout=15000)
        page.wait_for_timeout(5000)
        
        # Interação com o Slider (Mostra a reatividade do sistema)
        try:
            slider = page.locator('div[data-baseweb="slider"]').first
            if slider.is_visible():
                print("🎚️ Ajustando peso w (Trade-off)...")
                slider.click()
                for _ in range(12): # Move para a esquerda (foco em Staff)
                    page.keyboard.press("ArrowLeft")
                    page.wait_for_timeout(300)
        except Exception:
            print("⚠️ Não foi possível interagir com o slider.")
        
        page.wait_for_timeout(6000)
        print("🏁 Navegação concluída.")
        # -----------------------------

        # Finaliza
        context.close()
        video_path = page.video.path()
        browser.close()
        
        # Renomear o vídeo gerado (o Playwright gera nomes aleatórios)
        final_video_name = os.path.join(VIDEO_DIR, "dss_demo_raw.webm")
        if os.path.exists(final_video_name):
            os.remove(final_video_name)
        os.rename(video_path, final_video_name)
        print(f"🎬 Vídeo salvo em: {final_video_name}")

def main():
    try:
        clear_cache()
        app_process = start_app()
        record_demo()
    except Exception as e:
        print(f"❌ Erro durante a automação: {e}")
    finally:
        # Encerra o Streamlit ao finalizar
        if 'app_process' in locals():
            print("🛑 Encerrando servidor Streamlit...")
            os.killpg(os.getpgid(app_process.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
