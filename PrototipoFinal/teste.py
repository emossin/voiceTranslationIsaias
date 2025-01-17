import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence
import os
import uuid
import numpy as np
import sounddevice as sd
import torch
import whisper
from pydub import AudioSegment
from scipy.io.wavfile import write
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import pygame
from threading import Thread
from deep_translator import GoogleTranslator
import time
import gc
from tkinter import ttk
from threading import Event

# Configuração da chave da API ElevenLabs
ELEVENLABS_API_KEY = 'sk_60884db1a3626109ab9be2b13b8fd89adeb3ef5491789be6'
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Lista de idiomas suportados
idiomas_suportados = {
    "Chinês": "zh-CN",
    "Coreano": "ko",
    "Neerlandês": "nl",
    "Turco": "tr",
    "Sueco": "sv",
    "Indonésio ": "id",
    "Filipino": "tl",
    "Japonês": "ja",
    "Ucraniano": "uk",
    "Grego": "el",
    "Tcheco": "cs",
    "Finlandês": "fi",
    "Romeno": "ro",
    "Russo": "ru",
    "Dinamarquês": "da",
    "Búlgaro": "bg",
    "Malaio": "ms",
    "Eslovaco": "sk",
    "Croata": "hr",
    "Árabe": "ar",
    "Tâmil": "ta",
    "Inglês": "en",
    "Polonês": "pl",
    "Alemão": "de",
    "Espanhol": "es",
    "Francês": "fr",
    "Italiano": "it",
    "Híndi": "hi",
    "Português": "pt",
    "Húngaro": "hu",
    "Vietnamita": "vi",
    "Norueguês": "no"
}


def alterar_idioma(event):
    global idioma_selecionado
    idioma_selecionado.set(combo_idiomas.get())

# Função para registrar logs
def registrar_log(mensagem):
    with open("log_execucaoelevenlabsturbo.txt", "a") as log:
        log.write(f"{mensagem}\n")

# Função para síntese de texto para fala e conversão para wav
def text_to_speech_file(text: str) -> str:
    inicio_tts = time.time()
    response = client.text_to_speech.convert(
        voice_id="cgSgspJ2msm6clMCkdW9",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    
    mp3_file_path = f"{uuid.uuid4()}.mp3"
    with open(mp3_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    wav_file_path = "output.wav"
    audio_segment = AudioSegment.from_mp3(mp3_file_path)
    audio_segment.export(wav_file_path, format="wav")
    os.remove(mp3_file_path)

    fim_tts = time.time()
    registrar_log(f"Síntese de voz concluída em {fim_tts - inicio_tts:.2f} segundos")

    return wav_file_path

# Inicializar o Pygame
pygame.init()


# Carregar o modelo Whisper
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model("small")


# Função para processar áudio e exibir transcrição e tradução
def processar_audio(file_path):
    if model is None:
        registrar_log("Erro: modelo Whisper não foi carregado.")
        return
    
    registrar_log(f"Início do processamento de áudio em {time.ctime()}")
    inicio_processamento = time.time()
    audio = whisper.load_audio(file_path)
    fim_processamento = time.time()
    registrar_log(f"Processamento do áudio concluído em {fim_processamento - inicio_processamento:.2f} segundos")

    inicio_decodificacao = time.time()
    result = model.transcribe(audio)
    transcricao = (result["text"])
    fim_decodificacao = time.time()
    registrar_log(f"Decodificação concluída em {fim_decodificacao - inicio_decodificacao:.2f} segundos")
    registrar_log(f"Texto reconhecido: {transcricao}")

    inicio_traducao = time.time()
    try:
        inicio_traducao = time.time()
        codigo_idioma = idiomas_suportados[idioma_selecionado.get()]  # Obtem o código do idioma
        translated = GoogleTranslator(source='auto', target=codigo_idioma).translate(transcricao)
        fim_traducao = time.time()
        registrar_log(f"Tradução concluída em {fim_traducao - inicio_traducao:.2f} segundos")
        registrar_log(f"Texto traduzido ({idioma_selecionado.get()}): {translated}")
    except Exception as e:
        registrar_log(f"Erro ao traduzir texto: {e}")
        translated = "Erro ao traduzir o texto."
    fim_traducao = time.time()
    registrar_log(f"Tradução concluída em {fim_traducao - inicio_traducao:.2f} segundos")
    registrar_log(f"Texto traduzido: {translated}")

    audio_path = text_to_speech_file(translated)
    if os.path.exists(audio_path):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Aguarde a reprodução terminar
        pygame.mixer.quit()

        balao_transcricao.config(text="Transcrição: " + transcricao)
        balao_traducao.config(text=f"Tradução ({idioma_selecionado.get()}): {translated}")

        return transcricao  # Retorna a transcrição em português

# Função para selecionar um áudio
def selecionar_audio():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.wma *.opus *.amr")]
    )
    if file_path:
        # Lista de extensões que requerem conversão para .mp3
        extensoes_para_converter = ['.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus', '.amr']
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in extensoes_para_converter:
            registrar_log(f"Arquivo {ext.upper()} selecionado, convertendo para MP3...")
            audio = AudioSegment.from_file(file_path, format=ext[1:])
            mp3_file_path = "audio_temp.mp3"
            audio.export(mp3_file_path, format="mp3")
            file_path = mp3_file_path  # Atualiza o caminho para o arquivo MP3 convertido
            registrar_log("Conversão para MP3 concluída.")

        thread_audio = Thread(target=processar_audio, args=(file_path,))
        thread_audio.start()

# Função para gravar e processar áudio

# Variáveis globais para controle de gravação
gravando = Event()
gravacao_buffer = []



def iniciar_gravacao():
    """Alterna entre iniciar e parar gravação."""
    global thread_gravacao

    if not gravando.is_set():
        # Configura para iniciar a gravação
        gravando.set()
        gravacao_buffer.clear()
        registrar_log("Iniciando gravação...")
        botao_gravacao.config(text="PARAR GRAVAÇÃO")
        balao_transcricao.config(text="Gravando áudio... Pressione 'PARAR GRAVAÇÃO' para finalizar.")
        janela.update_idletasks()

        # Inicia a gravação em uma thread
        thread_gravacao = Thread(target=gravar_audio)
        thread_gravacao.start()
    else:
        # Configura para parar a gravação
        gravando.clear()
        registrar_log("Finalizando gravação e processando...")
        botao_gravacao.config(text="GRAVAR ÁUDIO")
        balao_transcricao.config(text="Processando gravação, aguarde...")
        janela.update_idletasks()

        # Processa o áudio gravado em uma nova thread
        thread_processamento = Thread(target=processar_gravacao)
        thread_processamento.start()


def gravar_audio():
    """Função que grava áudio em tempo real enquanto 'gravando' estiver ativo."""
    fs = 44100  # Taxa de amostragem
    try:
        while gravando.is_set():
            bloco = sd.rec(int(fs * 1), samplerate=fs, channels=2)  # Grava 1 segundo por vez
            sd.wait()
            gravacao_buffer.append(bloco)
    except Exception as e:
        registrar_log(f"Erro durante a gravação: {e}")
        gravando.clear()


def processar_gravacao():
    """Processa o áudio gravado e atualiza a interface."""
    try:
        # Combina os blocos gravados
        audio_data = np.concatenate(gravacao_buffer, axis=0)
        write("gravacao.wav", 44100, audio_data)
        audio = AudioSegment.from_wav("gravacao.wav")
        audio.export("gravacao.mp3", format="mp3")
        os.remove("gravacao.wav")

        # Processa o áudio e exibe a transcrição
        transcricao = processar_audio("gravacao.mp3")
        balao_transcricao.config(text="Transcrição (PT): " + transcricao)
    except Exception as e:
        registrar_log(f"Erro durante o processamento do áudio: {e}")
        balao_transcricao.config(text="Erro ao processar o áudio gravado.")
    finally:
        registrar_log("Processamento de áudio concluído.")
        janela.update_idletasks()

# Configuração da interface principal
janela = tk.Tk()

idioma_selecionado = tk.StringVar(value="en")

modelo_vram = tk.StringVar(value="small") 
model = None


# Configurações do programa
janela.title("Tradutor Whisper Turbo")  # Altere o título do programa aqui

# Alterar o ícone da janela (use o nome do seu arquivo .ico)
try:
    janela.iconbitmap("icone.ico")  # Certifique-se de que o arquivo 'icone.ico' está no mesmo diretório
except Exception as e:
    print(f"Erro ao carregar o ícone: {e}")

janela.geometry("1280x720")
janela.configure(bg="#1a1a1a")

# Cores e fontes
cor_fundo = "#1a1a1a"
cor_destaque = "#9b00ff"
cor_texto = "#00ff99"
cor_branco = "#ffffff"
cor_alerta = "#ff4444"
fonte_titulo = ("Fixedsys", 16, "bold")
fonte_roboto = ("Roboto", 16, "bold")
fonte_conteudo = ("Fixedsys", 10)

# Função para estilizar botões
def estilizar_botao(botao, largura=20):
    botao.config(
        font=fonte_conteudo, bg=cor_destaque, fg="#ffffff", activebackground=cor_alerta,activeforeground="#ffffff", relief="solid", borderwidth=2, padx=15, pady=5,width=largura
   )

# Função para alternar telas
def mostrar_tela(frame):
    frame.tkraise()

# Carrega a imagem da logo
try:
    logo_rodape_imagem = Image.open("logo_pequena.png").resize((100, 100))  # Substitua "logo_pequena.png" pela sua imagem
    logo_rodape_imagem = ImageTk.PhotoImage(logo_rodape_imagem)
except FileNotFoundError:
    logo_rodape_imagem = None

def adicionar_logo_rodape(frame):
    """Adiciona a logo no canto inferior direito de um frame."""
    if logo_rodape_imagem:
        logo_rodape_label = tk.Label(frame, image=logo_rodape_imagem, bg=cor_fundo)
        logo_rodape_label.image = logo_rodape_imagem
        logo_rodape_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)  # Ajusta para o canto inferior direito

def exibir_ajuda():
    # Cria uma nova janela
    ajuda_janela = tk.Toplevel()
    ajuda_janela.title("Ajuda")
    ajuda_janela.geometry("600x400")
    ajuda_janela.configure(bg=cor_fundo)

    # Título
    titulo_ajuda = tk.Label(
        ajuda_janela, text="Ajuda - Funcionamento do Programa", font=fonte_titulo, fg=cor_branco, bg=cor_fundo
    )
    titulo_ajuda.pack(pady=10)

    # Conteúdo de ajuda
    conteudo = (
        "Bem-vindo ao Tradutor Whisper Turbo!\n\n"
        "Este programa permite que você: \n"
        "1. Grave áudio diretamente do microfone e o processe.\n"
        "2. Carregue arquivos de áudio para transcrição e tradução.\n"
        "3. Escute a tradução sintetizada em áudio.\n\n"
        "Funcionalidades: \n"
        "- Clique em 'INICIAR' para acessar a interface principal.\n"
        "- Use 'Selecionar Áudio' para carregar arquivos de áudio para transcrição e tradução.\n"
        "- Use 'Gravar Áudio' para capturar áudio com o microfone.\n"
        "- O programa exibirá a transcrição e tradução no idioma escolhido.\n"
        "- A tradução será reproduzida em voz sintetizada.\n"
        "\nPara dúvidas ou suporte, entre em contato com o desenvolvedor."
    )

    texto_ajuda = tk.Label(
        ajuda_janela, text=conteudo, font=fonte_conteudo, fg=cor_texto, bg=cor_fundo, wraplength=580, justify="left"
    )
    texto_ajuda.pack(pady=10, padx=20)

    # Botão para fechar a janela de ajuda
    botao_fechar = tk.Button(
        ajuda_janela, text="FECHAR", command=ajuda_janela.destroy
    )
    estilizar_botao(botao_fechar, largura=15)
    botao_fechar.pack(pady=20)

# Tela do menu principal
menu_principal = tk.Frame(janela, bg=cor_fundo)
menu_principal.place(relwidth=1, relheight=1)

# Adiciona logo centralizado no menu principal
try:
    logo_imagem = Image.open("logo.png").resize((500, 500))  # Substitua "logo.png" pela sua imagem
    logo_imagem = ImageTk.PhotoImage(logo_imagem)
    logo_label = tk.Label(menu_principal, image=logo_imagem, bg=cor_fundo)
    logo_label.image = logo_imagem
    logo_label.pack(pady=0)
except FileNotFoundError:
    logo_label = tk.Label(menu_principal, text="LOGO", font=fonte_titulo, fg=cor_branco, bg=cor_fundo)
    logo_label.pack(pady=0)

# Botões do menu principal
botao_iniciar = tk.Button(menu_principal, text="INICIAR", command=lambda: mostrar_tela(tela_principal))
estilizar_botao(botao_iniciar, largura=20)
botao_iniciar.pack(pady=5)

botao_configuracoes = tk.Button(menu_principal, text="CONFIGURAÇÕES", command=lambda: mostrar_tela(tela_configuracoes))
estilizar_botao(botao_configuracoes, largura=20)
botao_configuracoes.pack(pady=5)

botao_ajuda = tk.Button(menu_principal, text="AJUDA", command=lambda: exibir_ajuda())
estilizar_botao(botao_ajuda, largura=20)
botao_ajuda.pack(pady=5)

botao_sair = tk.Button(menu_principal, text="SAIR", command=janela.quit)
estilizar_botao(botao_sair, largura=10)
botao_sair.pack(pady=15)

adicionar_logo_rodape(menu_principal)

# Tela principal
tela_principal = tk.Frame(janela, bg=cor_fundo)
tela_principal.place(relwidth=1, relheight=1)

titulo = tk.Label(tela_principal, text="TRADUTOR W TURBO/EL", font=fonte_titulo, fg=cor_branco, bg=cor_fundo)
titulo.pack(pady=10)

balao_transcricao = tk.Label(
    tela_principal, text="Transcrição: ", font=fonte_conteudo, fg=cor_texto, bg=cor_fundo,
    padx=10, pady=5, wraplength=550, borderwidth=3, relief="solid"
)
balao_transcricao.pack(pady=5, padx=10, anchor="w", fill="x")

balao_traducao = tk.Label(
    tela_principal, text="Tradução: ", font=fonte_conteudo, fg=cor_texto, bg=cor_fundo,
    padx=10, pady=5, wraplength=550, borderwidth=3, relief="solid"
)
balao_traducao.pack(pady=5, padx=10, anchor="w", fill="x")

#Botões
botao_gravacao = tk.Button(tela_principal, text="GRAVAR ÁUDIO", command=iniciar_gravacao)
estilizar_botao(botao_gravacao)
botao_gravacao.pack(pady=10)
botao_gravacao.place(relx=0.25, rely=0.3, anchor="w")

botao_selecionar = tk.Button(tela_principal, text="SELECIONAR ÁUDIO", command=selecionar_audio)
estilizar_botao(botao_selecionar)
botao_selecionar.pack(pady=10)
botao_selecionar.place(relx=0.25, rely=0.36, anchor="w")

# Dropdown para seleção de idioma
label_idioma = tk.Label(tela_principal, text="Selecione o idioma alvo:", font=fonte_conteudo, fg=cor_texto, bg=cor_fundo)
label_idioma.pack(pady=5)
label_idioma.place(relx=0.6, rely=0.30, anchor="w")

combo_idiomas = ttk.Combobox(tela_principal, values=list(idiomas_suportados.keys()), state="readonly", width=30)  
combo_idiomas.bind("<<ComboboxSelected>>", alterar_idioma)
combo_idiomas.pack(pady=5)

combo_idiomas.set("Selecione o idioma")  # Texto inicial
combo_idiomas.place(relx=0.6, rely=0.33, anchor="w")  # Ajustado para ficar mais acima

botao_voltar = tk.Button(tela_principal, text="VOLTAR", command=lambda: mostrar_tela(menu_principal))
estilizar_botao(botao_voltar)
botao_voltar.pack(side="bottom", pady=45)

adicionar_logo_rodape(tela_principal)

# Tela de configurações

tela_configuracoes = tk.Frame(janela, bg=cor_fundo)
tela_configuracoes.place(relwidth=1, relheight=1)

titulo_config = tk.Label(tela_configuracoes, text="CONFIGURAÇÕES", font=fonte_titulo, fg=cor_branco, bg=cor_fundo)
titulo_config.pack(pady=20)

label_status_modelo = tk.Label(
    tela_configuracoes, text="Modelo atual: Nenhum carregado", 
    font=fonte_conteudo, fg=cor_texto, bg=cor_fundo, wraplength=600, justify="left"
)
label_status_modelo.pack(pady=10)

def carregar_modelo_threaded():
    """Função que cria uma thread para carregar o modelo."""
    thread = Thread(target=carregar_modelo, daemon=True)  # Daemon permite encerramento automático
    thread.start()

def carregar_modelo():
    """Carrega o modelo Whisper com base na configuração de VRAM selecionada."""
    global model
    modelo_selecionado = modelo_vram.get()

    # Atualiza o rótulo com mensagem de carregamento
    label_status_modelo.config(text=f"Carregando modelo {modelo_selecionado}... Aguarde.")
    janela.update_idletasks()

    # Libera a memória de vídeo de forma robusta
    if model is not None:
        registrar_log("Liberando o modelo Whisper atual da memória.")
        try:
            del model  # Remove a referência ao modelo atual
            torch.cuda.empty_cache()  # Libera a memória não usada
            torch.cuda.synchronize()  # Garante finalização das operações pendentes
            torch.cuda.reset_peak_memory_stats()  # Reseta os contadores de uso
            registrar_log("Memória de vídeo liberada com sucesso.")
        except Exception as e:
            registrar_log(f"Erro ao liberar memória de vídeo: {e}")
    
    # Carrega o novo modelo
    registrar_log(f"Carregando modelo Whisper: {modelo_selecionado}")
    try:
        model = whisper.load_model(modelo_selecionado, device=device)
        registrar_log(f"Modelo {modelo_selecionado} carregado com sucesso.")
        label_status_modelo.config(text=f"Modelo {modelo_selecionado} carregado com sucesso.")
    except Exception as e:
        registrar_log(f"Erro ao carregar o modelo {modelo_selecionado}: {e}")
        model = None
        label_status_modelo.config(text=f"Erro ao carregar modelo {modelo_selecionado}.")
    finally:
        janela.update_idletasks()


def alterar_modelo():
    """Manipula a mudança de modelo selecionado."""
    registrar_log("Alteração de modelo solicitada pelo usuário.")
    carregar_modelo_threaded()

carregar_modelo()

# Radiobutton para 2GB (Whisper Small)
radio_vram_2gb = tk.Radiobutton(
    tela_configuracoes,
    text="2GB (Whisper Small)",
    variable=modelo_vram,
    value="small",
    font=fonte_conteudo,
    fg=cor_texto,
    bg=cor_fundo,
    selectcolor=cor_destaque,
    activebackground=cor_destaque,
    activeforeground=cor_branco,
    command=carregar_modelo  # Atualiza o modelo ao alterar a seleção
)
radio_vram_2gb.pack(pady=5)

# Radiobutton para 6GB (Whisper Medium)
radio_vram_6gb = tk.Radiobutton(
    tela_configuracoes,
    text="6GB (Whisper Medium)",
    variable=modelo_vram,
    value="medium",
    font=fonte_conteudo,
    fg=cor_texto,
    bg=cor_fundo,
    selectcolor=cor_destaque,
    activebackground=cor_destaque,
    activeforeground=cor_branco,
    command=carregar_modelo
)
radio_vram_6gb.pack(pady=5)

# Radiobutton para +8GB (Whisper Turbo)
radio_vram_8gb = tk.Radiobutton(
    tela_configuracoes,
    text="+8GB (Whisper Turbo)",
    variable=modelo_vram,
    value="turbo",
    font=fonte_conteudo,
    fg=cor_texto,
    bg=cor_fundo,
    selectcolor=cor_destaque,
    activebackground=cor_destaque,
    activeforeground=cor_branco,
    command=carregar_modelo
)
radio_vram_8gb.pack(pady=5)

botao_voltar_config = tk.Button(tela_configuracoes, text="VOLTAR", command=lambda: mostrar_tela(menu_principal))
estilizar_botao(botao_voltar_config)
botao_voltar_config.pack(pady=10)

adicionar_logo_rodape(tela_configuracoes)

# Exibe a tela inicial
mostrar_tela(menu_principal)

janela.mainloop()