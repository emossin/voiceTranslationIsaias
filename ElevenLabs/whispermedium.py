import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageSequence
import os
import uuid
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

ELEVENLABS_API_KEY = 'sk_60884db1a3626109ab9be2b13b8fd89adeb3ef5491789be6'
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def registrar_log(mensagem):
    with open("log_execucaoelevenlabsmedium.txt", "a") as log:
        log.write(f"{mensagem}\n")

def text_to_speech_file(text: str) -> str:
    inicio_tts = time.time()
    response = client.text_to_speech.convert(
        voice_id="cgSgspJ2msm6clMCkdW9",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
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
    registrar_log(f"Sintese de voz concluída em {fim_tts - inicio_tts:.2f} segundos")

    return wav_file_path

pygame.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model("medium")

def processar_audio(file_path):
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
    translated = GoogleTranslator(source='auto', target='en').translate(transcricao)
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
        balao_traducao.config(text="Tradução: " + translated)

        return transcricao  # Retorna a transcrição em português

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

def gravar_e_processar(): 
    # Define a função interna para gravar o áudio
    def gravar_audio():
        fs = 44100  # Taxa de amostragem
        segundos = 10  # Duração da gravação
        registrar_log("Iniciando gravação de áudio...")

        try:
            # Feedback visual na interface
            balao_transcricao.config(text="Gravando áudio... Aguarde.")
            janela.update_idletasks()  # Atualiza a interface gráfica
            
            # Gravação de áudio
            gravacao = sd.rec(int(segundos * fs), samplerate=fs, channels=2)
            sd.wait()  # Aguarda a gravação terminar
            registrar_log("Gravação concluída com sucesso.")
            
            # Feedback de gravação concluída
            balao_transcricao.config(text="Gravação concluída! Processando...")
            janela.update_idletasks()

            # Converter para MP3
            if os.path.exists("gravacao.wav"):
                os.remove("gravacao.wav")  # Remove o arquivo antigo se existir

            write("gravacao.wav", fs, gravacao)
            audio = AudioSegment.from_wav("gravacao.wav")
            audio.export("gravacao.mp3", format="mp3")
            os.remove("gravacao.wav")  # Remove o WAV após conversão
            
            # Processar o áudio gravado
            transcricao = processar_audio("gravacao.mp3")
            
            # Atualiza o balão com a transcrição em português após a reprodução
            balao_transcricao.config(text="Transcrição (PT): " + transcricao)
            janela.update_idletasks()
        
        except Exception as e:
            registrar_log(f"Erro durante a gravação de áudio: {e}")
            balao_transcricao.config(text="Erro na gravação de áudio. Verifique o microfone e tente novamente.")
        finally:
            janela.update_idletasks()

    # Inicia a gravação em uma thread separada para evitar travamentos
    thread_gravacao = Thread(target=gravar_audio)
    thread_gravacao.start()


janela = tk.Tk()
janela.title("Tradutor c/ Whisper")  
janela.geometry("600x400")
janela.configure(bg="#1a1a1a")

cor_fundo = "#1a1a1a"
cor_destaque = "#9b00ff"
cor_texto = "#00ff99"
cor_alerta = "#ff4444"
fonte_titulo = ("Fixedsys", 16, "bold")
fonte_conteudo = ("Fixedsys", 10)

titulo = tk.Label(janela, text="TRADUTOR W MEDIUM/EL", font=fonte_titulo, fg=cor_destaque, bg=cor_fundo)
titulo.pack(pady=10)

balao_transcricao = tk.Label(
    janela, text="Transcrição: ", font=fonte_conteudo, fg=cor_texto, bg=cor_fundo,
    padx=10, pady=5, wraplength=550, borderwidth=3, relief="solid"
)
balao_transcricao.pack(pady=5, padx=10, anchor="w", fill="x")

balao_traducao = tk.Label(
    janela, text="Tradução: ", font=fonte_conteudo, fg=cor_texto, bg=cor_fundo,
    padx=10, pady=5, wraplength=550, borderwidth=3, relief="solid"
)
balao_traducao.pack(pady=5, padx=10, anchor="w", fill="x")

def estilizar_botao(botao):
    botao.config(
        font=fonte_conteudo, bg=cor_alerta, fg="#ffffff", activebackground=cor_destaque,
        activeforeground="#ffffff", relief="solid", borderwidth=2, padx=15, pady=5
    )

botao_gravar = tk.Button(janela, text="GRAVAR ÁUDIO", command=gravar_e_processar)
estilizar_botao(botao_gravar)
botao_gravar.pack(pady=10)

botao_selecionar = tk.Button(janela, text="SELECIONAR ÁUDIO", command=selecionar_audio)
estilizar_botao(botao_selecionar)
botao_selecionar.pack(pady=10)

janela.mainloop()
