from IPython.display import clear_output
import requests
from requests.exceptions import ConnectionError, Timeout, RetryError as RequestsRetryError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError as URLLib3MaxRetryError

from datetime import datetime, timedelta
from json import load, dump
from pathlib import Path
from multiprocessing.synchronize import Event as EventType
import os
import multiprocessing
import shutil
import sys
import signal
import time
import random


INICIO_COLETA = datetime(2020, 1, 1)
FIM_COLETA    = datetime(2024, 12, 31)

REQUEST_SLEEP_SEC     = 0.25    
REQUEST_TIMEOUT_SEC   = 30     
BUFFER_MAX_FALLBACK   = 250_000  

MAX_ATTEMPTS_PER_DAY  = 5      
BACKOFF_START_SEC     = 3.0     
BACKOFF_FACTOR        = 2.0      
BACKOFF_JITTER_MAX    = 1.5      

PAUSA_ENTRE_CIDADES_MIN = 1.0
PAUSA_ENTRE_CIDADES_MAX = 2.0

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Mineracao"
LOG_DIR  = BASE_DIR / "log"
DADOS_JSON_PATH = BASE_DIR / "dados.json"


def gerar_datas(inicio, fim):
    data_atual = inicio
    while data_atual <= fim:
        yield data_atual.strftime('%Y-%m-%d')
        data_atual += timedelta(days=1)

def parse_data_ymd(s):
    return datetime.strptime(s, "%Y-%m-%d")

def proxima_data_str(s):
    return (parse_data_ymd(s) + timedelta(days=1)).strftime("%Y-%m-%d")


def salvar_seguro(path, dados):
    path = Path(path)
    temp_path = str(path) + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        dump(dados, f, indent=4, separators=(",", ":"))
    shutil.move(temp_path, str(path))

def ler_json(path, default):
    try:
        with open(str(path), 'r', encoding='utf-8') as f:
            return load(f)
    except Exception:
        return default

def validar_json(path):
    path = Path(path)
    if not path.is_file():
        return True
    try:
        with open(str(path), 'r', encoding='utf-8') as f:
            load(f)
        return True
    except Exception:
        return False

def validar_arquivos():
    erros = []
    try:
        with open(str(DADOS_JSON_PATH), 'r', encoding='utf-8') as f:
            cidades = load(f)
            if not isinstance(cidades, list):
                raise ValueError("dados.json não contém uma lista.")
    except Exception as e:
        erros.append(f"dados.json inválido ({DADOS_JSON_PATH}): {e}")

    for tipo in range(1, 17):
        path = DATA_DIR / f"Mineracao tipo {tipo}.json"
        if not validar_json(path):
            erros.append(f"{path} está corrompido ou mal formatado.")

        prog = DATA_DIR / f"Progresso tipo {tipo}.json"
        if not validar_json(prog):
            erros.append(f"{prog} está corrompido ou mal formatado.")
    return erros

def carregar_cidades_base():
    with open(str(DADOS_JSON_PATH), 'r', encoding='utf-8') as f:
        cidades_data = load(f)
    return [(int(c['id']), c['nome']) for c in cidades_data if 'id' in c and 'nome' in c]


def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        allowed_methods=frozenset(["GET"]),
        status_forcelist=[429, 502, 503, 504],
        raise_on_status=False,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (ColetaSISAWeb/1.0)"})
    return s

def get_with_retries(session, url, params, timeout):
    attempt = 0
    wait = BACKOFF_START_SEC
    while True:
        attempt += 1
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if 500 <= resp.status_code < 600:
                if attempt >= MAX_ATTEMPTS_PER_DAY:
                    return False, f"ERRO_HTTP_{resp.status_code}: {resp.text[:300]}", resp.status_code
                time.sleep(wait + random.random() * BACKOFF_JITTER_MAX)
                wait *= BACKOFF_FACTOR
                continue
            return True, resp.text, resp.status_code
        except (ConnectionError, Timeout, RequestsRetryError, URLLib3MaxRetryError) as e:
            if attempt >= MAX_ATTEMPTS_PER_DAY:
                return False, f"ERRO_REDE: {type(e).__name__}: {e}", None
            time.sleep(wait + random.random() * BACKOFF_JITTER_MAX)
            wait *= BACKOFF_FACTOR
        except Exception as e:
            return False, f"ERRO: {type(e).__name__}: {e}", None


def caminho_progresso(tipo):
    return DATA_DIR / f"Progresso tipo {tipo}.json"

def carregar_progresso(tipo):
    path = caminho_progresso(tipo)
    return ler_json(path, {})

def salvar_progresso(tipo, progresso_dict):
    path = caminho_progresso(tipo)
    salvar_seguro(path, progresso_dict)

def calcular_faltantes_para_tipo(tipo, ids_cidades):
    progresso = carregar_progresso(tipo)
    faltantes = []
    for cid, nome in ids_cidades:
        ultimo_str = progresso.get(str(cid))
        if not ultimo_str:
            faltantes.append((cid, nome))
        else:
            try:
                ultimo = parse_data_ymd(ultimo_str)
            except Exception:
                faltantes.append((cid, nome))
                continue
            if ultimo < FIM_COLETA:
                faltantes.append((cid, nome))
    return faltantes


def coletar_dados(tipo: int, stop_event: EventType, ids_cidades):
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f'log_tipo_{tipo}.txt'
    log_f = open(str(log_path), 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    DATA_DIR.mkdir(exist_ok=True)
    data_path = DATA_DIR / f'Mineracao tipo {tipo}.json'
    leitura = ler_json(data_path, [])

    vistos = set()
    for item in leitura:
        if isinstance(item, dict):
            k = (item.get('Id'), item.get('Tipo'), item.get('Data'))
            vistos.add(k)

    progresso = carregar_progresso(tipo)
    faltantes = calcular_faltantes_para_tipo(tipo, ids_cidades)

    total_cidades = len(ids_cidades)
    log(f"[INICIANDO] Tipo {tipo} — total de cidades: {total_cidades}, faltantes: {len(faltantes)}")

    if not faltantes:
        log(f"[PULADO] Tipo {tipo} — nenhuma cidade faltante. Encerrando worker.")
        log_f.close()
        return

    buffer = []

    def flush_buffer(motivo):
        nonlocal buffer, leitura, progresso
        if not buffer:
            return
        leitura.extend(buffer)
        salvar_seguro(data_path, leitura)
        log(f"[SALVO] Tipo {tipo} — {len(buffer)} registros gravados ({motivo}).")
        buffer = []

    def flush_progresso():
        salvar_progresso(tipo, progresso)
        log(f"[PROGRESSO] Tipo {tipo} — progresso salvo.")

    session = make_session()

    try:
        for cid, nome in faltantes:
            if interrupted or stop_event.is_set():
                break

            ultimo = progresso.get(str(cid))
            if ultimo:
                inicio_cidade_str = proxima_data_str(ultimo)
                inicio_cidade_dt = parse_data_ymd(inicio_cidade_str)
                if inicio_cidade_dt > FIM_COLETA:
                    log(f"[CIDADE OK] Tipo {tipo} — {cid} {nome} já completa. Pulando.")
                    continue
                inicio_dt = inicio_cidade_dt
            else:
                inicio_dt = INICIO_COLETA

            log(f"[CIDADE INÍCIO] Tipo {tipo} — {cid} {nome} de {inicio_dt.date()} até {FIM_COLETA.date()}")

            ultima_data_processada_para_gravar = None
            try:
                for data in gerar_datas(inicio_dt, FIM_COLETA):
                    if interrupted or stop_event.is_set():
                        raise KeyboardInterrupt

                    params = {
                        'tipo': tipo,
                        'id': cid,
                        'inicio': data,
                        'final': data
                        #'exec': 'Z[&im=1]'
                    }

                    ok, info, status = get_with_retries(
                        session,
                        'https://vigent.saude.sp.gov.br/sisaweb_api/dados.php',
                        params,
                        REQUEST_TIMEOUT_SEC
                    )

                    conteudo = info

                    chave = (cid, tipo, data)
                    if chave not in vistos:
                        buffer.append({
                            'Id': cid,
                            'Nome': nome,
                            'Tipo': tipo,
                            'Data': data,
                            'Informacao': conteudo
                        })
                        vistos.add(chave)

                        if ok:
                            ultima_data_processada_para_gravar = data

                    if len(buffer) >= BUFFER_MAX_FALLBACK:
                        flush_buffer("flush de segurança (buffer grande)")
                        if ultima_data_processada_para_gravar:
                            progresso[str(cid)] = ultima_data_processada_para_gravar
                            flush_progresso()

                    if REQUEST_SLEEP_SEC > 0:
                        time.sleep(REQUEST_SLEEP_SEC + random.random() * 0.1)

                flush_buffer(f"cidade {cid} concluída")
                if ultima_data_processada_para_gravar:
                    progresso[str(cid)] = ultima_data_processada_para_gravar
                    flush_progresso()
                log(f"[CIDADE FIM] Tipo {tipo} — {cid} {nome} concluída.")

                time.sleep(random.uniform(PAUSA_ENTRE_CIDADES_MIN, PAUSA_ENTRE_CIDADES_MAX))

            except KeyboardInterrupt:
                flush_buffer("interrupção")
                if ultima_data_processada_para_gravar:
                    progresso[str(cid)] = ultima_data_processada_para_gravar
                    flush_progresso()
                raise

        flush_buffer("finalização normal")
        flush_progresso()

    except KeyboardInterrupt:
        flush_buffer("interrupção (worker)")
        flush_progresso()
    finally:
        temp_path = str(data_path) + '.tmp'
        if os.path.exists(temp_path) and not interrupted:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        log_f.close()


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)

    DATA_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    erros_encontrados = validar_arquivos()
    if erros_encontrados:
        print("\n[ERROS DETECTADOS]")
        for erro in erros_encontrados:
            print(" -", erro)
        print("\n[ABORTADO] Corrija os erros acima antes de executar o script.")
        sys.exit(1)

    ids_cidades = carregar_cidades_base()

    tipos_a_executar = []
    for tipo in range(1, 17):
        faltantes = calcular_faltantes_para_tipo(tipo, ids_cidades)
        if faltantes:
            tipos_a_executar.append(tipo)
            print(f"[AGENDADO] Tipo {tipo} — faltantes: {len(faltantes)} cidades.")
        else:
            print(f"[PULANDO] Tipo {tipo} — todas as cidades completas.")

    if not tipos_a_executar:
        print("[FINALIZADO] Nada a fazer: todos os tipos estão completos.")
        sys.exit(0)

    stop_event = multiprocessing.Event()

    processos = []
    for tipo in tipos_a_executar:
        p = multiprocessing.Process(target=coletar_dados, args=(tipo, stop_event, ids_cidades))
        p.start()
        processos.append(p)

    try:
        for p in processos:
            p.join()
    except KeyboardInterrupt:
        try:
            clear_output()
        except Exception:
            pass
        print("[PRINCIPAL] Ctrl+C recebido. Solicitando encerramento gracioso (com salvamento)...")
        stop_event.set()

        for p in processos:
            p.join(timeout=25)

        for p in processos:
            if p.is_alive():
                print(f"[PRINCIPAL] Forçando término do processo PID={p.pid}...")
                p.terminate()
        sys.exit(0)
        