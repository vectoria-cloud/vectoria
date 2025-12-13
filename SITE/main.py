from flask import Flask, render_template, request, redirect, flash, url_for
from funcoes import agt
from os.path import isfile
from json import load
import pandas as pd
import webbrowser

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'QUEPARADA'
app.config['UPLOAD_FOLDER'] = 'static/files'

logado = [True,'01','Guilherme','sisa_web',True]

@app.route("/")
def index():
    global logado
    return render_template("agt-adm.html", logado = logado)


@app.route("/agrotech") 
def agrotech():
    global logado
    if logado[-1] and logado[0]:
        return render_template("agt-adm.html", logado = logado)
    
    elif logado[0]:
        return render_template("agt-user.html", logado = logado)
    else:
        return redirect("/")
    

@app.route("/perfil")
def perfil():
    global logado
    path = rf"files/json/{logado[3]}.json"
    if logado[0] and logado[-1]:
        if not isfile(path):
            return render_template("user-adm.html", logado = logado)
        else:
            with open(path, 'r') as f:
                historico = load(f)
            return render_template("user-adm.html", historico = historico, logado = logado)
        
    elif logado[0]:
        if not isfile(path):
            return render_template("user.html", logado = logado)
        else:
            with open(path, 'r') as f:
                historico = load(f)
            return render_template("user.html", historico = historico, logado = logado)
    else:
        return redirect("/")

    
@app.route("/arquivos")
def arquivos():
    cidades = pd.read_json(r"\files\SISAWEB\dados.json")
    cidade_nome = cidades['nome'].tolist()
    range_cidades = len(cidades)
    return render_template("sisaweb.html", logado = logado, cidades = cidade_nome, range_cidades = range_cidades)

@app.route("/buscar", methods=["POST"])
def buscar():
    cidades = pd.read_json(r"\files\SISAWEB\dados.json")
    cidade_nome = cidades['nome'].tolist()
    range_cidades = len(cidades)
    tipo = request.form['tipo']
    nome = request.form['nome']
    try:
        busca = pd.read_json(rf"C:/Trabalhos/TCC/SITE/files/SISAWEB/tipo {tipo}/{nome} {tipo}.json")
        busca_lista = busca.values.tolist()
        nomes_colunas = busca.columns
        range_colunas = len(nomes_colunas)
        range_busca = len(busca)
        return render_template("sisaweb.html", logado = logado, cidades = cidade_nome, range_cidades = range_cidades, busca_lista = busca_lista, range_busca = range_busca, range_colunas = range_colunas, nomes_colunas =nomes_colunas)
    except Exception as e:
        return render_template("sisaweb.html", logado = logado, cidades = cidade_nome, range_cidades = range_cidades, nada_encotnrado = "NENHUM DADO ENCONTRADO")

@app.route("/guia")   
def guia():
    global logado
    import os
    path = rf'files\ACERVO'
    nomes = os.listdir(path)
    range_nomes = len(nomes)

    return render_template("acervo.html", logado = logado, nomes = nomes, range_nomes = range_nomes)  

 
@app.route("/file_acervo", methods=["POST"])   
def file_acervo():
    file = request.form.get("btn-file")
    pdf_path = rf'files\ACERVO\{file}'
    webbrowser.open_new_tab(pdf_path)
    return redirect("/guia")


@app.route("/api", methods=["POST"])
def api():
    message = request.json.get("message", "")
    if message != "":
        resposta = agt.generate_response(message, str(logado[1]), user_key=logado[3])
        return {
            "role": "assistant",
            "content": resposta
        }
    return {"role": "assistant", "content": ""}


if __name__=='__main__':
    app.run(debug=True)

