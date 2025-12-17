const url = 'C:\Projeto_Integrador_III\Site\agrotech-final\files\acervo\artigo\Manejo do solo.pdf'
const btn = document.querySelector('#btn-file')

function openInNewTab(url){
    const win = window.open(url,'_blank')
}

btn.addEventListener('click',()=>{
    openInNewTab(url)
})