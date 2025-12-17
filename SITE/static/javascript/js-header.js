var btnExp = document.querySelector('#btn-expand');
var menu = document.querySelector('.menu-lateral');
var content = document.querySelector('.content');

btnExp.addEventListener('click', function(){
    menu.classList.toggle('expandir');
    content.classList.toggle('recall');
})