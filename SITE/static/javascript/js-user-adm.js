document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('jsonModal');
  const modalPre = document.getElementById('modalPre');
  const modalClose = document.getElementById('modalClose');

  function openModal(obj) {
    modalPre.textContent = JSON.stringify(obj, null, 2);
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    modal.hidden = true;
    modalPre.textContent = '';
    document.body.style.overflow = '';
  }

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;

    const action = btn.getAttribute('data-action');

    if (action === 'toggle-answer') {
      const card = btn.closest('.log-card');
      if (!card) return;
      const answer = card.querySelector('.log-a');
      if (!answer) return;
      const isHidden = answer.hidden;
      answer.hidden = !isHidden;
      btn.textContent = isHidden ? 'Ocultar resposta' : 'Ver resposta';
      return;
    }

    if (action === 'show-json') {
      const raw = btn.getAttribute('data-json');
      if (!raw) return;
      try {
        const obj = JSON.parse(raw);
        openModal(obj);
      } catch (err) {
        openModal({ error: 'Não foi possível ler o JSON do log.', details: String(err) });
      }
      return;
    }
  });

  if (modalClose) modalClose.addEventListener('click', closeModal);

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && !modal.hidden) closeModal();
  });
});
