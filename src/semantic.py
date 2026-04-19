"""
Semantic similarity scorer using sentence-transformers.
Compares transcript embeddings against pre-defined fraud/legit templates.
Model is loaded once and cached.
"""

from __future__ import annotations

import numpy as np

_model = None
_fraud_embeddings = None
_legit_embeddings = None

# Fraud scenario templates (what fraudsters typically say)
FRAUD_TEMPLATES = [
    "Здравствуйте, я сотрудник службы безопасности вашего банка.",
    "На ваш счёт поступила подозрительная операция, нам нужно её заблокировать.",
    "Для подтверждения личности назовите CVV вашей карты.",
    "Пожалуйста, назовите код из SMS, который пришёл вам на телефон.",
    "Ваши деньги находятся в опасности, нужно перевести их на безопасный счёт.",
    "Следователь по вашему делу просит вас снять наличные и положить на указанный счёт.",
    "Никому не сообщайте об этом разговоре, это конфиденциальная информация.",
    "На ваше имя пытаются оформить кредит, нужно срочно этому помешать.",
    "Установите приложение TeamViewer, чтобы наш специалист мог помочь вам.",
    "Я следователь из прокуратуры, ваш счёт заморожен в рамках расследования.",
    "Не кладите трубку, иначе мы не сможем защитить ваши деньги.",
    "Продиктуйте номер вашей карты и срок действия для верификации.",
    "Мошенники уже получили доступ к вашему онлайн-банку, действуйте срочно.",
    "Переведите деньги на резервный счёт банка для их сохранности.",
    "Сотрудник ФСБ звонит в связи с расследованием финансового преступления.",
]

# Legitimate call templates
LEGIT_TEMPLATES = [
    "Добрый день, это служба поддержки, чем могу помочь?",
    "Ваша заявка обрабатывается, результат придёт на электронную почту.",
    "Звоню напомнить о записи к врачу завтра в десять утра.",
    "Хочу уточнить детали доставки вашего заказа.",
    "Мы проводим опрос о качестве нашего обслуживания.",
    "Ваш заказ готов к выдаче, можете забрать в любое время.",
    "Добрый день, это страховая компания, у вас есть вопросы по вашему полису?",
    "Звоню из интернет-провайдера насчёт планового технического обслуживания.",
    "Пожалуйста, оцените качество нашего сервиса по шкале от одного до десяти.",
    "Ваша посылка прибыла на почту, не забудьте забрать.",
]


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _model


def _get_embeddings():
    global _fraud_embeddings, _legit_embeddings
    if _fraud_embeddings is None:
        model = _get_model()
        _fraud_embeddings = model.encode(FRAUD_TEMPLATES, normalize_embeddings=True)
        _legit_embeddings = model.encode(LEGIT_TEMPLATES, normalize_embeddings=True)
    return _fraud_embeddings, _legit_embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (D,) or (N, D), b: (M, D) → (N, M) or (M,)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    # Both already L2-normalised
    return (a @ b.T).squeeze()


def semantic_score(text: str, chunk_size: int = 200) -> float:
    """
    Returns a score in [0, 1].
    Splits text into overlapping chunks, embeds each, computes cosine similarity
    to fraud/legit templates, returns normalised fraud advantage.
    """
    if not text or len(text.strip()) < 20:
        return 0.0

    model = _get_model()
    fraud_emb, legit_emb = _get_embeddings()

    # Split transcript into chunks
    words = text.split()
    step = chunk_size // 2
    chunks = []
    for i in range(0, max(1, len(words) - chunk_size + 1), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    if not chunks:
        chunks = [text]

    chunk_embs = model.encode(chunks, normalize_embeddings=True)  # (C, D)

    # For each chunk: top-3 similarity to fraud / legit
    fraud_sims = chunk_embs @ fraud_emb.T   # (C, F)
    legit_sims = chunk_embs @ legit_emb.T   # (C, L)

    # Max over templates, then mean over chunks
    fraud_score = float(np.mean(np.max(fraud_sims, axis=1)))
    legit_score = float(np.mean(np.max(legit_sims, axis=1)))

    # Advantage: how much more similar to fraud than to legit
    advantage = fraud_score - legit_score
    # Map from roughly [-0.5, 0.5] to [0, 1]
    return float(np.clip((advantage + 0.5) / 1.0, 0.0, 1.0))
