import re
from typing import List

sentences: List[str] = [
    "وتفضلوا بقبول فائق الاحترام والتقدير",
    "وفي انتظار معالجتكم لشكايتنا تقبلو فائق التقدير والاحترام",
    "سلام تام بوجود مولانا الإمام، دام له النصر والتأييد",
    "veuillez agréer, madame, monsieur, l’expression de ma considération distinguée",
    "cordialement",
    "bien cordialement",
    "والسلام",
    "و السلام",

    "لأجل ذلك فإن العارض يتقدم إلى سيادتكم بهذه الشكاية طالبا منكم تطبيق مقتضيات المادة 279 من القانون رقم 99",
    "17 مع أمر شركة التأمين المشتكى بها بتنفيذ مقتضيات القرار المذكور",
]
def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Input text string.
        
        Returns:
            Estimated token count.
        """
        if not self.model:
            self._load_model()
        tokens = self.model.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )
        return len(tokens["input_ids"][0])
class TextCleaner:
    def __init__(self, sentences_to_remove: List[str]):
        self._patterns = [
            re.compile(rf"\s*{re.escape(s.strip())}\s*", flags=re.IGNORECASE)
            for s in sentences_to_remove
            if s and s.strip()
        ]

    def remove_sentences(self, text: str) -> str:
        """
        Remove specific sentences from the text (robust to extra spaces/newlines).

        Args:
            text: input text
        Returns:
            cleaned text without those sentences
        """
        if not text:
            return text

        cleaned = text
        for pat in self._patterns:
            cleaned = pat.sub(" ", cleaned)

        # cleanup: collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

