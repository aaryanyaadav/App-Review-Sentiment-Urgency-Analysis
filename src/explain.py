import numpy as np
class Explainer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.index_word = {v: k for k, v in tokenizer.word_index.items()}

    def explain(self, text, attention_weights, seq, top_k=2):  #we  provide the text ,attention weights , sequence and the top words 

        # Convert attention tensor → numpy
        attn = attention_weights.squeeze().detach().cpu().numpy()

        tokens = seq[0]  # padded sequence

        word_attention = []

        for i, token_id in enumerate(tokens):
            if token_id == 0:
                continue  # skip the padding

            if i >= len(attn):
                break

            word = self.index_word.get(token_id, "")

            if word != "":
                word_attention.append({
                    "word": word,
                    "score": float(attn[i])
                })

        #sorting the words by attention 
        word_attention = sorted(
            word_attention,
            key=lambda x: x["score"],
            reverse=True
        )

        # the top words
        top_words = word_attention[:top_k]

        return {
            "top_words": top_words
            
        }