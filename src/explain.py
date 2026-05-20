class Explainer:

    def __init__(self, word_index):

        # Reverse mapping
        self.index_word = {
            v: k for k, v in word_index.items()
        }

    def explain(
        self,
        text,
        attention_weights,
        seq,
        top_k=2
    ):

        # Convert attention tensor → numpy
        attn = (
            attention_weights
            .squeeze(-1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        # Sequence tokens
        tokens = seq[0]

        word_scores = []

        for idx, token_id in enumerate(tokens):

            # Skip padding
            if token_id == 0:
                continue

            word = self.index_word.get(
                token_id,
                "<UNK>"
            )

            score = float(attn[idx])

            word_scores.append({
                "word": word,
                "score": score
            })

        # Sort by attention score
        word_scores = sorted(
            word_scores,
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "top_words": word_scores[:top_k],
            "method": "attention"
        }