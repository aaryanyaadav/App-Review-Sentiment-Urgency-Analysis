class PriorityScorer:

    def compute(self, sentiment, urgency, text):

        score = 0

        # Sentiment
        if sentiment == "Negative":
            score += 0.6
        elif sentiment == "Neutral":
            score += 0.3

        # Urgency
        if urgency == "High":
            score += 0.5
        elif urgency == "Medium":
            score += 0.2

        
        if score >= 0.8:
            return "Critical"
        elif score >= 0.5:
            return "High"
        elif score >= 0.3:
            return "Medium"
        else:
            return "Low"