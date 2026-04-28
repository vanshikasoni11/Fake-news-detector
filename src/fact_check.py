import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

def fact_check(query):

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"

    response = requests.get(url, timeout=10)
    data = response.json()

    claims = data.get("claims", [])

    if not claims:
        return None

    results = []

    for c in claims[:3]:
        review = c.get("claimReview", [{}])[0]

        results.append({
            "text": c.get("text", "No claim text"),
            "rating": review.get("textualRating", "No rating"),
            "publisher": review.get("publisher", {}).get("name", "Unknown")
        })

    return results