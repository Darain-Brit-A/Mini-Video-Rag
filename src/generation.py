"""Grounded answer generation with an optional OpenAI wording step."""

import os

from dotenv import load_dotenv

load_dotenv()

REFUSAL = "I could not find enough information about this in the video."


def generate_answer(
    question: str, evidence: list[dict[str, object]], threshold: float = 0.30
) -> str:
    """Answer only from evidence; use an extractive fallback without an API key."""
    best_score = max((float(item.get("similarity", 0.0)) for item in evidence), default=0.0)
    refusal_message = (
        f"I could not find enough information about this in the video "
        f"(best match: {best_score:.2f}, threshold: {threshold:.2f})."
    )

    # Refusing on weak evidence is the grounding guard against unsupported answers.
    if (
        not evidence
        or best_score < threshold
        or max((len(str(item["text"]).split()) for item in evidence), default=0) < 5
    ):
        return refusal_message

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI

            context = "\n\n".join(
                f"[{item['start_time']} - {item['end_time']}] {item['text']}" for item in evidence
            )
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the transcript evidence supplied by the user. "
                            f"If it does not contain the answer, say exactly: {refusal_message}"
                        ),
                    },
                    {"role": "user", "content": f"Question: {question}\nEvidence:\n{context}"},
                ],
            )
            ans = response.choices[0].message.content.strip()
            if REFUSAL in ans:
                return refusal_message
            return ans
        except Exception:
            pass

    best = evidence[0]
    return (
        f"According to the video ({best['start_time']} - {best['end_time']}), "
        f"the relevant passage says: \"{best['text']}\""
    )
