"""
Sample MMLU questions across diverse subjects for benchmarking.
Each question includes the subject, question, choices, and correct answer.
"""

MMLU_QUESTIONS = [
    {
        "subject": "abstract_algebra",
        "question": "Find the degree of the extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
        "choices": ["0", "4", "2", "6"],
        "answer": "B",
    },
    {
        "subject": "anatomy",
        "question": "Which of the following is the body cavity that contains the pituitary gland?",
        "choices": ["Abdominal", "Move", "Cranial", "Thoracic"],
        "answer": "C",
    },
    {
        "subject": "astronomy",
        "question": "Why is Mars red?",
        "choices": [
            "Because the surface isite volcanic rock.",
            "Because of the iron oxide minerals in the soil.",
            "Because of its liquid water ocean.",
            "Because of a thick carbon dioxide atmosphere.",
        ],
        "answer": "B",
    },
    {
        "subject": "business_ethics",
        "question": "Beyond the business case for engagement with consumers, what other reasons exist for doing so?",
        "choices": [
            "Ethical and moral reasons",
            "Legal reasons",
            "Regulatory reasons",
            "All of the above",
        ],
        "answer": "D",
    },
    {
        "subject": "clinical_knowledge",
        "question": "The longest stage of the cell cycle is:",
        "choices": ["M phase", "G1 phase", "S phase", "G2 phase"],
        "answer": "B",
    },
    {
        "subject": "computer_science",
        "question": "Which of the following is an example of a lossy compression algorithm?",
        "choices": ["Run-length encoding", "Huffman coding", "JPEG", "LZW"],
        "answer": "C",
    },
    {
        "subject": "high_school_physics",
        "question": "A car starts from rest and accelerates uniformly at 2 m/s^2 for 10 seconds. What distance does it cover?",
        "choices": ["50 m", "100 m", "200 m", "20 m"],
        "answer": "B",
    },
    {
        "subject": "world_religions",
        "question": "What is the name of the Hindu god of destruction?",
        "choices": ["Vishnu", "Brahma", "Shiva", "Indra"],
        "answer": "C",
    },
    {
        "subject": "philosophy",
        "question": "According to Kant, what is the supreme principle of morality?",
        "choices": [
            "The Golden Rule",
            "The Categorical Imperative",
            "The Greatest Happiness Principle",
            "The Social Contract",
        ],
        "answer": "B",
    },
    {
        "subject": "global_facts",
        "question": "As of 2019, about what percentage of the world's population lives in urban areas?",
        "choices": ["25%", "40%", "55%", "70%"],
        "answer": "C",
    },
]


def format_mmlu_prompt(q: dict) -> str:
    """Format an MMLU question as a prompt string."""
    prompt = f"Question: {q['question']}\n"
    for i, choice in enumerate(q['choices']):
        label = chr(65 + i)  # A, B, C, D
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer: Let me think step by step.\n"
    return prompt


def get_all_prompts():
    """Return list of (subject, prompt, correct_answer) tuples."""
    return [
        (q["subject"], format_mmlu_prompt(q), q["answer"])
        for q in MMLU_QUESTIONS
    ]
