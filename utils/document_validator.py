"""
Document Validator

This module validates whether the uploaded PDF
is an academic research paper.

No LLM is used here.
Validation is performed using common research
paper section headings.
"""


def is_research_paper(text: str):
    """
    Validate whether the uploaded document
    is an academic research paper.

    Returns:
        (True, message)
        (False, message)
    """

    if not text.strip():
        return (
            False,
            "The uploaded PDF does not contain readable text."
        )

    text = text.lower()

    research_keywords = [
        "abstract",
        "introduction",
        "methodology",
        "materials and methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "keywords",
        "literature review",
    ]

    found_keywords = []

    for keyword in research_keywords:

        if keyword in text:
            found_keywords.append(keyword)

    # At least 4 sections should exist
    if len(found_keywords) >= 4:

        return (
            True,
            "Research paper detected."
        )

    return (
        False,
        """
❌ Unsupported Document.

This application only supports academic research papers.

Please upload documents that contain sections like:

• Abstract
• Introduction
• Methodology
• Results
• Conclusion
• References
"""
    )