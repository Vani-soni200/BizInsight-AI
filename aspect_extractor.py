"""
Aspect Extraction Module for BizInsight AI

This module identifies business-related aspects mentioned
in customer reviews using rule-based keyword matching.
"""

import re

ASPECT_KEYWORDS = {
    "Product Quality": [
        "quality",
        "durable",
        "defective",
        "broken",
        "damage",
        "material",
        "performance",
        "poor quality",
        "excellent quality",
        "good quality"
    ],

    "Price": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "value",
        "worth",
        "pricing",
        "affordable"
    ],

    "Delivery": [
        "delivery",
        "shipping",
        "shipment",
        "courier",
        "late",
        "delay",
        "arrived",
        "dispatch",
        "delivered"
    ],

    "Packaging": [
        "packaging",
        "package",
        "box",
        "packed",
        "seal",
        "wrapped",
        "wrapper"
    ],

    "Customer Service": [
        "service",
        "support",
        "staff",
        "customer care",
        "representative",
        "agent",
        "helpdesk",
        "response"
    ]
}


def clean_text(text: str) -> str:
    """
    Clean review text for keyword matching.

    Args:
        text (str): Customer review.

    Returns:
        str: Cleaned lowercase text.
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_aspects(review: str):
    """
    Detect aspects mentioned in a review.

    Args:
        review (str)

    Returns:
        list[str]
    """

    review = clean_text(review)

    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in review:
                detected.append(aspect)
                break

    return detected


def extract_aspects_bulk(reviews):
    """
    Detect aspects for multiple reviews.

    Args:
        reviews (list[str])

    Returns:
        list[list[str]]
    """

    return [extract_aspects(review) for review in reviews]