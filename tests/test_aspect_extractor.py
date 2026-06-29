from aspect_extractor import extract_aspects


def test_delivery():

    review = "The delivery was delayed."

    assert "Delivery" in extract_aspects(review)


def test_packaging():

    review = "Packaging was excellent."

    assert "Packaging" in extract_aspects(review)


def test_price():

    review = "The product is too expensive."

    assert "Price" in extract_aspects(review)


def test_quality():

    review = "The quality is amazing."

    assert "Product Quality" in extract_aspects(review)


def test_customer_service():

    review = "Customer support solved my issue."

    assert "Customer Service" in extract_aspects(review)


def test_multiple_aspects():

    review = (
        "Delivery was late but "
        "packaging was good."
    )

    result = extract_aspects(review)

    assert "Delivery" in result
    assert "Packaging" in result


def test_no_aspect():

    review = "I bought this yesterday."

    assert extract_aspects(review) == []