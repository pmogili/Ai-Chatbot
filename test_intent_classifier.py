"""
tests/test_intent_classifier.py
----------------------------------
Unit tests for the TF-IDF + Logistic Regression intent classifier.
"""

from services.intent_classifier import IntentClassifier


def test_classifier_trains_without_error():
    classifier = IntentClassifier()
    assert classifier.model is not None
    assert classifier.vectorizer is not None
    assert classifier.label_encoder is not None


def test_greeting_is_classified_correctly():
    classifier = IntentClassifier()
    prediction = classifier.predict("hello there")
    assert prediction.intent == "greeting"
    assert prediction.confidence > 0.3


def test_goodbye_is_classified_correctly():
    classifier = IntentClassifier()
    prediction = classifier.predict("bye, see you later")
    assert prediction.intent == "goodbye"


def test_thanks_is_classified_correctly():
    classifier = IntentClassifier()
    prediction = classifier.predict("thank you so much")
    assert prediction.intent == "thanks"


def test_empty_input_returns_unknown_with_zero_confidence():
    classifier = IntentClassifier()
    prediction = classifier.predict("")
    assert prediction.intent == "unknown"
    assert prediction.confidence == 0.0


def test_confidence_is_between_zero_and_one():
    classifier = IntentClassifier()
    prediction = classifier.predict("what is machine learning")
    assert 0.0 <= prediction.confidence <= 1.0
