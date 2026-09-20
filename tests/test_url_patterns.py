from app import detect_url_phishing


def test_detects_userinfo_phishing_url():
    assert detect_url_phishing("http://google.com@evil.com") == "Phishing"
    assert detect_url_phishing("https://user:pass@example.com/login") == "Phishing"


def test_keeps_normal_urls_safe():
    assert detect_url_phishing("https://www.google.com") == "Safe"
    assert detect_url_phishing("https://example.com/login") == "Safe"
