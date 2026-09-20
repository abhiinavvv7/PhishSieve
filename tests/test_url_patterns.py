from app import detect_url_phishing


def test_userinfo_phishing_pattern_is_flagged():
    assert detect_url_phishing("http://google.com@evil.com") == "Phishing"


def test_normal_url_stays_safe():
    assert detect_url_phishing("https://google.com") == "Safe"
