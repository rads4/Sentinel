"""
Feature extraction module for phishing detection.
Extracts 16 features from URLs using UCI encoding: {-1, 0, 1}
  -1 = phishing indicator
   0 = suspicious / neutral
   1 = legitimate indicator

All features are extractable from the URL string alone.
DNS/WHOIS features are attempted with a short timeout and fall back
to neutral (0) if unavailable — ensuring reliability during viva and
on Render free tier.
"""

import re
import socket
import ipaddress
from urllib.parse import urlparse

import tldextract as _tldextract_lib
# Use bundled snapshot — avoids network call to publicsuffix.org
# which may be blocked in sandboxed/Render environments
tldextract = _tldextract_lib.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

# ---------------------------------------------------------------------------
# Shortening services
# ---------------------------------------------------------------------------
SHORTENERS = {
    "bit.ly", "goo.gl", "shorte.st", "go2l.ink", "x.url.ph", "ow.ly",
    "t.co", "tinyurl.com", "tr.im", "is.gd", "cli.gs", "yfrog.com",
    "migre.me", "ff.im", "tiny.cc", "url4.eu", "twit.ac", "su.pr",
    "twurl.nl", "snipurl.com", "short.to", "budurl.com", "ping.fm",
    "post.ly", "just.as", "bkite.com", "snipr.com", "fic.kr", "loopt.us",
    "doiop.com", "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly",
    "to.ly", "bit.do", "t2m.io", "cutt.ly", "rb.gy", "shorturl.at",
}

# Suspicious TLDs frequently seen in phishing campaigns
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online",
    "site", "website", "host", "space", "live", "fun", "click",
    "link", "pw", "cc", "su", "ru", "cn",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parsed(url: str):
    """Return (parsed, extracted) tuple. Adds scheme if missing."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    extracted = tldextract(url)
    return parsed, extracted, url


def _is_ip(hostname: str) -> bool:
    """True if hostname is a raw IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _dns_exists(domain: str, timeout: float = 2.0) -> int:
    """
    Returns 1 if DNS A record exists, -1 if not, 0 on timeout/error.
    Uses a short timeout to avoid slowing down the viva demo.
    """
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        return 1
    except socket.gaierror:
        return -1
    except Exception:
        return 0
    finally:
        socket.setdefaulttimeout(old_timeout)


def _domain_age_feature(domain: str, timeout: float = 3.0) -> int:
    """
    Returns 1 if domain age > 12 months (legitimate signal),
    -1 if < 6 months (phishing signal), 0 on error/timeout.
    """
    try:
        import whois as pythonwhois
        import datetime
        import signal as _signal

        def _handler(signum, frame):
            raise TimeoutError()

        _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(int(timeout))
        try:
            w = pythonwhois.whois(domain)
        finally:
            _signal.alarm(0)

        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return 0
        age_days = (datetime.datetime.now() - creation).days
        if age_days > 365:
            return 1
        elif age_days < 180:
            return -1
        return 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Individual feature functions  (all return -1, 0, or 1)
# ---------------------------------------------------------------------------

def feat_having_ip(parsed, extracted, url) -> int:
    host = parsed.hostname or ""
    return -1 if _is_ip(host) else 1


def feat_url_length(parsed, extracted, url) -> int:
    n = len(url)
    if n < 54:
        return 1
    if n <= 75:
        return 0
    return -1


def feat_shortening_service(parsed, extracted, url) -> int:
    domain = (extracted.domain + "." + extracted.suffix).lower()
    return -1 if domain in SHORTENERS else 1


def feat_having_at_symbol(parsed, extracted, url) -> int:
    return -1 if "@" in url else 1


def feat_double_slash_redirect(parsed, extracted, url) -> int:
    # double slash after the scheme counts as a redirect trick
    path = parsed.path or ""
    return -1 if "//" in path else 1


def feat_prefix_suffix(parsed, extracted, url) -> int:
    return -1 if "-" in (extracted.domain or "") else 1


def feat_subdomain_depth(parsed, extracted, url) -> int:
    subdomain = extracted.subdomain or ""
    if not subdomain or subdomain == "www":
        return 1
    parts = [p for p in subdomain.split(".") if p and p != "www"]
    if len(parts) == 1:
        return 0
    return -1


def feat_https_present(parsed, extracted, url) -> int:
    return 1 if parsed.scheme == "https" else -1


def feat_https_in_domain(parsed, extracted, url) -> int:
    domain_str = (extracted.domain or "").lower()
    return -1 if "https" in domain_str or "http" in domain_str else 1


def feat_port_in_url(parsed, extracted, url) -> int:
    if parsed.port and parsed.port not in (80, 443):
        return -1
    return 1


def feat_submitting_to_email(parsed, extracted, url) -> int:
    return -1 if "mailto:" in url.lower() else 1


def feat_abnormal_url(parsed, extracted, url) -> int:
    """
    Checks if the registered domain appears in the URL's netloc.
    A mismatch (e.g. 'paypal' not in 'paypal-security-login.com') is suspicious.
    """
    host = parsed.netloc or ""
    reg_domain = extracted.domain or ""
    if not reg_domain:
        return 0
    return 1 if reg_domain in host else -1


def feat_url_depth(parsed, extracted, url) -> int:
    path = parsed.path or ""
    depth = len([p for p in path.split("/") if p])
    if depth <= 2:
        return 1
    if depth <= 4:
        return 0
    return -1


def feat_suspicious_tld(parsed, extracted, url) -> int:
    tld = (extracted.suffix or "").lower().split(".")[-1]
    return -1 if tld in SUSPICIOUS_TLDS else 1


def feat_dns_record(parsed, extracted, url) -> int:
    domain = extracted.registered_domain or parsed.hostname or ""
    if not domain or _is_ip(domain):
        return 0
    return _dns_exists(domain)


def feat_domain_age(parsed, extracted, url) -> int:
    domain = extracted.registered_domain or ""
    if not domain or _is_ip(domain):
        return 0
    return _domain_age_feature(domain)


# ---------------------------------------------------------------------------
# Feature registry — order must match training
# ---------------------------------------------------------------------------

FEATURE_REGISTRY = [
    ("having_ip_address",      feat_having_ip,            "URL uses an IP address instead of a domain name"),
    ("url_length",             feat_url_length,           "URL character length"),
    ("shortening_service",     feat_shortening_service,   "URL shortening service detected"),
    ("having_at_symbol",       feat_having_at_symbol,     "@ symbol present in URL"),
    ("double_slash_redirect",  feat_double_slash_redirect,"Double-slash redirect in URL path"),
    ("prefix_suffix_hyphen",   feat_prefix_suffix,        "Hyphen (-) in domain name"),
    ("subdomain_depth",        feat_subdomain_depth,      "Number of subdomains"),
    ("https_present",          feat_https_present,        "HTTPS protocol used"),
    ("https_in_domain",        feat_https_in_domain,      "Word 'https' embedded in domain name"),
    ("non_standard_port",      feat_port_in_url,          "Non-standard port in URL"),
    ("submitting_to_email",    feat_submitting_to_email,  "URL submits to email (mailto)"),
    ("abnormal_url",           feat_abnormal_url,         "Domain name mismatch in URL"),
    ("url_depth",              feat_url_depth,            "URL path depth (number of segments)"),
    ("suspicious_tld",         feat_suspicious_tld,       "Suspicious top-level domain"),
    ("dns_record",             feat_dns_record,           "Valid DNS record exists for domain"),
    ("domain_age",             feat_domain_age,           "Domain registration age"),
]

FEATURE_NAMES = [f[0] for f in FEATURE_REGISTRY]
FEATURE_DESCRIPTIONS = {f[0]: f[2] for f in FEATURE_REGISTRY}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract all 16 features from a URL.
    Returns a dict with:
      - 'vector': list of 16 int values in FEATURE_NAMES order
      - 'details': list of dicts with name, value, description, label
      - 'url_normalized': the URL as parsed
      - 'domain': registered domain
    """
    try:
        parsed, extracted, url_norm = _get_parsed(url)
    except Exception as e:
        raise ValueError(f"Could not parse URL: {e}")

    vector = []
    details = []

    for name, fn, desc in FEATURE_REGISTRY:
        try:
            val = fn(parsed, extracted, url_norm)
        except Exception:
            val = 0  # neutral on unexpected error

        vector.append(val)

        if val == 1:
            label = "legitimate"
        elif val == -1:
            label = "phishing"
        else:
            label = "neutral"

        details.append({
            "name": name,
            "value": val,
            "description": desc,
            "label": label,
        })

    return {
        "vector": vector,
        "details": details,
        "url_normalized": url_norm,
        "domain": extracted.registered_domain or parsed.hostname or url,
    }


def get_feature_names() -> list:
    return FEATURE_NAMES


def get_feature_descriptions() -> dict:
    return FEATURE_DESCRIPTIONS
