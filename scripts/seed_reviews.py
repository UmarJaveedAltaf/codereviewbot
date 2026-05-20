#!/usr/bin/env python3
"""Seed 50 realistic fake past-review entries into ChromaDB for demo purposes.

Usage
-----
Append 50 reviews (safe to re-run; uuids are random so no de-duplication):
    python scripts/seed_reviews.py

Clear *all* existing past_reviews, then seed fresh:
    python scripts/seed_reviews.py --reset

Distribution
------------
  Severity   CRITICAL  10 %  (5 reviews)
             WARNING   20 %  (10 reviews)
             SUGGESTION 70 %  (35 reviews)

  Accepted   70 % accepted (1)   30 % rejected (0)
             (no unrated entries — keeps acceptance-rate chart meaningful)

  Repos      ~17 each across three demo repositories
  Timestamps Random spread over the last 30 days
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Make project root importable when run directly ────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings as app_settings
from memory import vector_store as vs
from memory.vector_store import ReviewMetadata, add_review

# ── Reproducibility ────────────────────────────────────────────────────────────
_RNG = random.Random(42)

# ── Demo repositories & their files / PRs ────────────────────────────────────

REPOS = [
    "UmarJaveedAltaf/codereviewbot",
    "UmarJaveedAltaf/medease",
    "UmarJaveedAltaf/cardcraft",
]

_REPO_FILES = {
    "UmarJaveedAltaf/codereviewbot": [
        ("agent/reviewer.py",             10),
        ("memory/vector_store.py",        25),
        ("webhook/handler.py",            18),
        ("integrations/github_client.py", 42),
        ("parser/ast_parser.py",          67),
    ],
    "UmarJaveedAltaf/medease": [
        ("api/patients.py",           33),
        ("api/auth.py",               15),
        ("models/prescription.py",    80),
        ("services/billing.py",       54),
        ("utils/validators.py",       22),
    ],
    "UmarJaveedAltaf/cardcraft": [
        ("card/generator.py",         11),
        ("api/payments.py",           29),
        ("models/user.py",            47),
        ("utils/image_processor.py",  63),
        ("services/email_sender.py",  38),
    ],
}

_REPO_PRS = {
    "UmarJaveedAltaf/codereviewbot": [
        (1, "Add structured output for LLM reviews"),
        (2, "Fix memory leak in ChromaDB vector store"),
        (3, "Add GitHub Actions CI pipeline"),
        (4, "Refactor review agent prompt templates"),
        (5, "Improve HMAC webhook signature validation"),
    ],
    "UmarJaveedAltaf/medease": [
        (10, "Add patient data encryption at rest"),
        (11, "Fix prescription dosage calculation bug"),
        (12, "Implement two-factor authentication"),
        (13, "Add billing invoice PDF generation"),
        (14, "Fix appointment scheduling race condition"),
    ],
    "UmarJaveedAltaf/cardcraft": [
        (20, "Integrate Stripe payment gateway"),
        (21, "Implement card template SVG renderer"),
        (22, "Fix user account deletion cascade"),
        (23, "Add image compression pipeline"),
        (24, "Refactor email notification service"),
    ],
}

# ── Review template pools ──────────────────────────────────────────────────────
# Each entry: (code_snippet, review_comment, category)
# Severity is applied at build-time based on which pool an entry comes from.

_CRITICAL: list[tuple[str, str, str]] = [
    (
        'query = f"SELECT * FROM patients WHERE id = {patient_id}"',
        "SQL injection: `patient_id` is interpolated directly into the query string. "
        "An attacker can inject `1 OR 1=1` to exfiltrate the entire table. "
        "Use parameterised queries: `cursor.execute('... WHERE id = %s', (patient_id,))`.",
        "security",
    ),
    (
        'DB_PASSWORD = "prod_admin_2024"\nDB_HOST = "10.0.1.42"',
        "Hardcoded production credentials committed to source. Rotate these immediately "
        "and load via `os.environ['DB_PASSWORD']` or a secrets manager. Git history "
        "will retain them even after deletion — consider a secret-scanning pre-commit hook.",
        "security",
    ),
    (
        'result = eval(user_query)\nreturn jsonify(result)',
        "`eval()` on user-supplied input is a remote-code-execution vulnerability. "
        "An attacker can pass `__import__('subprocess').check_output(['cat','/etc/passwd'])`. "
        "Replace with `ast.literal_eval()` for data literals or a proper expression parser.",
        "security",
    ),
    (
        'user_obj = pickle.loads(base64.b64decode(cookie_data))',
        "Deserialising pickle from a user-controlled cookie allows arbitrary code "
        "execution — CVSS 10.0. Switch to a signed JWT or `itsdangerous.URLSafeTimedSerializer` "
        "for session data instead.",
        "security",
    ),
    (
        'os.system(f"convert {filename} -resize 800x600 output.jpg")',
        "Command injection: `filename` is passed unsanitised to a shell command. "
        "An attacker can supply `'; rm -rf /; '` as the filename. "
        "Use `subprocess.run([\"convert\", filename, ...], check=True)` — the list "
        "form never invokes a shell.",
        "security",
    ),
    (
        'url = request.args.get("redirect")\nreturn redirect(url)',
        "Open redirect: the `redirect` parameter is user-controlled and unvalidated. "
        "An attacker crafts a phishing URL on your domain that forwards to a malicious "
        "site. Validate against an allowlist or enforce a relative-path-only policy.",
        "security",
    ),
    (
        'token = request.headers.get("X-Auth-Token")\nif token == SECRET_TOKEN:\n    return admin_panel()',
        "Timing attack: `==` short-circuits on the first differing character, leaking "
        "information about the correct token via response latency. Use "
        "`hmac.compare_digest(token, SECRET_TOKEN)` for constant-time comparison.",
        "security",
    ),
    (
        'STRIPE_SECRET_KEY = ""sk_test_FAKE_KEY_FOR_DEMO_ONLY""',
        "Live Stripe secret key hardcoded in source. This grants full API access. "
        "Revoke it immediately in the Stripe dashboard, rotate, and store only in "
        "environment variables or a secrets vault — never in code.",
        "security",
    ),
]

_WARNING: list[tuple[str, str, str]] = [
    (
        'def apply_discount(price, discount_pct):\n    return price * (1 - discount_pct / 100)',
        "No guard for `discount_pct > 100`: this produces a negative price, which "
        "could result in negative charges. Add validation: "
        "`if not (0 <= discount_pct <= 100): raise ValueError(f'Invalid discount: {discount_pct}')`.",
        "bug",
    ),
    (
        'for card in user_cards:\n    user_cards.remove(card)',
        "Mutating a list while iterating over it skips every other element — Python "
        "advances the index after each removal. Use a comprehension instead: "
        "`user_cards = [c for c in user_cards if not should_remove(c)]`.",
        "bug",
    ),
    (
        'patient = db.get_patient(patient_id)\nreturn patient["name"], patient["dob"]',
        "`db.get_patient()` returns `None` when the ID is not found; indexing `None` "
        "raises `TypeError` and leaks a 500 to the client. Add: "
        "`if patient is None: raise NotFoundError(f'Patient {patient_id} not found')`.",
        "bug",
    ),
    (
        'def add_tag(tag, existing_tags=[]):\n    existing_tags.append(tag)\n    return existing_tags',
        "Mutable default argument: `existing_tags=[]` is shared across all calls. "
        "Tags accumulate across unrelated requests in a long-running server. "
        "Fix: `existing_tags=None` then `if existing_tags is None: existing_tags = []`.",
        "bug",
    ),
    (
        'resp = requests.get(f"{API_BASE}/v1/cards")\ndata = resp.json()',
        "No timeout set on the HTTP request. A slow upstream will block this thread "
        "indefinitely. Add `timeout=(3.05, 30)` and call `resp.raise_for_status()` "
        "before parsing JSON.",
        "bug",
    ),
    (
        'f = open("invoice_template.html")\nhtml = f.read()\nreturn render(html)',
        "File descriptor is never closed. If `render()` raises, the handle leaks. "
        "Use a context manager: `with open('invoice_template.html') as f: html = f.read()`.",
        "bug",
    ),
    (
        'if user.role == None:\n    assign_default_role(user)',
        "Identity comparison to `None` should use `is`, not `==`. A custom class "
        "could override `__eq__` to return `True` unexpectedly. "
        "Change to `if user.role is None:`.",
        "bug",
    ),
    (
        'thread = Thread(target=process_payment)\nthread.start()\nupdate_ledger(payment_id)',
        "Race condition: `update_ledger()` is called without waiting for the payment "
        "thread to finish. If the thread fails, the ledger records an unconfirmed "
        "payment. Call `thread.join()` and check for exceptions before updating.",
        "bug",
    ),
    (
        'password = hashlib.md5(password.encode()).hexdigest()\ndb.save_user(username, password)',
        "MD5 is cryptographically broken for password storage — GPU rainbow tables "
        "crack it in seconds. Replace with `bcrypt.hashpw(password.encode(), "
        "bcrypt.gensalt(rounds=12))` or `argon2-cffi`.",
        "security",
    ),
    (
        'SECRET = os.environ.get("APP_SECRET", "change_me_in_production")\napp.secret_key = SECRET',
        "Insecure default secret: if `APP_SECRET` is unset in production, the app "
        "silently uses a known fallback, making session tokens forgeable. "
        "Use `os.environ['APP_SECRET']` (no default) to fail fast on misconfiguration.",
        "security",
    ),
    (
        'cache = {}\ndef get_prescription(pid):\n    if pid not in cache:\n        cache[pid] = db.fetch(pid)\n    return cache[pid]',
        "Unbounded in-memory cache grows indefinitely with no eviction policy. "
        "Under sustained load this will exhaust heap memory. Replace with "
        "`@functools.lru_cache(maxsize=1024)` or `cachetools.TTLCache(1024, ttl=300)`.",
        "performance",
    ),
    (
        'for record in db.query("SELECT * FROM transactions").fetchall():\n    process_transaction(record)',
        "`fetchall()` loads the entire result set into memory. For large tables this "
        "exhausts RAM. Use server-side pagination: "
        "`db.query(Transaction).yield_per(500)` or `cursor.fetchmany(500)` in a loop.",
        "performance",
    ),
    (
        'amount_cents = int(amount_dollars * 100)\ncharge_card(amount_cents)',
        "Floating-point arithmetic for currency: `0.1 + 0.2 == 0.30000000000000004`. "
        "Use `decimal.Decimal` for all financial calculations: "
        "`amount_cents = int(Decimal(str(amount_dollars)) * 100)`.",
        "bug",
    ),
    (
        'user_input = request.form["search"]\nresults = es.search(query=user_input)',
        "No length or character validation before passing to Elasticsearch. An attacker "
        "can send a deeply-nested wildcard query causing query amplification. "
        "Validate and cap length: `if len(user_input) > 200: abort(400)`.",
        "security",
    ),
    (
        '@app.route("/admin")\ndef admin_dashboard():\n    return render_template("admin.html")',
        "Admin route has no authentication check — any user can access `/admin` "
        "directly. Add `@login_required` and `if not current_user.is_admin: abort(403)` "
        "before rendering the template.",
        "security",
    ),
]

_SUGGESTION: list[tuple[str, str, str]] = [
    (
        'def get_appointments(doctor_id):\n    result = []\n    for slot in db.get_slots(doctor_id):\n        patient = db.get_patient(slot.patient_id)\n        result.append((slot, patient))\n    return result',
        "N+1 query: `db.get_patient()` runs once per slot. Batch the lookup: "
        "`patient_map = db.get_patients_by_ids([s.patient_id for s in slots])` "
        "then `patient = patient_map[slot.patient_id]`.",
        "performance",
    ),
    (
        'all_records = [transform(r) for r in db.fetch_all_records()]',
        "Materialising all records into a list before processing wastes memory. "
        "If the result is iterated only once, use a generator: "
        "`all_records = (transform(r) for r in db.fetch_all_records())`.",
        "performance",
    ),
    (
        'def calculate_premium(age, risk_score, history, region, coverage):\n    # 80 lines\n    pass',
        "`calculate_premium` has 5 parameters and 80 lines. Extract a `PremiumFactors` "
        "dataclass and split the logic into `_base_rate()`, `_risk_multiplier()`, "
        "`_regional_adjustment()` — each independently testable.",
        "style",
    ),
    (
        'msg = "Hello " + user.first_name + " " + user.last_name + ", your order is ready."',
        "String concatenation should be an f-string: "
        '`msg = f"Hello {user.first_name} {user.last_name}, your order is ready."` '
        "— more readable and ~35% faster for 3+ operands.",
        "style",
    ),
    (
        'PLAN_BASIC = 1\nPLAN_PRO = 2\nPLAN_ENTERPRISE = 3\nif user.plan == 2:\n    unlock_feature()',
        "Magic number `2` in a condition. Use the named constant `PLAN_PRO`, or model "
        "plans as a `StrEnum(BASIC, PRO, ENTERPRISE)` for type-safe comparisons.",
        "style",
    ),
    (
        'result = []\nfor card in deck:\n    if card.suit == "hearts":\n        result.append(card.value)',
        "Replace the append loop with a list comprehension: "
        "`result = [c.value for c in deck if c.suit == 'hearts']`. "
        "More idiomatic and eliminates the per-iteration `result.append` lookup.",
        "style",
    ),
    (
        'import time\ntime.sleep(2)\nif not payment_confirmed():\n    raise PaymentError()',
        "Hard-coded `sleep(2)` is brittle — payment may confirm in 200 ms (wasted) "
        "or take 10 s (still fails). Implement exponential back-off polling: "
        "0.5 s, 1 s, 2 s, 4 s up to a configurable deadline.",
        "performance",
    ),
    (
        'logger.debug("Rendering for user " + str(user_id) + " tmpl=" + template_name)',
        "String concatenation in a `debug()` call is evaluated even when debug logging "
        "is disabled. Use lazy formatting: "
        '`logger.debug("Rendering for user %s tmpl=%s", user_id, template_name)`.',
        "performance",
    ),
    (
        'class CardTemplate:\n    def render(self): ...\n    def validate(self): ...\n    def save_to_db(self): ...\n    def send_preview_email(self): ...',
        "`CardTemplate` mixes rendering, validation, persistence, and notification. "
        "Apply the Single Responsibility Principle: split into `CardRenderer`, "
        "`CardValidator`, `CardRepository`, and let a `CardService` orchestrate them.",
        "style",
    ),
    (
        'with open(template_path) as f:\n    content = f.read()\nreturn jinja_env.from_string(content).render(**ctx)',
        "Reading the whole template into a string is unnecessary. Use Jinja2's "
        "`FileSystemLoader` which handles loading and caching: "
        "`Environment(loader=FileSystemLoader(dir)).get_template(name).render(**ctx)`.",
        "performance",
    ),
    (
        'def get_user_by_email(email):\n    return db.query(User).filter(User.email == email).all()',
        "Missing database index on `users.email`. Without one this is a full table "
        "scan on every call. Add a migration: "
        "`CREATE INDEX idx_users_email ON users(email)` and mark `index=True` in the model.",
        "performance",
    ),
    (
        'def test_charge_card():\n    result = charge_card("4242424242424242", 100)\n    assert result.status == "success"',
        "Test only covers the happy path. Add cases for: declined card "
        "(`PaymentDeclinedError`), expired card, zero/negative amount (`ValueError`), "
        "and network timeout (mock `requests.post` to raise `Timeout`).",
        "test_coverage",
    ),
    (
        'def test_create_patient():\n    patient = create_patient(name="John", dob="1990-01-01")\n    assert patient.id is not None',
        "No test for duplicate-email creation (expect `DuplicatePatientError`), "
        "invalid date format, or missing required fields. Edge cases are the most "
        "likely bug sources and should be tested explicitly.",
        "test_coverage",
    ),
    (
        'def validate_card_number(number: str) -> bool:\n    return len(number) == 16 and number.isdigit()',
        "Missing Luhn checksum validation — `1234567890123456` passes this check but "
        "is not a valid card number. Implement the Luhn algorithm or use `luhn-python` "
        "to validate the check digit.",
        "bug",
    ),
    (
        'except Exception:\n    pass',
        "Silent `pass` in an exception handler hides bugs and makes production "
        "debugging impossible. At minimum: `logger.exception('Unexpected error in %s', __name__)`. "
        "If silencing is intentional, add a comment explaining why.",
        "bug",
    ),
    (
        'if user.email.lower() == input_email.lower():\n    return True',
        "`.lower()` is called twice per comparison. Normalise emails to lowercase at "
        "storage time and use `.casefold()` at lookup (handles Unicode edge cases "
        "like German `ss` / `ß`).",
        "style",
    ),
    (
        'def generate_invoice_number():\n    return str(random.randint(100000, 999999))',
        "`random.randint` is seeded from the system clock and its output is predictable. "
        "For IDs use `secrets.randbelow(900_000) + 100_000` or `str(uuid.uuid4())` "
        "for unpredictable, globally unique identifiers.",
        "security",
    ),
    (
        'config = yaml.load(config_file)',
        "`yaml.load()` without an explicit `Loader` can execute arbitrary Python via "
        "`!!python/object/apply:` tags in the YAML. Always use `yaml.safe_load()` "
        "which only deserialises standard YAML types.",
        "security",
    ),
    (
        'MAX_FILE_SIZE = 52428800  # 50MB\nif file.size > MAX_FILE_SIZE:\n    abort(413)',
        "Magic number `52428800` is hard to read and verify. Express intent directly: "
        "`MAX_FILE_SIZE = 50 * 1024 * 1024`. Python evaluates this at compile time — "
        "zero runtime cost, maximum readability.",
        "style",
    ),
    (
        'def __init__(self, name, email, role, created_at, last_login, preferences):\n    ...',
        "Constructor has 6 parameters — a sign of growing responsibilities. Group "
        "related fields into value objects: `UserProfile(name, email)`, "
        "`UserAccess(role, created_at, last_login)`, `UserSettings(preferences)`.",
        "style",
    ),
    (
        'results = sorted(db.all_users(), key=lambda u: u.created_at, reverse=True)[:10]',
        "Sorting all users in Python to find the 10 most recent is O(N log N) and "
        "loads every row into memory. Push the work to the database: "
        "`db.query(User).order_by(User.created_at.desc()).limit(10).all()`.",
        "performance",
    ),
    (
        'def test_send_email():\n    send_email("test@example.com", "Hello", "Body")\n    # no assertion',
        "Test makes no assertion — if the function raises no exception it passes, "
        "regardless of whether an email was sent. Mock `smtplib.SMTP` and assert "
        "the mock was called with the expected arguments.",
        "test_coverage",
    ),
    (
        'if len(password) < 8:\n    return False\nif not re.search("[A-Z]", password):\n    return False',
        "Password validation rules are scattered and return `False` with no context. "
        "Encapsulate in a `PasswordPolicy.validate(pw) -> list[str]` that returns all "
        "violated rules at once — the UI can then show a complete checklist.",
        "style",
    ),
    (
        'def get_report(start_date, end_date, filters):\n    pass  # TODO: implement',
        "Public route handler with no implementation silently returns `None`, which "
        "will be serialised as `null` in the JSON response. Raise "
        "`NotImplementedError('get_report not yet implemented')` until it is done.",
        "bug",
    ),
    (
        'data = response.json()\nuser_id = data["user"]["id"]\nname = data["user"]["profile"]["name"]',
        "Deep key access without guards raises `KeyError` if the API response changes "
        "shape. Use `.get()` with defaults or deserialise into a Pydantic model: "
        "`UserResponse.model_validate(data)` so missing fields raise a clear error.",
        "bug",
    ),
]


# ── Distribution constants ────────────────────────────────────────────────────
_N_TOTAL      = 50
_N_CRITICAL   = 5    # 10 %
_N_WARNING    = 10   # 20 %
_N_SUGGESTION = 35   # 70 %

_N_ACCEPTED  = 35   # 70 %
_N_REJECTED  = 15   # 30 %


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_reviews() -> list[dict]:
    """Return a list of 50 review dicts ready for ``add_review()``."""

    # ── Pick templates with controlled severity distribution ──────────────
    picks: list[tuple[str, str, str, str]] = []   # (snippet, comment, category, severity)

    picks += [
        (s, c, cat, "CRITICAL")
        for s, c, cat in _RNG.sample(_CRITICAL, _N_CRITICAL)
    ]
    picks += [
        (s, c, cat, "WARNING")
        for s, c, cat in _RNG.sample(_WARNING, _N_WARNING)
    ]
    picks += [
        (s, c, cat, "SUGGESTION")
        for s, c, cat in _RNG.choices(_SUGGESTION, k=_N_SUGGESTION)
    ]

    _RNG.shuffle(picks)

    # ── Acceptance flags (70 / 30) ─────────────────────────────────────────
    accepted_flags = [1] * _N_ACCEPTED + [0] * _N_REJECTED
    _RNG.shuffle(accepted_flags)

    # ── Timestamps: random over last 30 days, in chronological order ───────
    now = datetime.now(timezone.utc)
    timestamps = sorted(
        now - timedelta(seconds=_RNG.randint(0, 30 * 24 * 3600))
        for _ in range(_N_TOTAL)
    )

    # ── Repos: evenly distributed ──────────────────────────────────────────
    # Build [r0, r1, r2, r0, r1, r2, ...] then take the first 50.
    _cycle = (REPOS * (_N_TOTAL // len(REPOS) + 1))[:_N_TOTAL]
    repos = list(_cycle)
    _RNG.shuffle(repos)

    # ── Assemble final list ────────────────────────────────────────────────
    reviews: list[dict] = []
    for i, (snippet, comment, category, severity) in enumerate(picks):
        repo = repos[i]
        file_name, base_line = _RNG.choice(_REPO_FILES[repo])
        pr_number, pr_title  = _RNG.choice(_REPO_PRS[repo])

        reviews.append({
            "code_snippet": snippet,
            "comment":      comment,
            "metadata": ReviewMetadata(
                file=file_name,
                line=base_line + _RNG.randint(0, 20),
                severity=severity,
                category=category,
                repo=repo,
                timestamp=timestamps[i].isoformat(),
                accepted=accepted_flags[i],
                pr_number=pr_number,
                pr_title=pr_title,
            ),
        })

    return reviews


# ── Reset helper ──────────────────────────────────────────────────────────────

def _reset_reviews() -> None:
    """Delete and re-create the ``past_reviews`` ChromaDB collection."""
    if os.environ.get("CHROMA_HOST"):
        client = chromadb.HttpClient(
            host=os.environ["CHROMA_HOST"],
            port=int(os.environ.get("CHROMA_PORT", "8000")),
        )
    else:
        client = chromadb.PersistentClient(
            path=app_settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    try:
        client.delete_collection("past_reviews")
        print("  Deleted existing past_reviews collection.")
    except Exception:
        print("  No existing past_reviews collection found (skipping delete).")

    vs._reviews_col = None
    print("  Collection reset complete.")


# ── Seed ──────────────────────────────────────────────────────────────────────

def _seed(reviews: list[dict]) -> int:
    """Insert reviews into the vector store. Returns the count seeded."""
    total  = len(reviews)
    seeded = 0

    for i, rev in enumerate(reviews, start=1):
        meta: ReviewMetadata = rev["metadata"]
        rid  = add_review(
            code_snippet=rev["code_snippet"],
            comment=rev["comment"],
            metadata=meta,
        )
        seeded += 1

        sev_icon = {"CRITICAL": "!!", "WARNING": "! ", "SUGGESTION": "  "}.get(
            meta.severity, "  "
        )
        print(
            f"  [{i:02d}/{total}] {sev_icon} {rid[:8]}... "
            f"[{meta.severity:<10}] [{meta.category:<13}] "
            f"{meta.repo.split('/')[1]:<15} {meta.file}"
        )

    return seeded


# ── CLI entry point ───────────────────────────────────────────────────────────

import os  # noqa: E402 — imported here to keep top-of-file imports clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed fake review data into CodeReviewBot's ChromaDB for demo purposes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing past_reviews collection before seeding.",
    )
    args = parser.parse_args()

    print("CodeReviewBot -- review seeder")
    print(f"  Store  : {app_settings.CHROMA_PERSIST_DIR}")
    print(f"  Mode   : {'RESET + SEED' if args.reset else 'APPEND'}")
    print()

    if args.reset:
        print("Resetting collection...")
        _reset_reviews()
        print()

    print(f"Seeding {_N_TOTAL} reviews ({_N_CRITICAL} CRITICAL / "
          f"{_N_WARNING} WARNING / {_N_SUGGESTION} SUGGESTION)...")
    print()

    reviews = _build_reviews()
    count   = _seed(reviews)

    print()
    print(f"Done. {count} review(s) seeded into past_reviews.")
    print()
    print(f"  Acceptance rate : {_N_ACCEPTED}/{_N_TOTAL} = "
          f"{_N_ACCEPTED / _N_TOTAL:.0%} accepted")
    print(f"  Repos           : {', '.join(r.split('/')[1] for r in REPOS)}")
    print(f"  Time range      : last 30 days")


if __name__ == "__main__":
    main()
