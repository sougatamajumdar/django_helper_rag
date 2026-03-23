import os, json, re, time
import requests
from bs4 import BeautifulSoup


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "qwen2.5:3b-instruct"  

ROOT = "rag_data"

TARGETS = [
    ("django/basics/models.md", "Django models basics: model class, fields, migrations, Meta, relationships."),
    ("django/basics/views.md", "Django views basics: function-based views, class-based views, request/response, templates, redirect."),
    ("django/basics/urls.md", "Django URL routing: path/re_path, include, converters, namespaces, reversing."),
    ("django/auth/custom_user.md", "Custom user model in Django: AbstractUser vs AbstractBaseUser, settings.AUTH_USER_MODEL, migrations, best practices."),
    ("django/auth/permissions.md", "Django permissions: model permissions, groups, user permissions, decorators/mixins, auth backends."),
    ("drf/serializers/model_serializer.md", "DRF ModelSerializer: fields, create/update, depth, read_only/write_only, nested serializers."),
    ("drf/serializers/validation.md", "DRF validation: field-level validate_<field>, validate(), validators, unique constraints, error formats."),
    ("drf/views/viewsets.md", "DRF ViewSets: ModelViewSet, routers, actions, permissions/auth, filtering/pagination basics."),
    ("drf/views/api_views.md", "DRF APIView and generics: APIView, GenericAPIView, mixins, concrete generic views."),
    ("code_patterns/good_practices.md", "Good practices for Django + DRF projects: structure, settings, security, testing, performance."),
    ("code_patterns/anti_patterns.md", "Common anti-patterns in Django + DRF: fat views, tight coupling, N+1 queries, insecure defaults."),
    ("examples/auth_example.py", "A minimal DRF auth example: token auth or JWT-like flow description; include settings snippets and example requests."),
    ("examples/crud_example.py", "A minimal DRF CRUD example: model, serializer, viewset, router; include curl examples."),
    ("examples/permissions_example.py", "A minimal permissions example: custom permission class + usage on a viewset; include a quick test snippet."),
]

# Seed sources (start small; add more URLs as you like)
SOURCE_URLS = [
    "https://raw.githubusercontent.com/django/django/main/docs/topics/auth/customizing.txt",
    "https://raw.githubusercontent.com/django/django/main/docs/topics/http/urls.txt",
    "https://raw.githubusercontent.com/django/django/main/docs/topics/db/models.txt",
    "https://raw.githubusercontent.com/encode/django-rest-framework/master/docs/api-guide/serializers.md",
    "https://raw.githubusercontent.com/encode/django-rest-framework/master/docs/api-guide/viewsets.md",
    "https://raw.githubusercontent.com/encode/django-rest-framework/master/docs/api-guide/permissions.md",
]


def ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def fetch_text(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def clean_text(s, max_chars=18000):
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s[:max_chars]

def ollama_generate(system, user, temperature=0.2):
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {"temperature": temperature}
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    print("OLLAMA STATUS:", resp.status_code)
    resp.raise_for_status()
    return resp.json()["response"]

def load_metadata():
    path = os.path.join(ROOT, "metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    os.makedirs(ROOT, exist_ok=True)
    metadata = load_metadata()

    # Pull sources
    sources = []
    for u in SOURCE_URLS:
        try:
            txt = fetch_text(u)
            sources.append({"url": u, "text": clean_text(txt)})
            time.sleep(0.2)
        except Exception as e:
            sources.append({"url": u, "text": f"[FAILED TO FETCH: {e}]"})
    sources_block = "\n\n".join(
        [f"### SOURCE: {s['url']}\n{s['text']}" for s in sources]
    )

    system_prompt = (
        "You are a senior technical writer and Django/DRF engineer. "
        "Write accurate, practical notes for a RAG knowledge base. "
        "Do not invent APIs. Prefer official Django/DRF docs from provided sources. "
        "Output must be ONLY the file content, with no surrounding commentary."
    )

    for rel_path, intent in TARGETS:
        out_path = os.path.join(ROOT, rel_path)
        ensure_parent(out_path)

        # Special case: JSON error files
        if rel_path.endswith(".json"):
            user_prompt = f"""
Create a JSON file for: {rel_path}

Intent: {intent}

Requirements:
- Output valid JSON only.
- Include keys: "error_name", "symptoms", "common_causes", "fix_steps", "example_traceback", "related_links".
- "related_links" must include 2-5 URLs from the provided sources if relevant.

Sources:
{sources_block}
"""
        # Python examples
        elif rel_path.endswith(".py"):
            user_prompt = f"""
Create the Python file: {rel_path}

Intent: {intent}

Requirements:
- Output only Python code.
- Keep it runnable (imports, minimal settings placeholders as comments).
- Include brief comments explaining what's happening.
- Use Django/DRF best practices (small, clear).
- Do not require external secret keys.

Sources:
{sources_block}
"""
        # Markdown notes
        else:
            user_prompt = f"""
Create the markdown file: {rel_path}

Intent: {intent}

Structure:
- Title (H1)
- 5–10 short sections with H2 headings
- Include 1–2 small code snippets where helpful
- Include a short "Common pitfalls" section
- End with "References" bullet list (URLs only) using the provided source URLs

Constraints:
- Be concise but correct.
- Only include claims supported by sources; if missing, say "Not covered in provided sources".

Metadata (optional):
{json.dumps(metadata, ensure_ascii=False)[:2000]}

Sources:
{sources_block}
"""

        content = ollama_generate(system_prompt, user_prompt)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

        print("Wrote:", out_path)


main()

