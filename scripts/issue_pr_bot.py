#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]

PRIORITY_LABELS = tuple(
    item.strip()
    for item in os.getenv("BOT_PRIORITY_LABELS", "p:1,p:2,p:3").split(",")
    if item.strip()
)
READY_LABEL = os.getenv("BOT_READY_LABEL", "status:ready").strip()

PLAN_LABEL = os.getenv("BOT_PLAN_LABEL", "bot:planned").strip()
PR_OPEN_LABEL = os.getenv("BOT_PR_OPEN_LABEL", "bot:pr-open").strip()
MANAGED_PR_LABEL = os.getenv("BOT_MANAGED_PR_LABEL", "bot:managed").strip()
ERROR_LABEL = os.getenv("BOT_ERROR_LABEL", "bot:error").strip()

BASE_BRANCH = os.getenv("BOT_BASE_BRANCH", "main").strip()
MAX_ISSUES_PER_RUN = int(os.getenv("BOT_MAX_ISSUES_PER_RUN", "3"))
MAX_FEEDBACK_PER_PR = int(os.getenv("BOT_MAX_FEEDBACK_PER_PR", "1"))
BOT_MAX_PATCH_RETRIES = int(os.getenv("BOT_MAX_PATCH_RETRIES", "2"))

DEFAULT_TEST_COMMAND = (
    "python -m compileall app.py catalog_ingestion.py catalog_service.py "
    "list_search.py list_settings_ui.py list_subsystem.py weather_service.py scripts target_tips"
)
RAW_TEST_COMMANDS = os.getenv("BOT_TEST_COMMANDS", "").strip()
TEST_COMMAND_SOURCE = RAW_TEST_COMMANDS if RAW_TEST_COMMANDS else DEFAULT_TEST_COMMAND
TEST_COMMANDS = [
    item.strip()
    for item in TEST_COMMAND_SOURCE.split(";;")
    if item.strip()
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

PLAN_MARKER = "<!-- bot-plan:v1 -->"
REPLY_MARKER_PREFIX = "<!-- bot-reply:"


@dataclass
class CommandResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class TestRun:
    ok: bool
    results: list[CommandResult]

    def to_markdown(self) -> str:
        if not self.results:
            return "- No test commands configured."

        lines: list[str] = []
        for result in self.results:
            status = "PASS" if result.returncode == 0 else "FAIL"
            icon = "✅" if result.returncode == 0 else "❌"
            lines.append(f"- {icon} `{result.cmd}` ({status})")

            combined = "\n".join(
                part for part in [result.stdout.strip(), result.stderr.strip()] if part
            ).strip()
            if combined:
                compact = "\n".join(combined.splitlines()[:20]).strip()
                lines.append("")
                lines.append("```text")
                lines.append(compact)
                lines.append("```")
        return "\n".join(lines).strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(message: str) -> None:
    print(f"[bot] {message}", flush=True)


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    cwd: Path | None = None,
    capture_output: bool = True,
    text: bool = True,
) -> CommandResult:
    process = subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        check=False,
        capture_output=capture_output,
        text=text,
    )
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    result = CommandResult(
        cmd=" ".join(shlex.quote(item) for item in args),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {result.cmd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_shell(command: str) -> CommandResult:
    process = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        shell=True,
    )
    return CommandResult(
        cmd=command,
        returncode=process.returncode,
        stdout=process.stdout or "",
        stderr=process.stderr or "",
    )


def gh_json(args: list[str]) -> Any:
    result = run_cmd(["gh", *args], check=True)
    text = result.stdout.strip()
    if not text:
        return None
    return json.loads(text)


def gh_output(args: list[str]) -> str:
    return run_cmd(["gh", *args], check=True).stdout.strip()


def label_names(item: dict[str, Any]) -> set[str]:
    labels = item.get("labels", [])
    names: set[str] = set()
    if not isinstance(labels, list):
        return names
    for label in labels:
        if not isinstance(label, dict):
            continue
        name = str(label.get("name", "")).strip()
        if name:
            names.add(name)
    return names


def issue_has_priority(item: dict[str, Any]) -> bool:
    return bool(label_names(item) & set(PRIORITY_LABELS))


def is_bot_login(login: str) -> bool:
    value = str(login or "").strip().lower()
    return value.endswith("[bot]") or value in {
        "github-actions",
        "github-actions[bot]",
    }


def repo_slug() -> str:
    env_repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if env_repo:
        return env_repo
    data = gh_json(["repo", "view", "--json", "nameWithOwner"])
    return str((data or {}).get("nameWithOwner", "")).strip()


def ensure_label(name: str, color: str, description: str) -> None:
    if not name:
        return
    result = run_cmd(
        [
            "gh",
            "label",
            "create",
            name,
            "--color",
            color,
            "--description",
            description,
        ],
        check=False,
    )
    if result.returncode == 0:
        log(f"created label '{name}'")
        return
    stderr = result.stderr.lower()
    if "already exists" in stderr:
        return
    raise RuntimeError(
        f"Unable to ensure label '{name}':\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def ensure_bot_labels() -> None:
    ensure_label(PLAN_LABEL, "1D76DB", "Automated plan posted")
    ensure_label(PR_OPEN_LABEL, "5319E7", "Automated PR has been opened")
    ensure_label(MANAGED_PR_LABEL, "0E8A16", "PR managed by automation bot")
    ensure_label(ERROR_LABEL, "B60205", "Automation encountered an error")


def add_issue_label(issue_number: int, name: str) -> None:
    if not name:
        return
    run_cmd(["gh", "issue", "edit", str(issue_number), "--add-label", name], check=True)


def remove_issue_label(issue_number: int, name: str) -> None:
    if not name:
        return
    run_cmd(["gh", "issue", "edit", str(issue_number), "--remove-label", name], check=False)


def add_pr_label(pr_number: int, name: str) -> None:
    if not name:
        return
    run_cmd(["gh", "pr", "edit", str(pr_number), "--add-label", name], check=True)


def issue_comment(issue_number: int, body: str) -> None:
    run_cmd(["gh", "issue", "comment", str(issue_number), "--body", body], check=True)


def pr_comment(pr_number: int, body: str) -> None:
    run_cmd(["gh", "pr", "comment", str(pr_number), "--body", body], check=True)


def slugify(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    if not slug:
        slug = "work"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "work"


def openai_generate_text(system_prompt: str, user_prompt: str) -> str:
    if not OPENAI_API_KEY:
        return ""

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/responses",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTPError {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    output_text = parsed.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = parsed.get("output", [])
    texts: list[str] = []
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n\n".join(texts).strip()


def fallback_plan(issue: dict[str, Any]) -> str:
    title = str(issue.get("title", "")).strip()
    body = str(issue.get("body", "")).strip()
    has_repro = "repro" in body.lower() or "steps" in body.lower()
    repro_step = (
        "Reproduce the issue from the provided steps and capture current behavior."
        if has_repro
        else "Establish a minimal reproduction path from the issue description."
    )
    return textwrap.dedent(
        f"""
        ### Automated Plan

        1. {repro_step}
        2. Identify the source files and logic paths related to: `{title}`.
        3. Implement a focused fix on a dedicated branch, keeping behavior changes scoped to this issue.
        4. Run standard project validation commands and collect results.
        5. Open a PR in **ready for review** state, attach test results, and link this issue.
        6. Monitor PR comments and push follow-up commits as needed until merge.

        ### Open Questions

        1. Confirm acceptance criteria and any edge cases that must be covered.
        2. Confirm whether behavior changes should be guarded behind any feature flag/settings toggle.
        """
    ).strip()


def generate_issue_plan(issue: dict[str, Any]) -> str:
    if not OPENAI_API_KEY:
        return fallback_plan(issue)

    system_prompt = (
        "You are a pragmatic senior software engineer. "
        "Write a concise, actionable implementation plan in markdown."
    )
    user_prompt = textwrap.dedent(
        f"""
        Repository: {repo_slug()}
        Issue #{issue.get("number")}: {issue.get("title", "")}

        Issue body:
        {issue.get("body", "")}

        Requirements:
        - Produce a concrete plan with numbered steps.
        - Include an explicit test/verification section.
        - Include an 'Open Questions' section if details are missing.
        - Do not include marketing or filler language.
        """
    ).strip()
    try:
        generated = openai_generate_text(system_prompt, user_prompt)
    except Exception as exc:
        log(f"plan generation failed for issue #{issue.get('number')}: {exc}")
        return fallback_plan(issue)
    return generated or fallback_plan(issue)


def extract_diff_block(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```diff\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return str(match.group(1)).strip()
    if text.strip().startswith("diff --git"):
        return text.strip()
    return ""


def generate_issue_patch(issue: dict[str, Any], extra_hint: str = "") -> tuple[str, str]:
    if not OPENAI_API_KEY:
        return "", "OPENAI_API_KEY is not configured."

    files = gh_output(["repo", "view", "--json", "nameWithOwner"]).strip()
    file_list = run_cmd(["git", "ls-files"], check=True).stdout.splitlines()
    file_list_preview = "\n".join(file_list[:240])

    system_prompt = (
        "You are a coding agent. Return ONLY a unified git diff in a ```diff``` fence. "
        "Patch must apply cleanly with git apply. Keep changes minimal and issue-focused."
    )
    user_prompt = textwrap.dedent(
        f"""
        Repo: {files}
        Base branch: {BASE_BRANCH}
        Issue #{issue.get("number")}: {issue.get("title", "")}

        Issue body:
        {issue.get("body", "")}

        Additional hint:
        {extra_hint or "none"}

        Repository files (first 240):
        {file_list_preview}

        Instructions:
        1. Implement the issue end-to-end.
        2. Update tests or add focused validation where reasonable.
        3. Return ONLY one ```diff``` block. No prose outside the fence.
        """
    ).strip()
    try:
        response = openai_generate_text(system_prompt, user_prompt)
    except Exception as exc:
        return "", f"Patch generation failed: {exc}"

    diff = extract_diff_block(response)
    if not diff:
        return "", "No diff block produced by the model."
    return diff, ""


def generate_feedback_patch(pr: dict[str, Any], feedback_text: str, marker_id: str) -> tuple[str, str]:
    if not OPENAI_API_KEY:
        return "", "OPENAI_API_KEY is not configured."

    changed_files = gh_output(
        ["pr", "view", str(pr["number"]), "--json", "files", "--jq", ".files[].path"]
    ).splitlines()
    file_preview = "\n".join(changed_files[:120])
    diff_preview = gh_output(["pr", "diff", str(pr["number"])]).splitlines()
    diff_compact = "\n".join(diff_preview[:400])

    system_prompt = (
        "You are a coding agent handling PR feedback. "
        "Return ONLY a unified git diff in a ```diff``` fence that addresses the feedback."
    )
    user_prompt = textwrap.dedent(
        f"""
        PR #{pr["number"]}: {pr.get("title", "")}
        Feedback marker: {marker_id}

        Feedback:
        {feedback_text}

        Changed files:
        {file_preview}

        Current diff excerpt:
        {diff_compact}

        Requirements:
        - Make only requested/related updates.
        - Preserve existing behavior outside requested changes.
        - Return ONLY one ```diff``` block.
        """
    ).strip()
    try:
        response = openai_generate_text(system_prompt, user_prompt)
    except Exception as exc:
        return "", f"Feedback patch generation failed: {exc}"

    diff = extract_diff_block(response)
    if not diff:
        return "", "No diff block produced for feedback."
    return diff, ""


def apply_diff(diff_text: str) -> tuple[bool, str]:
    if not diff_text.strip():
        return False, "Empty diff."
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".diff") as handle:
        handle.write(diff_text)
        patch_path = handle.name
    try:
        first = run_cmd(
            ["git", "apply", "--index", "--whitespace=fix", patch_path],
            check=False,
        )
        if first.returncode == 0:
            return True, ""
        second = run_cmd(
            ["git", "apply", "--index", "--reject", "--whitespace=fix", patch_path],
            check=False,
        )
        if second.returncode == 0:
            return True, ""
        error_text = "\n".join(
            [
                "git apply failed.",
                f"Attempt 1 stdout:\n{first.stdout}",
                f"Attempt 1 stderr:\n{first.stderr}",
                f"Attempt 2 stdout:\n{second.stdout}",
                f"Attempt 2 stderr:\n{second.stderr}",
            ]
        ).strip()
        return False, error_text
    finally:
        try:
            os.remove(patch_path)
        except OSError:
            pass


def run_tests() -> TestRun:
    results: list[CommandResult] = []
    for command in TEST_COMMANDS:
        result = run_shell(command)
        results.append(result)
    ok = all(result.returncode == 0 for result in results)
    return TestRun(ok=ok, results=results)


def has_uncommitted_changes() -> bool:
    out = run_cmd(["git", "status", "--porcelain"], check=True).stdout.strip()
    return bool(out)


def checkout_branch_from_base(branch_name: str) -> None:
    run_cmd(["git", "fetch", "origin", BASE_BRANCH], check=True)
    run_cmd(["git", "checkout", "-B", branch_name, f"origin/{BASE_BRANCH}"], check=True)


def checkout_pr_head(head_ref: str) -> None:
    run_cmd(["git", "fetch", "origin", head_ref], check=True)
    run_cmd(["git", "checkout", "-B", head_ref, f"origin/{head_ref}"], check=True)


def commit_all(message: str) -> None:
    run_cmd(["git", "add", "-A"], check=True)
    run_cmd(["git", "commit", "-m", message], check=True)


def push_branch(branch_name: str) -> None:
    run_cmd(["git", "push", "-u", "origin", branch_name], check=True)


def current_head_sha() -> str:
    return run_cmd(["git", "rev-parse", "HEAD"], check=True).stdout.strip()


def clear_error_label(issue_number: int) -> None:
    remove_issue_label(issue_number, ERROR_LABEL)


def format_implementation_comment(
    *,
    issue_number: int,
    pr_url: str,
    branch_name: str,
    test_run: TestRun,
) -> str:
    return textwrap.dedent(
        f"""
        Implemented in branch `{branch_name}` and opened PR: {pr_url}

        ### Test Results
        {test_run.to_markdown()}

        _Issue #{issue_number} is now waiting on PR review/approval before merge._
        """
    ).strip()


def format_failure_comment(reason: str) -> str:
    compact = reason.strip() or "unknown error"
    return textwrap.dedent(
        f"""
        Automation attempted implementation but could not complete it.

        ### Failure
        ```text
        {compact}
        ```

        Please add guidance or adjust issue details and keep `status:ready` for retry.
        """
    ).strip()


def list_open_priority_issues() -> list[dict[str, Any]]:
    payload = gh_json(
        [
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "number,title,body,labels,url,updatedAt",
        ]
    )
    issues = payload if isinstance(payload, list) else []
    filtered: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if not issue_has_priority(issue):
            continue
        filtered.append(issue)
    filtered.sort(key=lambda item: int(item.get("number", 0)))
    return filtered


def post_plan_if_needed(issue: dict[str, Any]) -> bool:
    number = int(issue["number"])
    labels = label_names(issue)
    if PLAN_LABEL in labels:
        return False

    plan_body = generate_issue_plan(issue)
    comment_body = f"{plan_body}\n\n{PLAN_MARKER}"
    issue_comment(number, comment_body)
    add_issue_label(number, PLAN_LABEL)
    log(f"posted plan for issue #{number}")
    return True


def find_open_pr_for_branch(branch_name: str) -> dict[str, Any] | None:
    payload = gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--head",
            branch_name,
            "--json",
            "number,url,headRefName",
        ]
    )
    prs = payload if isinstance(payload, list) else []
    if not prs:
        return None
    pr = prs[0]
    return pr if isinstance(pr, dict) else None


def create_pr_for_issue(issue: dict[str, Any], branch_name: str, test_run: TestRun) -> tuple[int, str]:
    number = int(issue["number"])
    title = f"Fix #{number}: {str(issue.get('title', '')).strip()}"
    body = textwrap.dedent(
        f"""
        Automated implementation for #{number}.

        ### Summary
        - Implemented issue requirements on branch `{branch_name}`.
        - Included validation results below.

        ### Test Results
        {test_run.to_markdown()}

        Closes #{number}.
        """
    ).strip()
    pr_url = gh_output(
        [
            "pr",
            "create",
            "--base",
            BASE_BRANCH,
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
        ]
    )
    pr_number_text = gh_output(
        ["pr", "view", branch_name, "--json", "number", "--jq", ".number"]
    )
    pr_number = int(str(pr_number_text).strip())
    add_pr_label(pr_number, MANAGED_PR_LABEL)
    return pr_number, pr_url


def implement_ready_issue(issue: dict[str, Any]) -> None:
    number = int(issue["number"])
    labels = label_names(issue)
    if READY_LABEL not in labels:
        return
    if PR_OPEN_LABEL in labels:
        return

    branch_name = f"issue/{number}-{slugify(issue.get('title', ''))}"
    existing_pr = find_open_pr_for_branch(branch_name)
    if existing_pr is not None:
        add_issue_label(number, PR_OPEN_LABEL)
        log(f"issue #{number} already has open PR {existing_pr.get('url')}")
        return

    try:
        checkout_branch_from_base(branch_name)
    except Exception as exc:
        add_issue_label(number, ERROR_LABEL)
        issue_comment(number, format_failure_comment(f"Failed to prepare branch: {exc}"))
        return

    patch_error = ""
    applied = False
    for attempt in range(1, BOT_MAX_PATCH_RETRIES + 1):
        diff_text, error_text = generate_issue_patch(issue, extra_hint=patch_error)
        if error_text:
            patch_error = error_text
            continue
        ok, apply_error = apply_diff(diff_text)
        if ok:
            applied = True
            break
        patch_error = apply_error

    if not applied:
        add_issue_label(number, ERROR_LABEL)
        issue_comment(number, format_failure_comment(f"Unable to apply patch: {patch_error}"))
        return

    if not has_uncommitted_changes():
        add_issue_label(number, ERROR_LABEL)
        issue_comment(number, format_failure_comment("Patch applied but no working-tree changes were detected."))
        return

    tests = run_tests()
    commit_message = f"fix: resolve #{number} {str(issue.get('title', '')).strip()[:72]}"
    try:
        commit_all(commit_message)
        push_branch(branch_name)
        pr_number, pr_url = create_pr_for_issue(issue, branch_name, tests)
    except Exception as exc:
        add_issue_label(number, ERROR_LABEL)
        issue_comment(number, format_failure_comment(f"Failed to push/create PR: {exc}"))
        return

    add_issue_label(number, PR_OPEN_LABEL)
    clear_error_label(number)
    issue_comment(
        number,
        format_implementation_comment(
            issue_number=number,
            pr_url=pr_url,
            branch_name=branch_name,
            test_run=tests,
        ),
    )
    log(f"implemented issue #{number}, opened PR #{pr_number}")


def load_issue_comments(repo: str, pr_number: int) -> list[dict[str, Any]]:
    payload = gh_json(
        [
            "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "--paginate",
        ]
    )
    if isinstance(payload, list):
        return payload
    return []


def load_review_comments(repo: str, pr_number: int) -> list[dict[str, Any]]:
    payload = gh_json(
        [
            "api",
            f"repos/{repo}/pulls/{pr_number}/comments",
            "--paginate",
        ]
    )
    if isinstance(payload, list):
        return payload
    return []


def load_reviews(repo: str, pr_number: int) -> list[dict[str, Any]]:
    payload = gh_json(
        [
            "api",
            f"repos/{repo}/pulls/{pr_number}/reviews",
            "--paginate",
        ]
    )
    if isinstance(payload, list):
        return payload
    return []


def known_reply_markers(issue_comments: list[dict[str, Any]]) -> set[str]:
    markers: set[str] = set()
    pattern = re.compile(r"<!--\s*bot-reply:([a-z0-9:_-]+)\s*-->")
    for comment in issue_comments:
        if not isinstance(comment, dict):
            continue
        body = str(comment.get("body", ""))
        for marker in pattern.findall(body):
            markers.add(marker.strip())
    return markers


def collect_unhandled_feedback(
    *,
    repo: str,
    pr_number: int,
) -> list[tuple[str, str, str]]:
    issue_comments = load_issue_comments(repo, pr_number)
    review_comments = load_review_comments(repo, pr_number)
    reviews = load_reviews(repo, pr_number)
    seen_markers = known_reply_markers(issue_comments)

    events: list[tuple[str, str, str, str]] = []

    for item in issue_comments:
        if not isinstance(item, dict):
            continue
        marker_id = f"issue-comment-{item.get('id')}"
        if marker_id in seen_markers:
            continue
        user = item.get("user", {})
        login = str((user or {}).get("login", ""))
        if is_bot_login(login):
            continue
        body = str(item.get("body", "")).strip()
        if not body:
            continue
        created = str(item.get("created_at", ""))
        events.append((created, marker_id, body, "issue comment"))

    for item in review_comments:
        if not isinstance(item, dict):
            continue
        marker_id = f"review-comment-{item.get('id')}"
        if marker_id in seen_markers:
            continue
        user = item.get("user", {})
        login = str((user or {}).get("login", ""))
        if is_bot_login(login):
            continue
        body = str(item.get("body", "")).strip()
        if not body:
            continue
        created = str(item.get("created_at", ""))
        events.append((created, marker_id, body, "review comment"))

    for item in reviews:
        if not isinstance(item, dict):
            continue
        marker_id = f"review-{item.get('id')}"
        if marker_id in seen_markers:
            continue
        user = item.get("user", {})
        login = str((user or {}).get("login", ""))
        if is_bot_login(login):
            continue
        body = str(item.get("body", "")).strip()
        if not body:
            continue
        created = str(item.get("submitted_at", "") or item.get("created_at", ""))
        events.append((created, marker_id, body, "review"))

    events.sort(key=lambda item: item[0])
    return [(marker_id, body, source) for _, marker_id, body, source in events]


def list_managed_open_prs() -> list[dict[str, Any]]:
    payload = gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--label",
            MANAGED_PR_LABEL,
            "--limit",
            "100",
            "--json",
            "number,title,body,headRefName,url",
        ]
    )
    prs = payload if isinstance(payload, list) else []
    return [pr for pr in prs if isinstance(pr, dict)]


def handle_feedback_for_pr(repo: str, pr: dict[str, Any]) -> None:
    pr_number = int(pr["number"])
    feedback_items = collect_unhandled_feedback(repo=repo, pr_number=pr_number)
    if not feedback_items:
        return

    head_ref = str(pr.get("headRefName", "")).strip()
    if not head_ref:
        return

    processed = 0
    for marker_id, feedback_text, source in feedback_items:
        if processed >= MAX_FEEDBACK_PER_PR:
            break

        try:
            checkout_pr_head(head_ref)
        except Exception as exc:
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    {REPLY_MARKER_PREFIX}{marker_id} -->
                    Could not check out PR branch for feedback processing.

                    ```text
                    {exc}
                    ```
                    """
                ).strip(),
            )
            processed += 1
            continue

        diff_text, error_text = generate_feedback_patch(pr, feedback_text, marker_id)
        if error_text:
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    {REPLY_MARKER_PREFIX}{marker_id} -->
                    Acknowledged {source}; unable to generate patch automatically.

                    ```text
                    {error_text}
                    ```
                    """
                ).strip(),
            )
            processed += 1
            continue

        ok, apply_error = apply_diff(diff_text)
        if not ok:
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    {REPLY_MARKER_PREFIX}{marker_id} -->
                    Acknowledged {source}; patch did not apply cleanly.

                    ```text
                    {apply_error}
                    ```
                    """
                ).strip(),
            )
            processed += 1
            continue

        if not has_uncommitted_changes():
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    {REPLY_MARKER_PREFIX}{marker_id} -->
                    Acknowledged {source}; no repository changes were required after analysis.
                    """
                ).strip(),
            )
            processed += 1
            continue

        tests = run_tests()
        commit_message = f"fix: address PR #{pr_number} feedback ({marker_id})"
        try:
            commit_all(commit_message)
            sha = current_head_sha()
            push_branch(head_ref)
        except Exception as exc:
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    {REPLY_MARKER_PREFIX}{marker_id} -->
                    Acknowledged {source}; failed while committing/pushing updates.

                    ```text
                    {exc}
                    ```
                    """
                ).strip(),
            )
            processed += 1
            continue

        pr_comment(
            pr_number,
            textwrap.dedent(
                f"""
                {REPLY_MARKER_PREFIX}{marker_id} -->
                Addressed {source} with commit `{sha[:12]}`.

                ### Test Results
                {tests.to_markdown()}
                """
            ).strip(),
        )
        processed += 1

    if processed > 0:
        log(f"processed {processed} feedback item(s) for PR #{pr_number}")


def process_issues(repo: str) -> None:
    issues = list_open_priority_issues()
    if not issues:
        log("no open priority issues")
        return

    processed = 0
    for issue in issues:
        if processed >= MAX_ISSUES_PER_RUN:
            break
        number = int(issue["number"])
        log(f"processing issue #{number}")
        try:
            post_plan_if_needed(issue)
            implement_ready_issue(issue)
        except Exception as exc:
            add_issue_label(number, ERROR_LABEL)
            issue_comment(number, format_failure_comment(str(exc)))
        processed += 1


def process_pr_feedback(repo: str) -> None:
    prs = list_managed_open_prs()
    if not prs:
        log("no managed open PRs")
        return
    for pr in prs:
        pr_number = int(pr["number"])
        log(f"monitoring PR #{pr_number}")
        try:
            handle_feedback_for_pr(repo, pr)
        except Exception as exc:
            pr_comment(
                pr_number,
                textwrap.dedent(
                    f"""
                    Automation monitor hit an unexpected error.

                    ```text
                    {exc}
                    ```
                    """
                ).strip(),
            )


def main() -> None:
    os.chdir(str(REPO_ROOT))
    log(f"run started at {utc_now_iso()}")
    repo = repo_slug()
    if not repo:
        raise RuntimeError("Unable to determine repository slug.")
    ensure_bot_labels()
    process_issues(repo)
    process_pr_feedback(repo)
    log("run completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[bot] fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
