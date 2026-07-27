"""git_read_only@v1 — the read-only-git proof at the approval point
(converged contract c5028 R2; the abstractcode corpus ported so the
client's 330-line shell twin can die).

The refiner may only DOWNGRADE to auto on a PROVEN read-only git command;
every doubt asks. The adversarial corpus rides verbatim: positional write
verbs, write/exec flags on allowed verbs, wrappers, shell operators,
substitution, globals-before-verb.
"""

from abstractruntime.integrations.abstractcore.effect_handlers import (
    _git_read_only_refiner,
)


def _call(command: str, **extra):
    args = {"command": command}
    args.update(extra)
    return {"name": "execute_command", "arguments": args}


class TestProvenReadsAuto:
    def test_plain_read_verbs_auto(self):
        for cmd in (
            "git status",
            "git log --oneline -20",
            "git diff HEAD~1",
            "git show abc123",
            "git ls-files",
            "/usr/bin/git status",
        ):
            assert _git_read_only_refiner(_call(cmd), None) == "auto", cmd

    def test_working_directory_arg_is_known_and_fine(self):
        assert _git_read_only_refiner(
            _call("git status", working_directory="/tmp/repo"), None
        ) == "auto"


class TestCorpusAsks:
    def test_positional_write_verbs_ask(self):
        for cmd in (
            "git remote set-url origin evil",
            "git reflog expire --all",
            "git branch -v newname",
            "git stash drop",
            "git push",
            "git commit -m x",
            "git checkout -- .",
        ):
            assert _git_read_only_refiner(_call(cmd), None) == "ask", cmd

    def test_write_exec_flags_on_allowed_verbs_ask(self):
        for cmd in (
            "git log --output=/tmp/x",
            "git diff --ext-diff",
            "git log -o /tmp/x",
        ):
            assert _git_read_only_refiner(_call(cmd), None) == "ask", cmd

    def test_shell_operators_and_substitution_ask(self):
        for cmd in (
            "git status && rm -rf /",
            "git log | tee /tmp/x",
            "git show `whoami`",
            "git diff $(cat /etc/passwd)",
            "git status; git push",
            "git status > /tmp/out",
        ):
            assert _git_read_only_refiner(_call(cmd), None) == "ask", cmd

    def test_wrappers_and_globals_ask(self):
        for cmd in (
            "env GIT_DIR=/x git status",   # wrapped git is unproven
            "nohup git log",
            "git -C /somewhere/else status",  # global before verb
            "git -c core.pager=evil log",
            "git --git-dir=/x status",
            "git",                          # bare
        ):
            assert _git_read_only_refiner(_call(cmd), None) == "ask", cmd

    def test_malformed_and_differential_shapes_ask(self):
        assert _git_read_only_refiner(_call("git status\ngit push"), None) == "ask"
        assert _git_read_only_refiner(_call('git "unclosed'), None) == "ask"
        assert _git_read_only_refiner(_call(""), None) == "ask"
        # Unknown argument keys = the wrapper-differential guard.
        assert _git_read_only_refiner(
            _call("git status", shell_wrapper="bash -c"), None
        ) == "ask"
        assert _git_read_only_refiner({"name": "execute_command"}, None) == "ask"
        assert _git_read_only_refiner(
            {"name": "execute_command", "arguments": "not-json{"}, None
        ) == "ask"


class TestRegistryAndCarveout:
    def test_refiner_registered_under_its_id(self):
        from abstractruntime.integrations.abstractcore.effect_handlers import (
            _TOOL_REFINERS,
        )

        assert _TOOL_REFINERS.get("git_read_only@v1") is _git_read_only_refiner

    def test_outreach_carveout_declared_on_served_rows(self):
        """Finding 3 (c5028 R4): a static-fold auto on an outreach-band tool
        carries the declared carve-out so facts-trusting clients see WHY
        approval_default disagrees with the rank band."""
        from abstractruntime.integrations.abstractcore.tool_inventory_facade import (
            annotate_tool_rows,
        )

        rows = [
            {"name": "send_telegram_message", "comms_send": True, "mutating": True},
            {"name": "read_file", "mutating": False},
        ]
        annotate_tool_rows(rows)
        telegram = rows[0]
        if telegram.get("approval_default") == "auto" and int(telegram.get("risk_rank") or 0) >= 3:
            assert telegram.get("approval_carveout") == "static-fold-comms-auto-2026-02-21"
        # A plain read never carries the carve-out.
        assert "approval_carveout" not in rows[1]
