"""Threat model stage — simple (single-pass) or full (paper-aligned multi-step)."""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from ...models import SSEvent, EventType, ThreatModel
from ...project.state import ProjectState
from ...llm.client import LLMClient
from ...prompts import (
    THREAT_MODELING_SIMPLE_PROMPT,
    TM_EXTRACT_CHAINS_PROMPT,
    TM_GENERATE_SC_AM_PROMPT,
    TM_ANALYZE_CHAIN_PROMPT,
    TM_JUDGE_PROMPT,
)


async def run_threat_model(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
    mode: str = "simple",
    candidates: int = 3,
) -> AsyncGenerator[SSEvent, None]:
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Generating threat model (mode={mode})...")

    if mode == "full":
        async for event in _run_full(prompt, state, llm, candidates):
            yield event
    else:
        async for event in _run_simple(prompt, state, llm):
            yield event


# ── Simple mode ───────────────────────────────────────────────────────────────

async def _run_simple(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
) -> AsyncGenerator[SSEvent, None]:
    arch = state.get_full_arch()
    p = THREAT_MODELING_SIMPLE_PROMPT.format(msg=prompt, prd=state.prd, arch=arch)
    response = await llm.chat([{"role": "user", "content": p}], max_tokens=8192)

    if not response or len(response.strip()) < 100:
        raise ValueError(f"Threat model too short ({len(response)} chars)")

    state.threat_model = ThreatModel(raw_text=response)
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=response[:300] + "...")


# ── Full mode (paper-aligned) ─────────────────────────────────────────────────

async def _run_full(
    prompt: str,
    state: ProjectState,
    llm: LLMClient,
    k: int,
) -> AsyncGenerator[SSEvent, None]:
    arch = state.get_full_arch()

    # Step 1: Extract entry interfaces + invocation chains
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content="Step 1/5: Extracting entry interfaces and invocation chains...")

    p = TM_EXTRACT_CHAINS_PROMPT.format(arch=arch)
    chains_resp = await llm.chat([{"role": "user", "content": p}], max_tokens=4096, json_mode=True)
    chains_data = _extract_json(chains_resp)
    entry_interfaces = chains_data.get("entry_interfaces", [])

    if not entry_interfaces:
        yield SSEvent(type=EventType.WARN, stage="threat_model",
                      content="No entry interfaces found — falling back to simple mode")
        async for e in _run_simple(prompt, state, llm):
            yield e
        return

    iface_summary = "\n".join(
        f"  - {ei.get('interface', '?')} → {ei.get('entry_function', '?')} "
        f"(chain: {len(ei.get('invocation_chain', []))} fns)"
        for ei in entry_interfaces
    )
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Found {len(entry_interfaces)} entry interfaces:\n{iface_summary}")

    # Step 2: Security context + attacker model
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content="Step 2/5: Generating Security Context + Attacker Model...")

    p = TM_GENERATE_SC_AM_PROMPT.format(
        msg=prompt, prd=state.prd, arch=arch,
        entry_interfaces=json.dumps(entry_interfaces, indent=2, ensure_ascii=False),
    )
    sc_am = await llm.chat([{"role": "user", "content": p}], max_tokens=4096)
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"SC + AM generated ({len(sc_am)} chars)")

    # Step 3: Per-chain function-level threat analysis
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Step 3/5: Analyzing {len(entry_interfaces)} chains...")

    global_ft: dict[str, list[dict]] = {}
    for idx, ei in enumerate(entry_interfaces, 1):
        iface_name = ei.get("interface", f"interface_{idx}")
        chain = ei.get("invocation_chain", [])
        if not chain:
            continue

        yield SSEvent(type=EventType.LOG, stage="threat_model",
                      content=f"  Chain {idx}/{len(entry_interfaces)}: {iface_name}")

        existing = {f: global_ft[f] for f in chain if f in global_ft}
        existing_text = json.dumps(existing, indent=2, ensure_ascii=False) if existing else "(none)"

        p = TM_ANALYZE_CHAIN_PROMPT.format(
            sc_am=sc_am,
            interface=iface_name,
            chain=" → ".join(chain),
            arch_excerpt=arch,
            existing_threats=existing_text,
        )
        try:
            resp = await llm.chat([{"role": "user", "content": p}], max_tokens=4096, json_mode=True)
            data = _extract_json(resp)
            for entry in data.get("function_threats", []):
                fn = entry.get("function", "")
                new_threats = entry.get("threats", [])
                if not fn or not new_threats:
                    continue
                if fn not in global_ft:
                    global_ft[fn] = new_threats
                else:
                    existing_cwes = {t.get("cwe", "") for t in global_ft[fn]}
                    for t in new_threats:
                        if t.get("cwe", "") not in existing_cwes:
                            global_ft[fn].append(t)
                            existing_cwes.add(t.get("cwe", ""))
        except Exception as e:
            yield SSEvent(type=EventType.WARN, stage="threat_model",
                          content=f"  Chain {idx} analysis failed: {e}")

    total_threats = sum(len(ts) for ts in global_ft.values())
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Function threats: {len(global_ft)} functions, {total_threats} threats")

    # Step 4: Assemble k candidates
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Step 4/5: Assembling {k} candidates...")

    candidates_text: list[str] = []
    for ci in range(k):
        ft_sections = []
        for fn, threats in global_ft.items():
            threat_lines = [
                f"  - **{t.get('cwe', '?')} {t.get('name', '')}**: "
                f"{t.get('attack_vector', '')}\n"
                f"    Protection: {t.get('protection', '')}"
                for t in threats
            ]
            ft_sections.append(f"#### {fn}\n" + "\n".join(threat_lines))

        candidate = (
            f"# Threat Model (Candidate {ci + 1})\n\n"
            f"## Security Context & Attacker Model\n{sc_am}\n\n"
            f"## Entry Interfaces\n{iface_summary}\n\n"
            f"## Function-level Threats\n\n" + "\n\n".join(ft_sections)
        )
        candidates_text.append(candidate)

        # Re-analyze with temperature variation for diversity (ci > 0)
        if ci < k - 1 and entry_interfaces:
            global_ft = await _reanalyze_variant(
                entry_interfaces, sc_am, arch, global_ft, llm, temp=0.6 + ci * 0.15
            )

    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Generated {len(candidates_text)} candidates")

    # Step 5: LLM judge
    if len(candidates_text) > 1:
        yield SSEvent(type=EventType.LOG, stage="threat_model",
                      content="Step 5/5: LLM-Judge scoring candidates...")
        best_idx = 0
        best_score = -1.0
        arch_brief = state.get_file_index()

        for ci, cand in enumerate(candidates_text):
            p = TM_JUDGE_PROMPT.format(
                msg=prompt, arch_summary=arch_brief, candidate=cand)
            try:
                judge_resp = await llm.chat([{"role": "user", "content": p}], max_tokens=512, json_mode=True)
                scores = _extract_json(judge_resp)
                s = (
                    float(scores.get("relevance", 5)) * 0.33
                    + float(scores.get("impact", 5)) * 0.33
                    + float(scores.get("exploitability", 5)) * 0.34
                )
                yield SSEvent(type=EventType.LOG, stage="threat_model",
                              content=f"  Candidate {ci + 1}: score={s:.2f}")
                if s > best_score:
                    best_score = s
                    best_idx = ci
            except Exception as e:
                yield SSEvent(type=EventType.WARN, stage="threat_model",
                              content=f"  Judge failed for candidate {ci + 1}: {e}")

        yield SSEvent(type=EventType.LOG, stage="threat_model",
                      content=f"Selected candidate {best_idx + 1}")
        final = candidates_text[best_idx]
    else:
        final = candidates_text[0]

    state.threat_model = ThreatModel(raw_text=final)
    yield SSEvent(type=EventType.LOG, stage="threat_model",
                  content=f"Threat model complete ({len(final)} chars, {len(global_ft)} functions)")


async def _reanalyze_variant(
    entry_interfaces, sc_am, arch, base_ft, llm, temp
) -> dict[str, list[dict]]:
    """Re-analyze chains at higher temperature to produce a variant threat map."""
    variant: dict[str, list[dict]] = {}
    for ei in entry_interfaces:
        chain = ei.get("invocation_chain", [])
        if not chain:
            continue
        existing = {f: variant.get(f, []) for f in chain if f in variant}
        existing_text = json.dumps(existing, indent=2, ensure_ascii=False) or "(none)"
        p = TM_ANALYZE_CHAIN_PROMPT.format(
            sc_am=sc_am,
            interface=ei.get("interface", ""),
            chain=" → ".join(chain),
            arch_excerpt=arch,
            existing_threats=existing_text,
        )
        try:
            resp = await llm.chat([{"role": "user", "content": p}],
                                  max_tokens=4096, temperature=temp, json_mode=True)
            data = _extract_json(resp)
            for entry in data.get("function_threats", []):
                fn = entry.get("function", "")
                ts = entry.get("threats", [])
                if fn and ts:
                    if fn not in variant:
                        variant[fn] = ts
                    else:
                        cwes = {t.get("cwe", "") for t in variant[fn]}
                        for t in ts:
                            if t.get("cwe", "") not in cwes:
                                variant[fn].append(t)
                                cwes.add(t.get("cwe", ""))
        except Exception:
            pass
    return variant if variant else base_ft


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                    break
    return {}
